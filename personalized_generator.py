# personalized_generator.py - 移除评分的精简版本
import logging
import re
import gc
import requests
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, GPT2TokenizerFast

try:
    from utils import logger, get_config
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("PersonalizedGenerator_Fallback")
    logger.warning("Could not import logger/get_config from utils, using fallback.")

    class DummyConfig:
        device = "cpu"
        llm_device = "cpu"
        personalized_text_max_length = 300
        # SiliconFlow API配置
        siliconflow_api_key = "sk-klnmpwfrfjowvolpblilseprcfwlalniumwxocgjrrcrtqib"
        siliconflow_api_url = "https://api.siliconflow.cn/v1/chat/completions"
        siliconflow_model = "deepseek-ai/DeepSeek-R1"
        # Ollama API配置
        llm_base_url = "http://172.18.147.77:11434"
        llm_model = "llama3:8b"
        llm_api_type = "ollama"  # 或 "siliconflow"
        local_model_temperature = 0.4
        local_model_top_p = 0.95
        local_model_top_k = 20
        enable_thinking = False
        local_model_max_tokens = 350
        profile_generation_attempts = 1
        use_fixed_seed = True
        llm_seed = 42

        def _update_text_length_constraints(self):
            pass

        def __getattr__(self, name):
            return None

    def get_config():
        return DummyConfig()


try:
    from prompt_templates import DynamicPromptTemplates
except ImportError:
    logger.error("Could not import DynamicPromptTemplates from prompt_templates.py.")

    class DynamicPromptTemplates:
        @staticmethod
        def format_memory_features(features: List[str]) -> str:
            if not features:
                return "No specific memory features available for this query."
            return "\n".join(features)


class PersonalizedGenerator:
    def __init__(self, config=None):
        self.config = config or get_config()

        # API类型选择
        self.api_type = getattr(self.config, "llm_api_type", "ollama")

        if self.api_type == "siliconflow":
            # SiliconFlow API配置
            self.api_url = getattr(
                self.config,
                "siliconflow_api_url",
                "https://api.siliconflow.cn/v1/chat/completions",
            )
            self.api_key = getattr(self.config, "siliconflow_api_key", "")
            self.model_name = getattr(
                self.config, "siliconflow_model", "deepseek-ai/DeepSeek-R1"
            )
            logger.info(
                f"PersonalizedGenerator initialized with SiliconFlow API (Model: {self.model_name})"
            )
        else:
            # Ollama API配置
            self.api_url = getattr(
                self.config, "llm_base_url", "http://172.18.147.77:11434"
            )
            self.model_name = getattr(self.config, "llm_model", "llama3:8b")
            logger.info(
                f"PersonalizedGenerator initialized with Ollama API (Model: {self.model_name})"
            )

        self.max_length = getattr(self.config, "personalized_text_max_length", 300)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(
                f"Could not load tokenizer for {self.model_name}: {e}. Falling back to GPT2 tokenizer"
            )
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        logger.info(
            f"Generator using max text length constraint: {self.max_length} tokens"
        )

        self.local_model_max_tokens = getattr(
            self.config, "local_model_max_tokens", 350
        )
        self.temperature = getattr(self.config, "local_model_temperature", 0.4)
        self.top_p = getattr(self.config, "local_model_top_p", 0.9)
        self.top_k = getattr(self.config, "local_model_top_k", 15)

        self.generation_attempts = 1  # 固定为单次生成
        self.use_fixed_seed = getattr(self.config, "use_fixed_seed", True)
        self.llm_seed = getattr(self.config, "llm_seed", 42)

        logger.info(
            f"Generator params: temp={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, max_new_tokens={self.local_model_max_tokens}"
        )

        self._check_api_connection()

    def _check_api_connection(self):
        """检查API连接"""
        if self.api_type == "siliconflow":
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                test_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                    "temperature": 0.1,
                }
                response = requests.post(
                    self.api_url, json=test_payload, headers=headers, timeout=10
                )
                if response.status_code == 200:
                    logger.info("SiliconFlow API connection successful.")
                else:
                    logger.warning(
                        f"SiliconFlow API responded with status: {response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Could not connect to SiliconFlow API: {e}")
        else:
            try:
                url = f"{self.api_url}/api/tags"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("Ollama API connection successful.")
                else:
                    logger.warning(
                        f"Ollama API responded with status: {response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Could not connect to Ollama API: {e}")

    def _llm_request(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        seed: int = None,
    ) -> str:
        """发送请求到API"""
        if max_tokens is None:
            max_tokens = self.local_model_max_tokens
        if temperature is None:
            temperature = self.temperature

        retry_count = 2
        wait_time = 1

        for attempt in range(retry_count + 1):
            try:
                if self.api_type == "siliconflow":
                    # SiliconFlow API调用
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }

                    payload = {
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert at generating concise, precise research profile descriptions. You MUST follow length constraints strictly.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "enable_thinking": False,
                        "frequency_penalty": 0.0,
                        "n": 1,
                        "response_format": {"type": "text"},
                    }

                    response = requests.post(
                        self.api_url, json=payload, headers=headers, timeout=120
                    )
                    response.raise_for_status()

                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        result = data["choices"][0]["message"]["content"].strip()
                    else:
                        logger.error(f"Unexpected API response format: {data}")
                        return ""
                else:
                    # Ollama API调用
                    url = f"{self.api_url}/api/generate"
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": self.top_p,
                            "top_k": self.top_k,
                            "num_predict": max_tokens,
                            "num_ctx": 4096,
                        },
                    }
                    if seed is not None:
                        payload["options"]["seed"] = seed

                    response = requests.post(url, json=payload, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    result = data.get("response", "").strip()

                if not result:
                    logger.warning(
                        f"LLM returned an empty response for prompt starting with: {prompt[:100]}..."
                    )
                    return ""
                return result

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"LLM request failed (attempt {attempt + 1}/{retry_count + 1}): {e}"
                )
                if attempt < retry_count:
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    return f"Error: API request failed - {str(e)}"
            except Exception as e:
                logger.error(f"Error processing LLM response: {e}")
                return f"Error: Failed to process response - {str(e)}"
        return ""

    def _build_unified_turn1_style_prompt(
        self, query: str, formatted_features: str
    ) -> str:
        """构建强化长度限制的Prompt"""
        # 根据长度选择更具体的指令
        if self.max_length <= 150:
            length_range = "30-150"
            length_emphasis = "extremely concise, ultra-compact"
        elif self.max_length <= 200:
            length_range = "50-200"
            length_emphasis = "very concise, compact"
        elif self.max_length <= 250:
            length_range = "80-250"
            length_emphasis = "concise, focused"
        else:
            length_range = f"100-{self.max_length}"
            length_emphasis = "concise"

        logger.debug(f"Building UNIFIED prompt for query: {query[:70]}...")

        # 强化的长度限制提示
        prompt_lines = [
            f"CRITICAL REQUIREMENT: Output MUST be between {length_range} tokens. This is MANDATORY.",
            "",
            'Task: Transform the "Current Query" into a concise, highly effective core statement or enhanced query representation.',
            "This output MUST be optimized for a downstream reranking system.",
            "",
            f'Current Query: "{query}"',
            "",
        ]

        if (
            formatted_features
            and formatted_features != "No relevant memory features available."
        ):
            prompt_lines.extend(["Relevant Contextual Cues:", formatted_features, ""])

        prompt_lines.extend(
            [
                "STRICT INSTRUCTIONS:",
                f"1. LENGTH CONSTRAINT: Your output MUST be {length_emphasis}, between {length_range} tokens.",
                "2. If your initial response exceeds the limit, you MUST condense it immediately.",
                "3. Focus on the absolute core: key entities, methods, and intent only.",
                "4. Remove ALL unnecessary words: articles (a, the), verbose phrases, explanations.",
                "5. Use keyword-dense formulation. Every word must add value.",
                "6. Transform questions into declarative keyword phrases.",
                "",
                f"REMINDER: Output length MUST be {length_range} tokens. Count carefully.",
                "",
                "Generate the core statement NOW (remember the length limit):",
            ]
        )

        return "\n".join(prompt_lines)

    def generate_personalized_text(
        self,
        query: str,
        memory_results: Dict,
        previous_profile: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> str:
        """生成个性化描述文本（单次生成，无评分）"""
        try:
            tagged_features_list = (
                memory_results.get("tagged_memory_features", [])
                if isinstance(memory_results, dict)
                else []
            )
            formatted_features = DynamicPromptTemplates.format_memory_features(
                tagged_features_list or []
            )

            prompt = self._build_unified_turn1_style_prompt(query, formatted_features)

            # 单次生成
            logger.info(
                f"Generating description for query '{query[:70]}...' (Max length: {self.max_length} tokens)"
            )

            raw_response = self._llm_request(
                prompt,
                temperature=self.temperature,
                seed=self.llm_seed if self.use_fixed_seed else None,
            )

            if raw_response.startswith("Error:"):
                logger.error(f"API call failed: {raw_response}")
                return self._generate_fallback_description(query, formatted_features)

            cleaned_text = self._clean_and_validate_description(raw_response)
            if not cleaned_text:
                logger.warning(
                    f"Cleaned text was empty for query '{query[:70]}...'. Using fallback."
                )
                return self._generate_fallback_description(query, formatted_features)

            token_len = len(
                self.tokenizer.encode(cleaned_text, add_special_tokens=False)
            )
            logger.info(
                f"Generated description for query '{query[:70]}...', length: {token_len} tokens"
            )
            return cleaned_text

        except Exception as e:
            logger.error(
                f"Error generating personalized text for query '{query[:70]}...': {e}",
                exc_info=True,
            )
            return self._generate_fallback_description(query, "")

    def _clean_and_validate_description(self, text: str) -> str:
        """清理和验证描述文本"""
        # 移除常见的前缀
        text = re.sub(
            r"^(Okay, |Sure, |Certainly, |Here('s| is) (the|your) (generated|core) (paragraph|statement|representation|output):?)\s*\n*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(
            r"^(Generated Core Statement|Core Statement|Enhanced Query Representation|Research Profile|Profile|Description|Paragraph|Output)[:\-\s]*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(
            r"^(This research profile|The profile|The descriptive paragraph|The core statement)[:\-\s]*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        text = text.strip("\"'")

        # 移除常见的解释性后缀
        common_postambles_patterns = [
            r"\s*\n\nThis (output|statement|text|response|generation|core statement|summary|paragraph|description) precisely captures.*",
            r"\s*\n\nThis (output|statement|text|response|generation|core statement|summary|paragraph|description) aims to.*",
            r"\s*\n\nThis (output|statement|text|response|generation|core statement|summary|paragraph|description) should help.*",
            r"\s*\n\nI hope this helps.*",
            r"\s*\n\nPlease let me know if you.*",
            r"\s*\n\nThe above (statement|text|paragraph|summary).*",
            r"\s*\n\nKey elements captured include.*",
        ]
        for pattern in common_postambles_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        text = text.strip()
        text = re.sub(r"^\s*[\-\*\+]\s+", "", text)

        if not text.strip():
            return ""

        # 确保非空文本以标点结尾
        if " " in text.strip() and not re.search(r"[.!?]$", text.strip()):
            text = text.strip() + "."

        # 强制长度限制 (按token计)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > self.max_length:
            logger.warning(
                f"Text exceeded max_length ({self.max_length}). Original length: {len(token_ids)} tokens. Truncating."
            )
            token_ids = token_ids[: self.max_length - 1]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if not text.endswith("..."):
                text = text.rstrip(".!?") + "..."

        # 验证是否有实质内容
        if not re.search(r"\b\w{2,}\b", text):
            logger.debug(
                f"Cleaned text '{text}' lacks substantial content after processing."
            )
            return ""
        return text.strip()

    def _generate_fallback_description(self, query: str, features: str) -> str:
        """生成后备描述"""
        logger.warning(f"Generating fallback description for query '{query[:70]}...'.")

        processed_query = query.lower()
        prefixes_to_remove = [
            "are there any research papers on ",
            "are there any studies that explore ",
            "are there any resources available for ",
            "are there any tools or studies that have focused on ",
            "are there papers that propose ",
            "are there studies that combine ",
            "what are ",
            "how to ",
            "can you find ",
        ]
        for prefix in prefixes_to_remove:
            if processed_query.startswith(prefix):
                processed_query = processed_query[len(prefix) :]
                break

        fallback_text = processed_query.capitalize().strip()
        if not fallback_text:
            fallback_text = f"Inquiry regarding: {query[:120]}."

        return self._clean_and_validate_description(fallback_text)

    def generate_personalized_text_batch(self, queries_data: List[Dict]) -> List[str]:
        """批量生成个性化描述文本（无评分）"""
        descriptions = []

        for i, data_item in enumerate(queries_data):
            query_text = data_item.get("query", "")
            if not query_text:
                logger.error(f"Skipping item {i+1} due to missing 'query' field.")
                descriptions.append(
                    self._generate_fallback_description("Missing query", "")
                )
                continue

            mem_res = data_item.get("memory_results", {})

            desc = self.generate_personalized_text(query_text, mem_res)
            if not desc:
                logger.error(
                    f"CRITICAL: desc is empty even after primary fallback for query '{query_text[:70]}...'. Using generic."
                )
                desc = f"Processing query: {query_text[:150]}."

            descriptions.append(desc)

            if (i + 1) % 10 == 0 or (i + 1) == len(queries_data):
                logger.info(f"Batch generation progress: {i+1}/{len(queries_data)}")
            gc.collect()

        logger.info(
            f"Batch generation complete. Generated {len(descriptions)} descriptions."
        )
        return descriptions
