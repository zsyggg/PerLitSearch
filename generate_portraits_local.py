# generate_portraits_gemini_async.py
# V12: 修复认知特征匹配问题

import os
import json
import argparse
import time
import logging
import asyncio
from typing import Optional, List, Dict, Tuple
from tqdm.asyncio import tqdm
import google.generativeai as genai
import tiktoken
from asyncio import Semaphore
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 代理设置
PROXY_URL = "http://127.0.0.1:7897"
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL
    logger.info(f"已启用代理设置: {PROXY_URL}")

# Google API Key
API_KEY = "AIzaSyBzh-yq2S_LS1YW33QNmteH9EjlnMqil4c"
genai.configure(api_key=API_KEY)

# 默认模型
DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"

# Tokenizer for approximate token counting
ENCODING = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    return len(ENCODING.encode(text))


# Prompt模板
FULL_PROMPT_TEMPLATE = """Task: Transform the "Current Query" into a concise, highly effective core statement or enhanced query representation.
This output MUST be optimized for a downstream reranking system. It should precisely capture all key entities, technical specifications, methodologies, and the fundamental intent of the original query.
Focus on extreme clarity, precision, and directness. Eliminate conversational phrasing (e.g., 'Are there any...'), interrogative structures, and any redundant verbiage. The output should be a factual statement or a dense keyword-rich phrase.

Current Query: "{query}"

Relevant Contextual Cues (if any, derived from query analysis):
{memory_features}

Instructions for Generating the Core Statement:
1. Identify the absolute essence of the "Current Query". What is the core subject? What are the specific methods, techniques, technologies, or objects of interest? What are the critical constraints, conditions, or desired properties mentioned?
2. Rephrase this essence into a single, coherent, declarative statement or a very compact series of closely related phrases. This statement will serve as the primary input for a relevance ranking model.
3. If the "Current Query" is phrased as a question (e.g., starting with 'Are there any research papers on...', 'What are...', 'How to...'), you MUST transform it into a factual statement about the topic of interest. For example, if the query is 'Are there studies on X using Y technique for Z purpose?', the output should focus on 'Studies on X using Y technique for Z purpose' or 'X: Y technique for Z purpose'.
4. Ensure ALL critical keywords, technical terms, named entities, and specific constraints from the "Current Query" (and "Contextual Cues", if provided) are explicitly present and prominent in the output. Do not omit or oversimplify critical details.
5. The output MUST be machine-understandable and optimized for information retrieval. Avoid ambiguity. Maximize keyword density relevant to the query's core.
6. Strictly limit the output length to between approximately 100 and {max_length} tokens, absolutely not exceeding {max_length} tokens.
7. Output ONLY the generated core statement/enhanced query representation. No introductory phrases (e.g., 'The core statement is:'), no explanations, and no quotation marks around your final output.

Now, generate the core statement for the provided "Current Query":"""


class AsyncGeminiPortraitGenerator:
    def __init__(self, config=None, max_concurrent: int = 15):
        self.config = config or {}
        self.model_name = getattr(config, "gemini_model", DEFAULT_MODEL)
        self.max_length = getattr(config, "personalized_text_max_length", 300)
        self.temperature = getattr(config, "local_model_temperature", 0.4)
        self.generation_attempts = getattr(config, "profile_generation_attempts", 2)

        self.tokenizer = ENCODING

        self.model = genai.GenerativeModel(self.model_name)
        self.semaphore = Semaphore(max_concurrent)
        self.api_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

        logger.info(
            f"异步生成器初始化: 模型={self.model_name}, 最大并发数={max_concurrent}"
        )

    async def generate_portrait_with_retry(
        self,
        query_data: Dict,
        memory_features: List[str],
        cognitive_data: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> Dict:
        """带重试机制的画像生成"""
        query_id = query_data.get("query_id", "")
        query_text = query_data.get("query", "")

        for attempt in range(max_retries):
            try:
                portrait = await self.generate_portrait_async(
                    query_text, memory_features
                )
                if attempt > 0:
                    logger.debug(f"查询 {query_id} 第 {attempt + 1} 次尝试成功")

                # 构建结果
                result = {
                    "query_id": query_id,
                    "query": query_text,
                    "personalized_features": portrait,
                    "tagged_memory_features": memory_features,  # 使用传入的memory_features
                }

                # 优先使用cognitive_data中的信息
                if cognitive_data:
                    if "topic_id" in cognitive_data:
                        result["topic_id"] = cognitive_data["topic_id"]
                    elif "topic_id" in query_data:
                        result["topic_id"] = query_data["topic_id"]

                    if "turn_id" in cognitive_data:
                        result["turn_id"] = cognitive_data["turn_id"]
                    elif "turn_id" in query_data:
                        result["turn_id"] = query_data["turn_id"]
                else:
                    # 如果没有cognitive_data，使用query_data中的信息
                    if "topic_id" in query_data:
                        result["topic_id"] = query_data["topic_id"]
                    if "turn_id" in query_data:
                        result["turn_id"] = query_data["turn_id"]

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"查询 {query_id} 第 {attempt + 1} 次尝试失败，{wait_time}秒后重试: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"查询 {query_id} 所有尝试都失败了")
                    # 返回fallback结果
                    result = {
                        "query_id": query_id,
                        "query": query_text,
                        "personalized_features": self._generate_fallback(query_text),
                        "tagged_memory_features": memory_features,
                    }

                    if cognitive_data:
                        if "topic_id" in cognitive_data:
                            result["topic_id"] = cognitive_data["topic_id"]
                        if "turn_id" in cognitive_data:
                            result["turn_id"] = cognitive_data["turn_id"]
                    else:
                        if "topic_id" in query_data:
                            result["topic_id"] = query_data["topic_id"]
                        if "turn_id" in query_data:
                            result["turn_id"] = query_data["turn_id"]

                    return result

    async def generate_portrait_async(
        self, query: str, memory_features: List[str]
    ) -> str:
        """异步生成画像"""
        async with self.semaphore:
            # 格式化memory features
            if memory_features and len(memory_features) > 0:
                formatted_features = "\n".join(memory_features)
            else:
                formatted_features = (
                    "No specific memory features available for this query."
                )

            # 构建prompt
            prompt = FULL_PROMPT_TEMPLATE.format(
                query=query,
                memory_features=formatted_features,
                max_length=self.max_length,
            )

            candidates = []
            temperatures = (
                [0.3, 0.5] if self.generation_attempts >= 2 else [self.temperature]
            )

            for i, temp in enumerate(temperatures[: self.generation_attempts]):
                try:
                    # 异步调用API
                    response = await self.model.generate_content_async(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temp, max_output_tokens=250, top_p=0.9, top_k=15
                        ),
                        safety_settings=[
                            {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_NONE",
                            },
                            {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_NONE",
                            },
                            {
                                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "threshold": "BLOCK_NONE",
                            },
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_NONE",
                            },
                        ],
                    )

                    self.api_calls += 1

                    if response.parts and response.text:
                        text = response.text.strip()
                        text = self._clean_and_truncate(text)
                        if text:
                            score = self._evaluate_quality(text, query)
                            candidates.append((text, score))
                            self.successful_calls += 1

                except Exception as e:
                    logger.debug(f"生成尝试 {i+1} 失败: {e}")
                    self.failed_calls += 1

                if i < len(temperatures) - 1:
                    await asyncio.sleep(0.5)

            if candidates:
                best_text, best_score = max(candidates, key=lambda x: x[1])
                return best_text
            else:
                return self._generate_fallback(query)

    def _clean_and_truncate(self, text: str) -> str:
        """清理并截断文本"""
        text = text.strip()
        prefixes_to_remove = [
            "Okay, ",
            "Sure, ",
            "Certainly, ",
            "Here's the generated",
            "Here is the generated",
            "Generated Core Statement:",
            "Core Statement:",
            "Enhanced Query Representation:",
            "Output:",
            "The core statement is:",
            "This is the core statement:",
        ]

        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()

        text = text.strip("\"'")

        if "\n" in text:
            text = text.split("\n")[0].strip()

        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > self.max_length:
            if " " in text:
                idx = text.rfind(" ", 0, self.max_length)
                text = text[: idx if idx != -1 else self.max_length].rstrip()
            token_ids = self.tokenizer.encode(text)[: self.max_length - 1]
            text = self.tokenizer.decode(token_ids).strip() + "..."

        if " " in text and not text.endswith((".", "!", "?", "...")):
            text += "."

        return text.strip()

    def _evaluate_quality(self, text: str, query: str) -> float:
        """评估生成质量"""
        if not text or "Error:" in text:
            return 0.0

        scores = {}
        length = len(self.tokenizer.encode(text))
        ideal_min = self.max_length * 0.4
        ideal_max = self.max_length

        if ideal_min <= length <= ideal_max:
            scores["length"] = 1.0
        elif ideal_min * 0.7 <= length < ideal_min:
            scores["length"] = 0.7
        elif length > ideal_max:
            scores["length"] = 0.8
        else:
            scores["length"] = length / ideal_min if ideal_min > 0 else 0.0

        content_score = 0.0
        core_verbs = [
            "define",
            "explain",
            "compare",
            "analyze",
            "list",
            "show",
            "find",
            "identify",
        ]
        if any(verb in text.lower() for verb in core_verbs):
            content_score += 0.3

        technical_indicators = [
            "method",
            "approach",
            "algorithm",
            "model",
            "system",
            "framework",
            "technique",
        ]
        technical_count = sum(1 for ind in technical_indicators if ind in text.lower())
        content_score += min(technical_count * 0.3, 0.6)
        scores["content"] = min(content_score, 1.0)

        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if query_words and text_words:
            overlap = len(query_words & text_words) / len(query_words)
            scores["relevance"] = overlap
        else:
            scores["relevance"] = 0.5

        weights = {"length": 0.3, "content": 0.3, "relevance": 0.4}
        total_score = sum(
            scores.get(key, 0) * weight for key, weight in weights.items()
        )
        return total_score

    def _generate_fallback(self, query: str) -> str:
        """生成fallback"""
        processed = query.lower()
        prefixes = [
            "are there any research papers on ",
            "are there any studies that explore ",
            "are there any resources available for ",
            "what are ",
            "how to ",
            "can you find ",
        ]

        for prefix in prefixes:
            if processed.startswith(prefix):
                processed = processed[len(prefix) :]
                break

        result = processed.capitalize().strip()
        if not result:
            result = f"Inquiry regarding: {query[:120]}."

        if len(result) > self.max_length:
            result = result[: self.max_length - 3] + "..."
        elif " " in result and not result.endswith("."):
            result += "."

        return result


def load_cognitive_features(cognitive_features_path: str) -> Dict[str, Dict]:
    """加载认知特征文件 - 处理各种query_id格式"""
    features_dict = {}
    features_dict_str = {}  # 字符串版本
    features_dict_int = {}  # 整数版本

    if not os.path.exists(cognitive_features_path):
        logger.error(f"认知特征文件不存在: {cognitive_features_path}")
        return features_dict

    with open(cognitive_features_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                query_id = data.get("query_id", "")

                if query_id:
                    # 存储原始版本
                    features_dict[str(query_id)] = data
                    features_dict_str[str(query_id)] = data

                    # 如果可以转换为整数，也存储整数版本
                    try:
                        query_id_int = int(query_id)
                        features_dict_int[query_id_int] = data
                    except (ValueError, TypeError):
                        pass

                    # 调试信息
                    if line_num <= 5:  # 只打印前5条
                        logger.debug(
                            f"加载认知特征 - query_id: {query_id} (type: {type(query_id)}), "
                            f"features: {len(data.get('tagged_memory_features', []))} 条"
                        )

            except json.JSONDecodeError:
                logger.warning(f"跳过第 {line_num} 行的无效JSON")

    # 返回一个支持多种查询方式的字典
    class FlexibleDict(dict):
        def __init__(self, str_dict, int_dict):
            super().__init__(str_dict)
            self._str_dict = str_dict
            self._int_dict = int_dict

        def __getitem__(self, key):
            # 先尝试字符串版本
            if str(key) in self._str_dict:
                return self._str_dict[str(key)]
            # 再尝试整数版本
            try:
                int_key = int(key)
                if int_key in self._int_dict:
                    return self._int_dict[int_key]
            except (ValueError, TypeError):
                pass
            # 都没找到，抛出KeyError
            raise KeyError(key)

        def get(self, key, default=None):
            try:
                return self.__getitem__(key)
            except KeyError:
                return default

        def __contains__(self, key):
            return str(key) in self._str_dict or (
                isinstance(key, int) and key in self._int_dict
            )

    flexible_dict = FlexibleDict(features_dict_str, features_dict_int)
    logger.info(f"从 {cognitive_features_path} 加载了 {len(features_dict)} 条认知特征")

    return flexible_dict


async def process_file_async(
    input_path: str,
    output_path: str,
    cognitive_features_path: str,
    config,
    max_concurrent: int = 15,
    batch_size: int = 50,
):
    """异步处理文件"""
    logger.info(f"异步处理文件: {input_path}")
    logger.info(f"认知特征文件: {cognitive_features_path}")

    # 加载认知特征
    cognitive_features = load_cognitive_features(cognitive_features_path)

    # 加载查询
    queries_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                queries_data.append(data)

                # 调试信息
                if line_num <= 5:
                    query_id = data.get("query_id", "")
                    logger.debug(
                        f"加载查询 - query_id: {query_id} (type: {type(query_id)})"
                    )

            except json.JSONDecodeError:
                logger.warning(f"跳过第 {line_num} 行的无效JSON")

    if not queries_data:
        logger.error("未找到有效查询")
        return

    logger.info(f"准备处理 {len(queries_data)} 个查询")

    # 创建异步生成器
    generator = AsyncGeminiPortraitGenerator(config, max_concurrent)

    # 记录开始时间
    start_time = time.time()

    # 分批处理
    all_results = []
    total_batches = (len(queries_data) + batch_size - 1) // batch_size

    # 统计信息
    queries_with_features = 0
    queries_without_features = 0

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(queries_data))
        batch_queries = queries_data[batch_start:batch_end]

        logger.info(
            f"\n处理批次 {batch_idx + 1}/{total_batches} (查询 {batch_start + 1}-{batch_end})"
        )

        # 创建异步任务
        tasks = []
        for query_data in batch_queries:
            query_id = query_data.get("query_id", "")

            # 获取认知特征
            memory_features = []
            cognitive_data = None

            # 尝试不同的查找方式
            if query_id in cognitive_features:
                cognitive_data = cognitive_features[query_id]
                memory_features = cognitive_data.get("tagged_memory_features", [])
                queries_with_features += 1
                logger.debug(
                    f"找到查询 {query_id} 的认知特征: {len(memory_features)} 条"
                )
            else:
                queries_without_features += 1
                logger.debug(f"未找到查询 {query_id} 的认知特征")

            # 创建任务
            task = generator.generate_portrait_with_retry(
                query_data, memory_features, cognitive_data
            )
            tasks.append(task)

        # 并发执行
        batch_results = await tqdm.gather(*tasks, desc=f"批次 {batch_idx + 1}")
        all_results.extend(batch_results)

        # 批次间短暂休息
        if batch_idx < total_batches - 1:
            await asyncio.sleep(1)

    # 计算总时间
    total_time = time.time() - start_time

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 统计信息
    logger.info(f"\n=== 处理完成 ===")
    logger.info(f"生成了 {len(all_results)} 条画像")
    logger.info(f"有认知特征的查询: {queries_with_features}")
    logger.info(f"无认知特征的查询: {queries_without_features}")
    logger.info(f"总用时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    logger.info(f"平均每个查询: {total_time/len(queries_data):.2f} 秒")
    logger.info(f"API调用次数: {generator.api_calls}")
    logger.info(f"成功: {generator.successful_calls}, 失败: {generator.failed_calls}")

    # 效率对比
    estimated_serial_time = len(queries_data) * 3
    logger.info(f"预计串行时间: {estimated_serial_time:.1f} 秒")
    logger.info(f"加速比: {estimated_serial_time/total_time:.1f}x")


async def main():
    parser = argparse.ArgumentParser(description="使用Gemini API异步生成画像")
    parser.add_argument("--dataset", type=str, default="both")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=250)
    parser.add_argument("--target-length", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./dataset")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--cognitive-dir", type=str, default="./results")
    parser.add_argument("--concurrent", type=int, default=15, help="最大并发数")
    parser.add_argument("--batch", type=int, default=50, help="批次大小")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    # 调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    class SimpleConfig:
        def __init__(self):
            self.gemini_model = args.model
            self.personalized_text_max_length = args.max_length
            self.personalized_text_target_length = args.target_length
            self.local_model_temperature = args.temperature
            self.profile_generation_attempts = args.attempts

    config = SimpleConfig()

    datasets = ["LitSearch", "MedCorpus"] if args.dataset == "both" else [args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in datasets:
        logger.info(f"\n=== 处理 {dataset_name} ===")

        input_path = os.path.join(args.data_dir, dataset_name, "queries.jsonl")
        cognitive_features_path = os.path.join(
            args.cognitive_dir, dataset_name, "cognitive_features_detailed.jsonl"
        )

        output_filename = f"personalized_queries_L{args.target_length}.jsonl"
        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            continue

        if not os.path.exists(cognitive_features_path):
            logger.warning(f"认知特征文件不存在: {cognitive_features_path}")
            logger.warning("将继续处理但不使用memory features")

        if os.path.exists(output_path):
            logger.warning(f"输出文件已存在: {output_path}")
            overwrite = input("覆盖？(y/n): ")
            if overwrite.lower() != "y":
                continue

        try:
            await process_file_async(
                input_path,
                output_path,
                cognitive_features_path,
                config,
                args.concurrent,
                args.batch,
            )
        except Exception as e:
            logger.error(f"处理出错: {e}", exc_info=True)

    logger.info("\n所有任务完成！")


if __name__ == "__main__":
    asyncio.run(main())
