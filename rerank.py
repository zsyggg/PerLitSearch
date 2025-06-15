# rerank.py - Refactored for selectable input, including profile_and_query
import gc
import json
import logging
import os
import torch
import argparse
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict # 确保导入 defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

try:
    from utils import get_config, logger
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('Rerank_Fallback')
    logger.error(f"Failed to import from utils: {e}.")
    class DummyConfig:
        device="cpu"; reranker_path=None; dataset_name="unknown";
        reranker_type="jina"; length_suffix="_L300";
        retrieved_results_path="results/unknown/retrieved.jsonl";
        initial_top_k=100; final_top_k=10; batch_size=8; reranker_max_length=512;
        dataset_type="unknown"; local_model_max_tokens=512;
        test_query_limit: Optional[int] = None
        two_pass_rerank = False
        intermediate_top_k_two_pass = 20
        rerank_input_type = "profile_and_query" 
        
        results_dir = "./results"
        _personalized_queries_base = "personalized_queries"
        _final_results_base = "ranked"
        
        @property
        def personalized_queries_path(self):
            len_sfx = getattr(self, 'length_suffix', '_L300')
            return os.path.join(self.results_dir, self.dataset_name, f"{self._personalized_queries_base}{len_sfx}.jsonl")

        @property
        def final_results_path(self):
            input_type_sfx = f"_{getattr(self, 'rerank_input_type', 'profile_and_query')}"
            if getattr(self, 'rerank_input_type') == "profile_and_query": input_type_sfx = "_profileQuery"
            elif getattr(self, 'rerank_input_type') == "profile_only": input_type_sfx = "_profileOnly"
            elif getattr(self, 'rerank_input_type') == "query_only": input_type_sfx = "_queryOnly"

            type_suffix = f"_{getattr(self, 'reranker_type', 'jina')}"
            k_suffix = f"_top{getattr(self, 'final_top_k', 10)}"
            len_sfx = getattr(self, 'length_suffix', '_L300')
            two_pass_sfx = "_2pass" if getattr(self, 'two_pass_rerank', False) else ""
            base_filename = f"{self._final_results_base}{type_suffix}{input_type_sfx}{two_pass_sfx}"
            return os.path.join(self.results_dir, self.dataset_name, f"{base_filename}{len_sfx}{k_suffix}.jsonl")

        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()

from tqdm import tqdm

def load_personalized_features(personalized_queries_path_with_suffix: str) -> Dict[str, Dict[str, Any]]:
    features_data = {}
    if not os.path.exists(personalized_queries_path_with_suffix):
        logger.warning(f"个性化特征文件未找到: {personalized_queries_path_with_suffix}。")
        return features_data
    try:
        with open(personalized_queries_path_with_suffix, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    p_text = data.get("personalized_features", "")
                    if query_id:
                        features_data[query_id] = {"personalized_features": p_text if "错误:" not in p_text else ""}
                except json.JSONDecodeError:
                    logger.warning(f"跳过 {personalized_queries_path_with_suffix} 中的无效JSON")
        logger.info(f"从 {personalized_queries_path_with_suffix} 加载了 {len(features_data)} 个查询的画像")
    except Exception as e:
        logger.error(f"加载 {personalized_queries_path_with_suffix} 时出错: {e}")
    return features_data

def load_retrieved_results(retrieved_results_path: str) -> Dict[str, Dict]:
    retrieved_data = {}
    if not os.path.exists(retrieved_results_path):
        logger.error(f"检索结果文件未找到: {retrieved_results_path}"); return retrieved_data
    try:
        with open(retrieved_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    if query_id:
                        retrieved_data[query_id] = {
                            "query": data.get("query", ""),
                            "results": data.get("results", [])
                        }
                except json.JSONDecodeError:
                    logger.warning(f"跳过 {retrieved_results_path} 中的无效JSON")
    except Exception as e:
        logger.error(f"加载 {retrieved_results_path} 时出错: {e}")
    return retrieved_data

class RerankerPromptFormatter:
    def _get_doc_content(self, document_text_dict: Dict[str, Any]) -> str:
        title = document_text_dict.get("title", "") or ""
        text = document_text_dict.get("text", "") or ""
        full_paper = document_text_dict.get("full_paper", "") or ""
        doc_parts = [str(p) for p in [title, text, full_paper] if p]
        return " ".join(doc_parts).strip().replace("\n", " ")

    def format_input(self,
                     input_text: str,
                     document_text_dict: Dict[str, Any],
                     reranker_type: str,
                     input_mode: str = "query_only"):
        
        doc_content = self._get_doc_content(document_text_dict)
        
        if input_mode == "query_only":
            query_label = "Query"
            instruction_text = (
                "Your task is to assess the relevance of a document to a given query. "
                "Evaluate how well the document answers or relates to the query."
            )
        elif input_mode == "profile_only":
            query_label = "User Research Profile"
            instruction_text = (
                "Your task is to assess the relevance of a document to a user's research profile. "
                "The profile describes their research interests and focus areas. "
                "Evaluate how well the document aligns with these interests."
            )
        elif input_mode == "profile_and_query":
            query_label = "Combined Context (Profile and Query)" 
            instruction_text = (
                "Your task is to assess the relevance of a document to a user's research profile AND their current query. "
                "Evaluate how well the document aligns with both the overall research interests and the specific query."
            )
        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        if reranker_type == "jina":
            if input_mode == "profile_and_query":
                return (input_text.strip(), doc_content)
            else:
                return (f"{query_label}: {input_text}".strip(), doc_content)
        
        elif reranker_type == "minicpm":
            if input_mode == "profile_and_query":
                 formatted_text = (
                    f"<s>Instruction: {instruction_text}\n\n"
                    f"{input_text}\n\n" 
                    f"Document: {doc_content}</s>"
                )
            else:
                formatted_text = (
                    f"<s>Instruction: {instruction_text}\n\n"
                    f"{query_label}: {input_text}\n\n"
                    f"Document: {doc_content}</s>"
                )
            return formatted_text
        
        else: 
            if input_mode == "profile_and_query":
                return f"{input_text}\nDocument: {doc_content}"
            else:
                return f"{query_label}: {input_text}\nDocument: {doc_content}"


def batch_compute_scores(model, tokenizer, inputs: List[Any], dev: str, r_type: str, max_len: int) -> np.ndarray:
    if not inputs: return np.array([])
    try:
        with torch.no_grad():
            if r_type == "jina":
                scores = model.compute_score(inputs, max_length=max_len) # type: ignore
            else:
                tokenized = tokenizer(inputs, padding=True, truncation=True,
                                    return_tensors='pt', max_length=max_len).to(dev)
                outputs = model(**tokenized, return_dict=True)
                if outputs.logits.shape[-1] > 1: # type: ignore
                    scores = torch.softmax(outputs.logits, dim=-1)[:, 1].float().cpu().numpy() # type: ignore
                else:
                    scores = outputs.logits.view(-1).float().cpu().numpy() # type: ignore
        return np.array(scores)
    except Exception as e:
        logger.error(f"批处理评分错误 (reranker_type: {r_type}): {e}", exc_info=True)
        return np.zeros(len(inputs))


def rerank_documents_for_input_type(model, tokenizer,
                                   input_text_for_reranking: str,
                                   docs: List[Dict[str, Any]],
                                   dev: str,
                                   r_type: str,
                                   b_size: int,
                                   max_len: int,
                                   input_mode: str):
    if not docs: return []
    if not input_text_for_reranking and input_mode != "query_only": 
        logger.warning(f"用于重排的输入文本 (mode: {input_mode}) 为空，所有文档得分将为0。")
        return [{"text_id":doc.get("text_id",""), "title":doc.get("title",""), "text":doc.get("text",""), "score":0.0, **({"full_paper": doc["full_paper"]} if "full_paper" in doc and doc.get("full_paper") else {})} for doc in docs]

    formatter = RerankerPromptFormatter()
    all_formatted_inputs = []
    for doc in docs:
        fmt_input = formatter.format_input(input_text_for_reranking, doc, r_type, input_mode)
        all_formatted_inputs.append(fmt_input)
    
    all_scores = []
    logger.debug(f"计算 {input_mode} 相关性分数 (共{len(docs)}个文档)")
    for i in range(0, len(all_formatted_inputs), b_size):
        batch_inputs = all_formatted_inputs[i:i+b_size]
        scores_batch = batch_compute_scores(model, tokenizer, batch_inputs,
                                            dev, r_type, max_len)
        all_scores.extend(scores_batch)
    
    all_scores_np = np.array(all_scores)
    
    reranked_docs = []
    for i, doc in enumerate(docs):
        res_doc = doc.copy()
        res_doc["score"] = float(all_scores_np[i]) if i < len(all_scores_np) else 0.0
        reranked_docs.append(res_doc)
    
    return reranked_docs


def get_model_and_tokenizer(config, reranker_type):
    DEFAULT_RERANKER_PATHS = {
        "gte": "/workspace/.cache/modelscope/hub/iic/gte_passage-ranking_multilingual-base",
        "jina": "/workspace/.cache/modelscope/hub/jinaai/jina-reranker-v2-base-multilingual",
        "minicpm": "/workspace/.cache/modelscope/hub/OpenBMB/MiniCPM-Reranker-Light"
    }
    model_path = config.reranker_path
    if not model_path and reranker_type in DEFAULT_RERANKER_PATHS:
        model_path = DEFAULT_RERANKER_PATHS[reranker_type]
        logger.info(f"reranker_path 未在配置中明确指定，使用 {reranker_type} 的默认路径: {model_path}")
    if not model_path:
        logger.error(f"无法确定 {reranker_type} 的模型路径。请在配置中指定 reranker_path。")
        raise ValueError(f"模型路径未找到 {reranker_type}")

    logger.info(f"正在加载 {reranker_type} 从 {model_path} 到 {config.device}")
    dtype = torch.float16 if torch.cuda.is_available() and "cuda" in str(config.device) else torch.float32
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if reranker_type == 'minicpm' and getattr(config, 'use_flash_attention', False) and torch.cuda.is_available() and "cuda" in str(config.device):
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("为 MiniCPM 启用 Flash Attention 2。")
            else:
                logger.warning("请求了 Flash Attention 2，但当前 PyTorch 版本可能不支持。回退到默认注意力机制。")
        except Exception as fa_e:
            logger.warning(f"尝试为 MiniCPM 启用 Flash Attention 2 时出错: {fa_e}。回退到默认注意力机制。")
    try:
        trust_remote_flag = reranker_type in ["jina", "minicpm"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_flag, padding_side="right")
        if reranker_type == "minicpm" and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs).to(config.device).eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载模型 {model_path} 时出错: {e}", exc_info=True)
        raise


def run_reranking_with_selected_input(config):
    final_output_path = config.final_results_path
    personalized_queries_path_for_run = config.personalized_queries_path
    rerank_input_mode = getattr(config, 'rerank_input_type', 'profile_only')

    logger.info(f"--- 文档重排 (输入模式: {rerank_input_mode}) ---")
    logger.info(f"数据集: {config.dataset_name}, 重排器: {config.reranker_type}")
    if rerank_input_mode == 'profile_only' or rerank_input_mode == 'profile_and_query':
        logger.info(f"个性化画像来源: {personalized_queries_path_for_run}")
    logger.info(f"最终重排结果将保存到: {final_output_path}")
    
    # 打印 test_query_limit 的初始值
    if config.test_query_limit is not None:
        logger.info(f"配置中的 test_query_limit: {config.test_query_limit} (将根据数据集类型解释)")

    retrieved_data = load_retrieved_results(config.retrieved_results_path)
    all_p_data = {}
    if rerank_input_mode == 'profile_only' or rerank_input_mode == 'profile_and_query':
        all_p_data = load_personalized_features(personalized_queries_path_for_run)
        if not all_p_data:
             logger.error(f"请求了 '{rerank_input_mode}' 模式但未加载任何画像特征。退出。")
             return

    if not retrieved_data:
        logger.error("没有检索结果。退出。")
        return

    try:
        model, tokenizer = get_model_and_tokenizer(config, config.reranker_type)
    except Exception as e:
        logger.error(f"加载重排器模型失败: {e}。退出。")
        return

    # 初始的待处理查询列表 (来自检索结果)
    queries_to_process = list(retrieved_data.keys())

    # 如果需要画像，则筛选出同时存在于检索结果和画像文件中的查询
    if rerank_input_mode == 'profile_only' or rerank_input_mode == 'profile_and_query':
        queries_with_profiles = set(all_p_data.keys())
        queries_to_process = [qid for qid in queries_to_process if qid in queries_with_profiles]
        logger.info(f"'{rerank_input_mode}' 模式: 初步筛选后，有 {len(queries_to_process)} 个查询同时存在于检索和画像数据中。")
    
    # --- START: 修改 test_query_limit 应用逻辑 ---
    if config.test_query_limit is not None and config.test_query_limit > 0:
        logger.info(f"应用 test_query_limit: {config.test_query_limit}")
        if config.dataset_type == "medcorpus":
            logger.info(f"MedCorpus 测试模式: 将限制处理前 {config.test_query_limit} 个主题 (topics)。")
            
            topic_to_qids = defaultdict(list)
            for qid in queries_to_process:
                try:
                    topic_id = '_'.join(qid.split('_')[:-1]) if '_' in qid else qid 
                    topic_to_qids[topic_id].append(qid)
                except Exception as e:
                    logger.warning(f"无法从查询ID '{qid}' 中提取主题ID: {e}。该查询可能不会被正确地按主题限制。")
                    topic_to_qids[qid].append(qid) # Fallback: treat as unique topic

            # 按主题ID排序以确保一致性
            sorted_unique_topic_ids = sorted(list(topic_to_qids.keys()))
            
            if len(sorted_unique_topic_ids) > config.test_query_limit:
                topics_to_keep = set(sorted_unique_topic_ids[:config.test_query_limit])
                limited_queries = []
                for topic_id_to_keep in topics_to_keep:
                    limited_queries.extend(topic_to_qids[topic_id_to_keep])
                
                queries_to_process = sorted(limited_queries) 
                logger.info(f"MedCorpus: 由于 test_query_limit={config.test_query_limit}, "
                            f"已选择 {len(topics_to_keep)} 个主题。处理的查询总数: {len(queries_to_process)}。")
            else:
                logger.info(f"MedCorpus: test_query_limit ({config.test_query_limit}) 大于或等于唯一主题数 ({len(sorted_unique_topic_ids)})。"
                            f"将处理所有已筛选的查询 ({len(queries_to_process)})。")
        
        else: # For LitSearch or other single-turn datasets
            if len(queries_to_process) > config.test_query_limit:
                logger.info(f"{config.dataset_name} 测试模式: 将查询数量限制为前 {config.test_query_limit} 个。")
                queries_to_process = queries_to_process[:config.test_query_limit]
            else:
                logger.info(f"{config.dataset_name}: test_query_limit ({config.test_query_limit}) 大于或等于查询数。"
                            f"将处理所有已筛选的查询 ({len(queries_to_process)})。")
    # --- END: 修改 test_query_limit 应用逻辑 ---

    if not queries_to_process:
        logger.warning("没有查询需要处理 (在应用 test_query_limit 后)。退出重排。")
        return

    logger.info(f"最终将为 {len(queries_to_process)} 个查询执行重排。")

    final_data_list = []
    
    if config.two_pass_rerank:
        logger.info(f"启用两阶段重排: 第一阶段保留top-{config.intermediate_top_k_two_pass}，第二阶段保留top-{config.final_top_k}")
    
    for qid in tqdm(queries_to_process, desc=f"重排查询 (模式: {rerank_input_mode})"):
        if qid not in retrieved_data:
            logger.warning(f"查询ID {qid} 在待处理列表中，但未在 retrieved_data 中找到。跳过。")
            continue
        
        q_info = retrieved_data[qid]
        original_q_text = q_info["query"]
        cand_docs = q_info["results"][:config.initial_top_k]

        input_text_for_this_rerank = ""
        profile_text_for_this_rerank = ""

        if rerank_input_mode == 'profile_only' or rerank_input_mode == 'profile_and_query':
            p_data_q = all_p_data.get(qid, {})
            profile_text_for_this_rerank = p_data_q.get("personalized_features", "")
            if not profile_text_for_this_rerank:
                logger.warning(f"查询 {qid} ({rerank_input_mode} 模式): 未找到有效的个性化画像文本，跳过此查询的重排。")
                # 如果没有画像，但模式需要画像，则跳过此查询
                continue 
        
        if rerank_input_mode == 'profile_only':
            input_text_for_this_rerank = profile_text_for_this_rerank
        elif rerank_input_mode == 'query_only':
            input_text_for_this_rerank = original_q_text
        elif rerank_input_mode == 'profile_and_query':
            input_text_for_this_rerank = f"User Research Profile: {profile_text_for_this_rerank}\n\nNew Query: {original_q_text}"
            # logger.debug(f"Combined input for QID {qid}: {input_text_for_this_rerank[:200]}...") 
        else:
            logger.error(f"未知的 rerank_input_mode: {rerank_input_mode}。跳过查询 {qid}。")
            continue

        max_len = getattr(config, 'reranker_max_length', 512)

        ranked_docs_output = rerank_documents_for_input_type(
            model=model, tokenizer=tokenizer,
            input_text_for_reranking=input_text_for_this_rerank,
            docs=cand_docs, dev=config.device,
            r_type=config.reranker_type, b_size=config.batch_size,
            max_len=max_len, input_mode=rerank_input_mode
        )
        
        final_ranked_docs_for_query = sorted(ranked_docs_output, key=lambda x: x['score'], reverse=True)
        
        if config.two_pass_rerank and len(final_ranked_docs_for_query) > config.intermediate_top_k_two_pass:
            intermediate_docs = final_ranked_docs_for_query[:config.intermediate_top_k_two_pass]
            second_pass_ranked = rerank_documents_for_input_type(
                model=model, tokenizer=tokenizer,
                input_text_for_reranking=input_text_for_this_rerank,
                docs=intermediate_docs, dev=config.device,
                r_type=config.reranker_type, b_size=config.batch_size, max_len=max_len,
                input_mode=rerank_input_mode
            )
            final_ranked_docs_for_query = sorted(second_pass_ranked, key=lambda x: x['score'], reverse=True)
        
        final_ranked_docs_for_query = final_ranked_docs_for_query[:config.final_top_k]
        
        output_entry = {
            "query_id": qid, "query": original_q_text,
            "rerank_mode_used": rerank_input_mode,
            "ranked_results": final_ranked_docs_for_query
        }
        if rerank_input_mode == 'profile_only' or rerank_input_mode == 'profile_and_query':
            output_entry["personalized_profile_used_for_rerank"] = profile_text_for_this_rerank
        if rerank_input_mode == 'profile_and_query':
             output_entry["combined_input_text_example_for_rerank"] = input_text_for_this_rerank[:500]

        final_data_list.append(output_entry)

    try:
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        with open(final_output_path, 'w', encoding='utf-8') as fout:
            for data_item in final_data_list:
                fout.write(json.dumps(data_item, ensure_ascii=False) + "\n")
        logger.info(f"重排完成。结果已保存到 {final_output_path}")
    except IOError as e:
        logger.error(f"写入 {final_output_path} 失败: {e}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"--- 文档重排 (模式: {rerank_input_mode}) 完成 ---")


def main():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="PersLitRank 重排器 (可选输入模式)")
        # 保持原有的参数定义不变
        parser.add_argument("--dataset_name", type=str, required=True)
        parser.add_argument("--reranker_type", type=str, default="jina", choices=["gte", "jina", "minicpm"])
        parser.add_argument("--reranker_path", type=str, help="重排器模型的显式路径")
        parser.add_argument("--data_dir", type=str, default="/workspace/PerMed/data")
        parser.add_argument("--results_dir", type=str, default="./results")
        
        parser.add_argument("--rerank_input_type", type=str, default="profile_and_query",
                            choices=["profile_only", "query_only", "profile_and_query"],
                            help="选择重排的输入类型")

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--initial_top_k", type=int, default=100)
        parser.add_argument("--final_top_k", type=int, default=10)
        parser.add_argument("--max_length", type=int, default=512, help="重排器最大序列长度")
        parser.add_argument("--gpu_id", type=int, default=0)
        
        parser.add_argument("--personalized_text_target_length", type=int, default=300,
                            help="个性化画像的目标长度 (用于定位画像文件)")
        
        parser.add_argument("--test_query_limit", type=int, default=None, help="测试时限制处理的查询数量 (MedCorpus下为主题数)")
        parser.add_argument("--use_flash_attention", action="store_true", help="为 MiniCPM 启用 Flash Attention 2")
        parser.add_argument("--two_pass_rerank", action="store_true", help="启用两阶段重排")
        parser.add_argument("--intermediate_top_k_two_pass", type=int, default=20, help="两阶段重排中第一阶段保留的文档数")

        args = parser.parse_args()
        config = get_config()
        config.update(args) # config对象会通过args更新，包括dataset_name和test_query_limit
        
        if hasattr(args, 'max_length') and args.max_length is not None:
             config.reranker_max_length = args.max_length
        # 其他属性更新...
        if hasattr(args, 'rerank_input_type') and args.rerank_input_type is not None: # 确保传递
            config.rerank_input_type = args.rerank_input_type
        if hasattr(args, 'reranker_path') and args.reranker_path:
             config.reranker_path = args.reranker_path
        if hasattr(args, 'use_flash_attention'):
            config.use_flash_attention = args.use_flash_attention
        
        if hasattr(args, 'personalized_text_target_length') and args.personalized_text_target_length is not None:
            config.personalized_text_target_length = args.personalized_text_target_length
            config.length_suffix = f"_L{config.personalized_text_target_length}"
        
        # 确保 dataset_type 在 config 中是最新的
        if hasattr(args, 'dataset_name') and args.dataset_name:
            config.dataset_name = args.dataset_name # 更新数据集名称
            config.dataset_type = config._infer_dataset_type() # 根据新的数据集名称推断类型
            logger.info(f"Config updated: dataset_name='{config.dataset_name}', dataset_type='{config.dataset_type}'")


        run_reranking_with_selected_input(config)
    else: # 当 rerank.py 被 run.py 导入并调用 main 时
        config = get_config()
        # 确保从 run.py 传递过来的 config 已经包含了正确的 dataset_type 和 test_query_limit
        if not hasattr(config, 'reranker_max_length') or config.reranker_max_length is None:
             config.reranker_max_length = getattr(config, 'local_model_max_tokens', 512)
        if not hasattr(config, 'rerank_input_type'): # 如果config没有这个属性，则设置默认值
            config.rerank_input_type = "profile_and_query"
        
        # 确保 dataset_type 在这里也是最新的，以防万一
        if hasattr(config, 'dataset_name') and (not hasattr(config, 'dataset_type') or \
            (config.dataset_name.lower() not in config.dataset_type.lower() if config.dataset_type else True)):
            config.dataset_type = config._infer_dataset_type()
            logger.info(f"Config (imported context): dataset_name='{config.dataset_name}', dataset_type='{config.dataset_type}'")

        run_reranking_with_selected_input(config)

if __name__ == "__main__":
    main()
