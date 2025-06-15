# run.py - 支持Llama3和SiliconFlow API
import os
import argparse
import logging
import time
import json
from tqdm import tqdm
import gc
import torch
from collections import defaultdict

from utils import get_config, logger, Query

def parse_args():
    parser = argparse.ArgumentParser(description="运行 PersLitRank 系统")
    parser.add_argument("--mode", type=str,
                        choices=["all", "extract_cognitive_features", "generate_narratives", "retrieve", "rerank"],
                        required=True, help="处理模式")

    # 通用参数
    parser.add_argument("--dataset_name", type=str, default="MedCorpus", choices=["MedCorpus", "LitSearch", "CORAL"])
    parser.add_argument("--data_dir", type=str, default="/workspace/PerMed/data", help="基础数据目录")
    parser.add_argument("--results_dir", type=str, default="./results", help="基础结果目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="当前进程的GPU ID")
    parser.add_argument("--batch_size", type=int, help="适用的批处理大小 (用于检索或重排)")
    parser.add_argument("--personalized_text_max_length", type=int, help="个性化画像的最大字符长度")
    parser.add_argument("--personalized_text_target_length", type=int, default=300,
                        help="个性化画像的目标长度 (主要用于文件名后缀)")
    
    # LLM API选择
    parser.add_argument("--llm_api_type", type=str, default="ollama",
                        choices=["ollama", "siliconflow"],
                        help="选择使用的LLM API类型")
    
    # Llama3 API 参数
    parser.add_argument("--llm_base_url", type=str, default="http://172.18.147.77:11434",
                        help="Ollama API 基础URL")
    parser.add_argument("--llm_model", type=str, default="llama3:8b",
                        help="使用的Llama3模型 (如 llama3:8b, llama3.3:72b-32k-context)")
    
    # SiliconFlow API 参数
    parser.add_argument("--siliconflow_api_key", type=str,
                        default="sk-klnmpwfrfjowvolpblilseprcfwlalniumwxocgjrrcrtqib",
                        help="SiliconFlow API密钥")
    parser.add_argument("--siliconflow_model", type=str,
                        default="deepseek-ai/DeepSeek-R1",
                        help="使用的SiliconFlow模型")
    
    # 通用LLM参数
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--top_p", type=float, help="LLM top_p")
    parser.add_argument("--top_k", type=int, help="LLM top_k")
    parser.add_argument("--local_model_max_tokens", type=int, help="LLM为画像生成的最大token数")

    # Reranker 参数
    parser.add_argument("--reranker_type", type=str, choices=["gte", "jina", "minicpm"], help="重排器类型")
    parser.add_argument("--reranker_path", type=str, help="重排器模型的显式路径")
    parser.add_argument("--reranker_max_length", type=int, help="重排器的最大序列长度")
    parser.add_argument("--rerank_input_type", type=str, default="profile_and_query",
                        choices=["profile_only", "query_only", "profile_and_query"],
                        help="选择重排的输入类型")

    # 其他参数
    parser.add_argument("--feature_extractor", type=str, help="特征提取器类型")
    parser.add_argument("--memory_type", type=str, help="内存类型")
    parser.add_argument("--memory_components", type=str, help="要使用的内存组件，逗号分隔")
    parser.add_argument("--conversational", action="store_true", help="是否为对话模式")
    parser.add_argument("--initial_top_k", type=int, help="初始检索召回的文档数")
    parser.add_argument("--final_top_k", type=int, help="重排后最终保留的文档数")
    parser.add_argument("--test_query_limit", type=int, default=None,
                        help="测试模式下限制处理的查询数量 (对MedCorpus是topic/组的数量)")
    parser.add_argument("--use_flash_attention", action="store_true", help="为 MiniCPM 启用 Flash Attention 2")

    return parser.parse_args()

def run_cognitive_feature_extraction_stage(config):
    logger.info(f"--- 阶段1: 认知特征提取 ---")
    logger.info(f"数据集: {config.dataset_name}")
    logger.info(f"详细认知特征输出到: {config.cognitive_features_detailed_path}")
    stage_success = False
    try:
        from cognitive_retrieval import main as cognitive_main
        cognitive_main()
        logger.info(f"--- 阶段1: 认知特征提取完成 ---")
        stage_success = True
    except ImportError:
        logger.error("无法导入 cognitive_retrieval。跳过阶段1。", exc_info=True)
    except Exception as e:
        logger.error(f"阶段1 (认知特征提取) 期间出错: {e}", exc_info=True)
    return stage_success

def run_narrative_generation_stage(config):
    api_type = getattr(config, 'llm_api_type', 'ollama')
    if api_type == 'siliconflow':
        logger.info(f"--- 阶段2: 个性化画像生成 (使用SiliconFlow API - {config.siliconflow_model}) ---")
        logger.info(f"使用 SiliconFlow API: {config.siliconflow_api_url}")
    else:
        logger.info(f"--- 阶段2: 个性化画像生成 (使用Llama3 API - {config.llm_model}) ---")
        logger.info(f"使用 Llama3 API: {config.llm_base_url}")
    
    logger.info(f"从以下位置输入认知特征: {config.cognitive_features_detailed_path}")
    logger.info(f"画像文件输出 (基于目标长度 {config.personalized_text_target_length} 的文件名后缀): {config.personalized_queries_path}")
    logger.info(f"画像实际最大长度由 PersonalizedGenerator 内的 max_length ({getattr(config, 'personalized_text_max_length', 300)}) 控制。")
    
    if config.test_query_limit is not None:
        if config.dataset_type == "medcorpus":
            logger.info(f"测试模式: 将为前 {config.test_query_limit} 个组（topic）生成画像。")
        else:
            logger.info(f"测试模式: 将为前 {config.test_query_limit} 个查询生成画像。")
    
    stage_success = False
    try:
        from personalized_generator import PersonalizedGenerator
    except ImportError:
        logger.error("无法导入 PersonalizedGenerator。无法运行画像生成阶段。", exc_info=True)
        return False

    if not os.path.exists(config.cognitive_features_detailed_path):
        logger.error(f"认知特征文件未找到: {config.cognitive_features_detailed_path}。请先运行 'extract_cognitive_features' 模式。")
        return False

    try:
        narrative_generator = PersonalizedGenerator(config=config)
    except Exception as e:
        logger.error(f"初始化 PersonalizedGenerator 时出错: {e}", exc_info=True)
        return False

    generated_narratives_data = []
    queries_processed_count = 0
    topic_profiles = {}
    
    try:
        cognitive_features_input_lines = []
        with open(config.cognitive_features_detailed_path, 'r', encoding='utf-8') as f_in:
            cognitive_features_input_lines = f_in.readlines()
        
        queries_by_topic = defaultdict(list)
        for line_idx, line in enumerate(cognitive_features_input_lines):
            try:
                data = json.loads(line)
                queries_by_topic[data.get("topic_id", f"unknown_topic_{line_idx}")].append(data)
            except json.JSONDecodeError:
                logger.warning(f"跳过无效的JSON行 (行号 {line_idx+1}): {line.strip()}")
                continue
        
        sorted_topic_ids = sorted(queries_by_topic.keys())
        
        topics_to_process_keys = sorted_topic_ids
        if config.test_query_limit is not None and config.test_query_limit > 0:
            if config.dataset_type == "medcorpus":
                if len(sorted_topic_ids) > config.test_query_limit:
                    logger.info(f"MedCorpus: 限制为前 {config.test_query_limit} 个 topic 进行画像生成。")
                    topics_to_process_keys = sorted_topic_ids[:config.test_query_limit]

        total_queries_to_generate_for = 0
        if config.dataset_type == "medcorpus":
            for topic_id_key in topics_to_process_keys:
                total_queries_to_generate_for += len(queries_by_topic[topic_id_key])
        else:
            for topic_id_key in topics_to_process_keys:
                for _ in queries_by_topic[topic_id_key]:
                    total_queries_to_generate_for +=1
            if config.test_query_limit is not None and config.test_query_limit > 0:
                 total_queries_to_generate_for = min(total_queries_to_generate_for, config.test_query_limit)

        logger.info(f"准备为约 {total_queries_to_generate_for} 个查询条目生成画像。")
        
        current_query_overall_count = 0
        with tqdm(total=total_queries_to_generate_for, desc=f"生成描述性画像") as pbar:
            for topic_id in topics_to_process_keys:
                topic_queries_sorted = sorted(queries_by_topic[topic_id], key=lambda x: x.get("turn_id", 0))
                
                for cognitive_data in topic_queries_sorted:
                    if config.dataset_type != "medcorpus" and \
                       config.test_query_limit is not None and \
                       current_query_overall_count >= config.test_query_limit:
                        logger.info(f"{config.dataset_name}: 已达到 test_query_limit ({config.test_query_limit})，停止生成。")
                        break

                    query_id = cognitive_data.get("query_id")
                    original_query_text = cognitive_data.get("query")
                    memory_features_for_generator = {"tagged_memory_features": cognitive_data.get("tagged_memory_features", [])}
                    current_turn_id = cognitive_data.get("turn_id")

                    if not query_id or not original_query_text:
                        logger.warning(f"由于数据缺失跳过条目: {cognitive_data.get('query_id', 'N/A')}")
                        continue

                    previous_profile_for_turn = topic_profiles.get(topic_id)
                    narrative_text = narrative_generator.generate_personalized_text(
                        query=original_query_text,
                        memory_results=memory_features_for_generator,
                        previous_profile=previous_profile_for_turn,
                        turn_id=current_turn_id
                    )
                    topic_profiles[topic_id] = narrative_text
                    
                    output_entry = {
                        "query_id": query_id, "topic_id": topic_id, "turn_id": current_turn_id,
                        "query": original_query_text, "personalized_features": narrative_text,
                        "tagged_memory_features": cognitive_data.get("tagged_memory_features", [])
                    }
                    generated_narratives_data.append(output_entry)
                    queries_processed_count += 1
                    current_query_overall_count +=1
                    pbar.update(1)

                    if queries_processed_count % 50 == 0:
                        logger.info(f"已为 {queries_processed_count}/{total_queries_to_generate_for} 个查询生成画像...")
                
                if config.dataset_type != "medcorpus" and \
                   config.test_query_limit is not None and \
                   current_query_overall_count >= config.test_query_limit:
                    break

        output_narrative_file = config.personalized_queries_path
        os.makedirs(os.path.dirname(output_narrative_file), exist_ok=True)
        with open(output_narrative_file, 'w', encoding='utf-8') as f_out:
            for entry in generated_narratives_data:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"已将 {len(generated_narratives_data)} 条个性化画像保存到 {output_narrative_file}")
        stage_success = True

    except Exception as e:
        logger.error(f"在画像生成阶段发生错误: {e}", exc_info=True)
    finally:
        if 'narrative_generator' in locals() and narrative_generator is not None:
            del narrative_generator
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    logger.info(f"--- 阶段2: 个性化画像生成 {'完成' if stage_success else '失败'} ---")
    return stage_success

def main():
    args = parse_args()
    config = get_config()
    config.update(args)

    start_time = time.time()
    logger.info(f"--- PersLitRank 运行 ---")
    logger.info(f"模式: {args.mode}, 数据集: {config.dataset_name} (类型: {config.dataset_type})")
    logger.info(f"当前进程 GPU: {config.device}")
    
    if args.mode == "generate_narratives" or args.mode == "all":
        logger.info(f"画像文件名后缀将基于目标长度: {config.personalized_text_target_length} -> 长度后缀: '{config.length_suffix}'")
        logger.info(f"画像生成器内部实际最大长度约束: {getattr(config, 'personalized_text_max_length', 300)} 字符。")
        logger.info(f"个性化画像 (阶段2) 输出 (文件名基于 L{config.personalized_text_target_length}): {config.personalized_queries_path}")
        
        api_type = getattr(config, 'llm_api_type', 'ollama')
        if api_type == 'siliconflow':
            logger.info(f"使用 SiliconFlow API: {config.siliconflow_api_url}, 模型: {config.siliconflow_model}")
        else:
            logger.info(f"使用 Llama3 API: {config.llm_base_url}, 模型: {config.llm_model}")

    if config.test_query_limit is not None:
        logger.info(f"*** 测试模式已激活: 后续相关阶段将限制处理前 {config.test_query_limit} 个查询/主题。 ***")
    
    if args.mode == "extract_cognitive_features" or args.mode == "all":
        logger.info(f"认知特征 (阶段1) 输出: {config.cognitive_features_detailed_path}")
    
    if args.mode == "retrieve" or args.mode == "all":
        logger.info(f"检索到的文档输出: {config.retrieved_results_path}")

    if args.mode == "rerank" or args.mode == "all":
        logger.info(f"重排器: {config.reranker_type}, 重排输入模式: {config.rerank_input_type}")
        rerank_input_narrative_path = config.personalized_queries_path
        if config.rerank_input_type == "profile_only" or config.rerank_input_type == "profile_and_query":
            logger.info(f"个性化画像 (重排输入, 文件名基于 L{config.personalized_text_target_length}): {rerank_input_narrative_path}")
        logger.info(f"重排输出 (文件名基于 L{config.personalized_text_target_length}, 输入模式: {config.rerank_input_type}, TopK: {config.final_top_k}): {config.final_results_path}")

    # --- 模式执行 ---
    if args.mode == "extract_cognitive_features":
        run_cognitive_feature_extraction_stage(config)

    elif args.mode == "generate_narratives":
        run_narrative_generation_stage(config)

    elif args.mode == "retrieve":
        logger.info("--- 执行: 初始文档检索 ---")
        try:
            from feature_retrieval import main as retrieval_main
            retrieval_main()
            logger.info("--- 初始文档检索完成 ---")
        except ImportError:
            logger.error("无法导入 feature_retrieval。", exc_info=True)
        except Exception as e:
            logger.error(f"检索期间出错: {e}", exc_info=True)

    elif args.mode == "rerank":
        logger.info("--- 执行: 文档重排 ---")
        
        retrieved_exists = os.path.exists(config.retrieved_results_path)
        narratives_needed = (config.rerank_input_type == 'profile_only' or config.rerank_input_type == 'profile_and_query')
        path_to_narrative_for_rerank = config.personalized_queries_path
        narratives_exist = os.path.exists(path_to_narrative_for_rerank)

        if not retrieved_exists:
             logger.error(f"重排输入缺失: {config.retrieved_results_path}。")
        elif narratives_needed and not narratives_exist:
             logger.error(f"用于重排的画像文件缺失 (模式: {config.rerank_input_type}): {path_to_narrative_for_rerank}。请使用目标长度 {config.personalized_text_target_length} 运行 'generate_narratives'。")
        
        if retrieved_exists and (not narratives_needed or narratives_exist):
            try:
                from rerank import main as rerank_main
                rerank_main()
                logger.info("--- 文档重排完成 ---")
            except ImportError:
                logger.error("无法导入 rerank。", exc_info=True)
            except Exception as e:
                logger.error(f"重排期间出错: {e}", exc_info=True)
        else:
            logger.warning("由于输入缺失，跳过重排。")
            
    elif args.mode == "all":
        logger.info("--- 执行所有阶段 ---")
        s1_ok = run_cognitive_feature_extraction_stage(config)
        
        s_retrieve_ok = False
        if s1_ok:
            logger.info("--- ALL 模式: 运行初始文档检索 ---")
            try:
                from feature_retrieval import main as retrieval_main
                retrieval_main()
                logger.info("--- ALL 模式: 初始文档检索完成 ---")
                s_retrieve_ok = True
            except Exception as e:
                logger.error(f"ALL 模式检索期间出错: {e}", exc_info=True)
        else:
            logger.error("由于阶段1失败，在ALL模式下跳过检索。")

        s2_ok = False
        if s1_ok:
            s2_ok = run_narrative_generation_stage(config)
        else:
            logger.error("由于阶段1失败，在ALL模式下跳过画像生成。")
        
        narratives_needed_for_all_rerank = (config.rerank_input_type == 'profile_only' or config.rerank_input_type == 'profile_and_query')
        if s_retrieve_ok and (not narratives_needed_for_all_rerank or s2_ok) :
            logger.info("--- ALL 模式: 运行文档重排 ---")
            try:
                from rerank import main as rerank_main
                rerank_main()
                logger.info("--- ALL 模式: 文档重排完成 ---")
            except Exception as e:
                logger.error(f"ALL 模式重排期间出错: {e}", exc_info=True)
        else:
            logger.warning("由于输入缺失或先前阶段失败，在ALL模式下跳过重排。")

    end_time = time.time()
    logger.info(f"--- PersLitRank 处理模式: {args.mode} 已完成 ---")
    logger.info(f"总执行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
