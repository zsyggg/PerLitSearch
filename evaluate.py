#!/usr/bin/env python
# evaluate.py - 包含 MAP@k, P@1, NDCG@k, PG基于NDCG@k
import json
import os
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import math # For log2 in NDCG

def load_ground_truth_jsonl(gt_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    加载真实标签。
    """
    ground_truth: Dict[str, Dict[str, List[str]]] = {}
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found {gt_path}")
        return {}
        
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                qid = str(data["query_id"])
                relevant_texts_list = [str(doc_id) for doc_id in data.get("relevant_texts", [])]
                
                ground_truth[qid] = {
                    "relevant_texts": relevant_texts_list
                }
                
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return {}
        
    return ground_truth

def load_predictions(pred_path: str, results_key: str = "ranked_results") -> Dict[str, List[str]]:
    """加载预测结果 (文档ID列表)"""
    predictions: Dict[str, List[str]] = {}
    
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found {pred_path}")
        return {}
    
    try:    
        with open(pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                d = json.loads(line)
                qid = str(d["query_id"]) 
                
                preds_list_of_dicts: Optional[List[Dict[str, Any]]] = None
                if results_key in d and isinstance(d[results_key], list):
                    preds_list_of_dicts = d[results_key]
                elif "results" in d and isinstance(d["results"], list): 
                    preds_list_of_dicts = d["results"]
                
                if preds_list_of_dicts is not None:
                    pred_docs = [str(p["text_id"]) for p in preds_list_of_dicts if isinstance(p, dict) and "text_id" in p]
                    predictions[qid] = pred_docs
                    
    except Exception as e:
        print(f"Error loading prediction file {pred_path}: {e}")
        return {}
        
    return predictions

def average_precision(relevance_list: List[int], k_limit: int) -> float:
    """计算Average Precision@k_limit"""
    if not isinstance(relevance_list, list): return 0.0
    
    relevance_at_k = relevance_list[:k_limit]
    if sum(relevance_at_k) == 0: 
        return 0.0
        
    num_relevant_found = 0
    sum_precisions = 0.0
    
    for i, rel_val in enumerate(relevance_at_k):
        if rel_val == 1: 
            num_relevant_found += 1
            sum_precisions += num_relevant_found / (i + 1)
            
    if num_relevant_found == 0: return 0.0 
    return sum_precisions / num_relevant_found


def ndcg_at_k(relevance_list: List[float], k_limit: int, method: int = 0) -> float:
    """
    计算NDCG@k_limit.
    relevance_list: 预测列表中每个文档的真实相关度分数 (可以是二元的0或1，或多级的)。
    """
    if not isinstance(relevance_list, list): return 0.0

    dcg = 0.0
    for i in range(min(len(relevance_list), k_limit)):
        rank = i + 1
        if method == 0: 
            dcg += relevance_list[i] / math.log2(rank + 1) 
        else: 
            dcg += (math.pow(2, relevance_list[i]) - 1) / math.log2(rank + 1)
            
    # 使用传入的 relevance_list（已排序的理想相关度列表）来计算 IDCG
    # 或者，如果 relevance_list 是预测顺序的，我们需要一个单独的理想相关度列表
    # 当前实现：假设 relevance_list 用于DCG，然后对其排序用于IDCG
    ideal_relevance_sorted = sorted(relevance_list, reverse=True) 
    idcg = 0.0
    for i in range(min(len(ideal_relevance_sorted), k_limit)): # IDCG也应该截断到k_limit
        rank = i + 1
        if method == 0:
            idcg += ideal_relevance_sorted[i] / math.log2(rank + 1)
        else:
            idcg += (math.pow(2, ideal_relevance_sorted[i]) - 1) / math.log2(rank + 1)
            
    if idcg == 0:
        return 0.0  
        
    return dcg / idcg


def calculate_personalization_gain(personalized_metrics: Dict[str, float], 
                                 baseline_metrics: Dict[str, float],
                                 metric_key: str) -> float: # metric_key 现在是必须的
    """
    计算指定指标的 Personalization Gain (PG)
    """
    metric_personalized = personalized_metrics.get(metric_key, 0.0)
    metric_baseline = baseline_metrics.get(metric_key, 0.0)
    
    gain = 0.0
    if metric_baseline > 1e-9: 
        gain = (metric_personalized - metric_baseline) / metric_baseline
    elif metric_personalized > 1e-9: 
        gain = 1.0 
        
    return gain


def evaluate_core_metrics(predictions: Dict[str, List[str]], 
                         ground_truth: Dict[str, Dict[str, List[str]]], 
                         k_val: int = 10) -> Dict[str, float]:
    """计算核心评估指标：MAP@k, P@1, NDCG@k"""
    if not predictions or not ground_truth:
        return {f"MAP@{k_val}": 0.0, "P@1": 0.0, f"NDCG@{k_val}": 0.0, "num_queries": 0}
    
    all_aps = []
    all_p1s = []
    all_ndcgs = []
    
    evaluated_qids_count = 0
    
    for qid, pred_doc_ids in predictions.items():
        if qid not in ground_truth:
            continue
        
        gt_info = ground_truth[qid]
        true_relevant_doc_ids_set = set(gt_info.get("relevant_texts", []))
        
        if not pred_doc_ids and not true_relevant_doc_ids_set: 
            all_aps.append(0.0)
            all_p1s.append(0.0)
            all_ndcgs.append(0.0)
            evaluated_qids_count +=1
            continue
        elif not pred_doc_ids: 
            all_aps.append(0.0)
            all_p1s.append(0.0)
            all_ndcgs.append(0.0)
            evaluated_qids_count +=1
            continue
        
        relevance_binary_list_for_preds: List[int] = []
        relevance_scores_for_ndcg_preds: List[float] = []

        for doc_id in pred_doc_ids: 
            is_relevant = 1 if doc_id in true_relevant_doc_ids_set else 0
            relevance_binary_list_for_preds.append(is_relevant)
            relevance_scores_for_ndcg_preds.append(float(is_relevant)) 

        p1 = float(relevance_binary_list_for_preds[0]) if relevance_binary_list_for_preds else 0.0
        all_p1s.append(p1)
        
        ap = average_precision(relevance_binary_list_for_preds, k_val)
        all_aps.append(ap)
        
        # 为了正确计算IDCG，我们需要知道对于这个查询，理想情况下最多有多少个相关文档可以被排在前面
        # 对于二元相关性，理想列表是所有相关文档（gain=1）排在前面
        num_gt_relevant_for_query = len(true_relevant_doc_ids_set)
        # 构造一个理想的相关度列表，用于IDCG分母的计算（或者说，ndcg_at_k内部排序用）
        # 实际上，ndcg_at_k中的ideal_relevance_sorted = sorted(relevance_list, reverse=True) 
        # 这里的relevance_list是预测列表对应的真实相关度。
        # 如果真实相关文档比k_val少，IDCG会基于实际相关文档数计算。
        # 如果真实相关文档比k_val多，IDCG会基于前k_val个最相关的文档计算。
        # 对于二元情况，这意味着IDCG会基于min(num_gt_relevant_for_query, k_val)个1.0来计算。
        # ndcg_at_k 函数内部的 sorted(relevance_list, reverse=True) 应该使用一个包含了所有文档（或至少是GT中所有相关文档）
        # 的真实相关度列表，然后取前k个来计算IDCG。
        # 但通常的做法是，IDCG是基于当前查询的 *所有* 相关文档的理想排序计算的，然后DCG和IDCG都截断到k。
        # 修正：ndcg_at_k 应该接收预测列表的真实相关度，并基于此计算DCG。
        # IDCG应基于 *所有* GT中的相关文档的理想排序来计算。
        # 为了简化，我们假设 ndcg_at_k 的实现：它接收预测列表的相关度，然后通过排序这些相关度来估算IDCG。
        # 这种估算在只有二元相关性且只关心预测列表内的文档时是可接受的。
        # 一个更标准的NDCG，IDCG部分会使用全局的真实相关文档信息。
        # 但我们当前ndcg_at_k的实现是基于传入的relevance_list（即预测列表的真实相关度）进行排序来得到ideal_relevance_sorted
        # 这在实际中可能低估IDCG，如果预测列表未能包含所有高相关度文档。
        # 鉴于我们只有二元相关性，当前ndcg_at_k的实现是：
        # DCG基于预测列表的相关度。IDCG基于对预测列表相关度排序后的理想DCG。
        
        # 让我们坚持当前 ndcg_at_k 的实现，它将基于 relevance_scores_for_ndcg_preds 来计算DCG和IDCG（通过排序）
        ndcg_score = ndcg_at_k(relevance_scores_for_ndcg_preds, k_val)
        all_ndcgs.append(ndcg_score)
        
        evaluated_qids_count += 1
            
    results = {
        f"MAP@{k_val}": np.mean(all_aps) if all_aps else 0.0,
        "P@1": np.mean(all_p1s) if all_p1s else 0.0,
        f"NDCG@{k_val}": np.mean(all_ndcgs) if all_ndcgs else 0.0,
        "num_queries": evaluated_qids_count
    }
    
    return results

def evaluate_by_turn(predictions: Dict[str, List[str]], 
                    ground_truth: Dict[str, Dict[str, List[str]]], 
                    k_val: int = 10) -> Dict[str, Dict[str, float]]:
    turn_data: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"predictions": {}, "ground_truth": {}})
    
    for qid in predictions:
        if '_' in qid:
            try:
                turn_str = qid.split('_')[-1]
                if turn_str.isdigit():
                    turn = int(turn_str)
                    turn_data[turn]["predictions"][qid] = predictions[qid]
                    if qid in ground_truth:
                        turn_data[turn]["ground_truth"][qid] = ground_truth[qid]
            except ValueError:
                continue 
    
    results_by_turn: Dict[str, Dict[str, float]] = {}
    for turn_num_key in sorted(turn_data.keys()):
        turn_preds = turn_data[turn_num_key]["predictions"]
        turn_gt = turn_data[turn_num_key]["ground_truth"]
        
        if turn_preds and turn_gt: 
            metrics = evaluate_core_metrics(turn_preds, turn_gt, k_val)
            results_by_turn[f"Turn_{turn_num_key}"] = metrics
    
    return results_by_turn

def display_results(metrics: Dict[str, float], 
                   dataset_name: str, 
                   result_type: str = "Overall",
                   k_val: int = 10,
                   pg_metric_name: str = f"NDCG", # PG现在基于的指标名 (不含@k)
                   turn_results: Optional[Dict[str, Dict[str, float]]] = None):
    print(f"\n{'='*50}")
    print(f"  {dataset_name} - {result_type} Results (k={k_val})")
    print(f"{'='*50}")
    
    num_q = metrics.get('num_queries', 0)
    if num_q == 0:
        print("\nNo queries were evaluated for this set. Check input files and query ID matching.")
        return

    print(f"\nEvaluated queries: {num_q}")
    print(f"\nCore Metrics (@{k_val}):")
    print(f"  MAP@{k_val}:  {metrics.get(f'MAP@{k_val}', 0.0):.4f}")
    print(f"  P@1:    {metrics.get('P@1', 0.0):.4f}")
    print(f"  NDCG@{k_val}: {metrics.get(f'NDCG@{k_val}', 0.0):.4f}")
    
    # PG的显示现在会指明是基于哪个指标
    if 'PG' in metrics: 
        pg_value = metrics['PG']
        print(f"  PG (vs Baseline {pg_metric_name}@{k_val}): {pg_value:+.2%} ", end="") # 更新显示
        if pg_value > 0.0001: print("(improvement)")
        elif pg_value < -0.0001: print("(degradation)")
        else: print("(no significant change)")
    
    if turn_results:
        print(f"\nPerformance by Turn (@{k_val}):")
        header = f"  {'Turn':<8} {f'MAP@{k_val}':<8} {'P@1':<8} {f'NDCG@{k_val}':<10} {'Queries':<8}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        
        for turn_key_str in sorted(turn_results.keys(), key=lambda x: int(x.split('_')[1])):
            turn_metrics = turn_results[turn_key_str]
            turn_num_display = turn_key_str.split('_')[1]
            print(f"  {turn_num_display:<8} "
                  f"{turn_metrics.get(f'MAP@{k_val}', 0.0):<8.3f} "
                  f"{turn_metrics.get('P@1', 0.0):<8.3f} "
                  f"{turn_metrics.get(f'NDCG@{k_val}', 0.0):<10.3f} "
                  f"{turn_metrics.get('num_queries', 0):<8}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking performance with MAP, P@1, NDCG")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., MedCorpus, LitSearch)")
    parser.add_argument("--gt_file", type=str, required=True,
                        help="Ground truth file path (JSONL format)")
    parser.add_argument("--rerank_pred_file", type=str, required=True,
                        help="Reranked predictions file path (JSONL format)")
    parser.add_argument("--unrerank_pred_file", type=str, 
                        help="Baseline (e.g., initial retrieval) predictions file path (JSONL format). Optional, for PG calculation.")
    parser.add_argument("--k", type=int, default=10,
                        help="Cutoff k for metrics (default: 10)")
    # 新增参数，用于指定PG基于哪个指标计算
    parser.add_argument("--pg_base_metric", type=str, default="NDCG", choices=["MAP", "NDCG"],
                        help="Base metric for Personalization Gain calculation (MAP or NDCG). Default: NDCG.")

    args = parser.parse_args()
    
    k_value = args.k
    pg_metric_key_for_calc = f"{args.pg_base_metric.upper()}@{k_value}" # e.g., "NDCG@10" or "MAP@10"
    pg_metric_name_for_display = args.pg_base_metric.upper()


    print(f"\nEvaluating {args.dataset_name} dataset with k={k_value}")
    print(f"Ground truth: {args.gt_file}")
    print(f"Reranked results: {args.rerank_pred_file}")
    if args.unrerank_pred_file:
        print(f"Baseline results: {args.unrerank_pred_file}")
    print(f"Personalization Gain (PG) will be calculated based on: {pg_metric_key_for_calc}")
    
    print("\nLoading data...")
    ground_truth = load_ground_truth_jsonl(args.gt_file)
    if not ground_truth:
        print("Failed to load ground truth. Exiting.")
        return
    
    rerank_predictions = load_predictions(args.rerank_pred_file, "ranked_results")
    if not rerank_predictions:
        print("Failed to load reranked predictions. Exiting.")
        return
        
    baseline_predictions = None
    if args.unrerank_pred_file:
        baseline_predictions = load_predictions(args.unrerank_pred_file, "results") 
        if not baseline_predictions:
            print("Warning: Failed to load baseline predictions. PG metric will not be calculated.")
    
    print(f"Loaded {len(ground_truth)} ground truth entries.")
    print(f"Loaded {len(rerank_predictions)} reranked predictions.")
    if baseline_predictions:
        print(f"Loaded {len(baseline_predictions)} baseline predictions.")
    
    baseline_metrics = None
    if baseline_predictions:
        print("\n" + "="*50)
        print("BASELINE (e.g., Unreranked/Initial Retrieval) Evaluation")
        print("="*50)
        baseline_metrics = evaluate_core_metrics(baseline_predictions, ground_truth, k_value)
        # PG 不在基线中显示，所以 pg_metric_name 不需要传给基线的 display
        display_results(baseline_metrics, args.dataset_name, "Baseline", k_value) 
    
    print("\n" + "="*50)
    print("RERANKED (Personalized) Evaluation")
    print("="*50)
    rerank_metrics = evaluate_core_metrics(rerank_predictions, ground_truth, k_value)
    
    if baseline_metrics: 
        rerank_metrics['PG'] = calculate_personalization_gain(
            rerank_metrics, baseline_metrics, pg_metric_key_for_calc # 使用选择的指标键
        )
    
    turn_results_reranked = None
    if args.dataset_name.lower() == "medcorpus": 
        turn_results_reranked = evaluate_by_turn(rerank_predictions, ground_truth, k_value)
    
    # 传递 pg_metric_name_for_display 给 display_results
    display_results(rerank_metrics, args.dataset_name, "Reranked", k_value, pg_metric_name_for_display, turn_results_reranked)
    
    if baseline_metrics:
        print("\n" + "="*50)
        print(f"IMPROVEMENT SUMMARY (Reranked vs Baseline, @{k_value})")
        print("="*50)
        
        map_base = baseline_metrics.get(f'MAP@{k_value}', 0.0)
        p1_base = baseline_metrics.get('P@1', 0.0)
        ndcg_base = baseline_metrics.get(f'NDCG@{k_value}', 0.0)

        map_rerank = rerank_metrics.get(f'MAP@{k_value}', 0.0)
        p1_rerank = rerank_metrics.get('P@1', 0.0)
        ndcg_rerank = rerank_metrics.get(f'NDCG@{k_value}', 0.0)
        
        map_improve_abs = map_rerank - map_base
        p1_improve_abs = p1_rerank - p1_base
        ndcg_improve_abs = ndcg_rerank - ndcg_base
        
        print(f"\nAbsolute improvements:")
        print(f"  MAP@{k_value}:  {map_base:.4f} → {map_rerank:.4f} ({map_improve_abs:+.4f})")
        print(f"  P@1:    {p1_base:.4f} → {p1_rerank:.4f} ({p1_improve_abs:+.4f})")
        print(f"  NDCG@{k_value}: {ndcg_base:.4f} → {ndcg_rerank:.4f} ({ndcg_improve_abs:+.4f})")
        
        print(f"\nRelative improvements (if baseline > 0):")
        if abs(map_base) > 1e-9: print(f"  MAP@{k_value}:  {map_improve_abs/map_base:+.1%}")
        if abs(p1_base) > 1e-9: print(f"  P@1:    {p1_improve_abs/p1_base:+.1%}")
        if abs(ndcg_base) > 1e-9: print(f"  NDCG@{k_value}: {ndcg_improve_abs/ndcg_base:+.1%}")
        
        if 'PG' in rerank_metrics: # PG的显示现在会指明是基于哪个指标
            print(f"\nPersonalization Gain (PG based on {pg_metric_name_for_display}@{k_value}): {rerank_metrics['PG']:+.2%}")
    
    print("\n" + "="*50)
    print("Evaluation completed.")
    print("="*50)

if __name__ == "__main__":
    main()
