#!/usr/bin/env python
# memory_ablation.py
"""
Memory ablation experiment.
Reads detailed cognitive features, filters them based on ablation settings,
generates an "abated" personalized narrative using PersonalizedGenerator,
and then reranks documents using this ablated narrative.
"""
import os
import json
import argparse
import logging
import torch
import re 
import gc
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Assuming utils.py and personalized_generator.py are in the python path
try:
    from utils import get_config, logger as ablation_logger 
    from personalized_generator import PersonalizedGenerator
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ablation_logger = logging.getLogger("MemoryAblation_Fallback")
    ablation_logger.error(f"Failed to import necessary modules: {e}. Using fallback.")
    class DummyConfig:
        device="cpu"; reranker_path=None; personalized_text_target_length=200;
        reranker_type="gte"; local_model_path="dummy_llm_path"; llm_device="cpu"; 
        local_model_tokenizer=None; local_model_dtype="float16"; local_model_max_tokens=512;
        local_model_temperature=0.7; local_model_top_p=0.8; local_model_top_k=20;
        local_model_presence_penalty=None; local_model_repetition_penalty=1.0; enable_thinking=False;
        dataset_name="unknown"; # Added for the check
        def _update_text_length_constraints(self): pass 
        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()
    class PersonalizedGenerator:
        def __init__(self, config=None): 
            self.config = config or DummyConfig()
            self.model = None 
            self.tokenizer = None
            ablation_logger.info("Using Dummy PersonalizedGenerator")
        def generate_personalized_text(self, query, memory_results, excluded_memory=None): 
            return f"Ablated narrative for {query[:30]}... (target length {self.config.personalized_text_target_length})"

logger = ablation_logger


# --- Reranker Factory and Formatting (Simplified for ablated narrative) ---
DEFAULT_RERANKER_PATHS = {
    "gte": "/workspace/.cache/modelscope/hub/iic/gte_passage-ranking_multilingual-base",
    "jina": "/workspace/.cache/modelscope/hub/jinaai/jina-reranker-v2-base-multilingual",
    "minicpm": "/workspace/.cache/modelscope/hub/OpenBMB/MiniCPM-Reranker-Light"
}

def get_reranker_model_and_tokenizer(reranker_type: str, model_path: Optional[str], device: str, use_flash_attention: bool = False):
    actual_path = model_path or DEFAULT_RERANKER_PATHS.get(reranker_type)
    if not actual_path:
        raise ValueError(f"Path for reranker '{reranker_type}' not found.")
    logger.info(f"Loading {reranker_type} model: {actual_path} to {device}. FA2: {use_flash_attention and reranker_type == 'minicpm'}")
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    try:
        trust_remote = reranker_type in ["jina", "minicpm"] 
        tokenizer = AutoTokenizer.from_pretrained(actual_path, trust_remote_code=trust_remote, padding_side="right")
        model_kwargs = {"trust_remote_code": trust_remote, "torch_dtype": torch.float16 if torch.cuda.is_available() and "cuda" in device else torch.float32}
        if use_flash_attention and reranker_type == 'minicpm' and torch.cuda.is_available() and "cuda" in device : 
            model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForSequenceClassification.from_pretrained(actual_path, **model_kwargs).to(device).eval()
        return model, tokenizer
    except Exception as e: logger.error(f"Failed to load model {actual_path}: {e}", exc_info=True); raise


class AblatedRerankerPromptFormatter:
    def format_input(self, query: str,
                     ablated_narrative: Optional[str], 
                     document_text_dict: Dict[str, Any],
                     reranker_type: str):
        doc_content = " ".join(filter(None, [document_text_dict.get("title",""), document_text_dict.get("text",""), document_text_dict.get("full_paper","")])).strip().replace("\n"," ")
        
        is_personalized_mode = bool(ablated_narrative) 

        if reranker_type == "jina":
            query_part = f"Query: {query}"
            if is_personalized_mode:
                query_part += f" User Background: {ablated_narrative}"
            return (query_part.strip(), doc_content)
        else: 
            if is_personalized_mode:
                template = self._get_personalized_template()
                formatted_text = template.format(query=query, personalized_features=ablated_narrative, document_text=doc_content)
            else: 
                template = self._get_baseline_template()
                formatted_text = template.format(query=query, document_text=doc_content)

            if reranker_type == "minicpm":
                instruction = "Evaluate document relevance for the query."
                if is_personalized_mode:
                    instruction = "Considering user background, evaluate document relevance for the query."
                    formatted_text = f"<s>Instruction: {instruction}\nQuery: {query}\nUser Background: {ablated_narrative}\nDocument: {doc_content}</s>"
                else:
                    formatted_text = f"<s>Instruction: {instruction}\nQuery: {query}\nDocument: {doc_content}</s>"
            return formatted_text

    def _get_personalized_template(self):
        return """Task: Evaluate document relevance for the query, considering the user's background and interests.
Aspects: 1. Technical relevance. 2. Alignment with user's profile. 3. Usefulness.

Query: {query}
User Background and Interests: 
{personalized_features}
Document: {document_text}"""

    def _get_baseline_template(self):
        return "Task: Evaluate document relevance for the query based on content.\n\nQuery: {query}\n\nDocument: {document_text}"


def batch_rerank_documents_ablation(
    reranker_type: str, model, tokenizer,
    query: str, ablated_narrative: Optional[str], 
    docs: List[Dict], device: str, batch_size=4, max_length=512
) -> List[Dict]:
    results = []
    formatter = AblatedRerankerPromptFormatter()
    for i in range(0, len(docs), batch_size):
        batch_docs_data = docs[i:i+batch_size]
        batch_formatted_inputs = []
        original_docs_for_batch = []
        for doc_data in batch_docs_data:
            formatted_input = formatter.format_input(query, ablated_narrative, doc_data, reranker_type)
            if formatted_input:
                batch_formatted_inputs.append(formatted_input)
                original_docs_for_batch.append(doc_data)
        
        if not batch_formatted_inputs: continue
        try:
            with torch.no_grad():
                if reranker_type == "jina":
                    scores = model.compute_score(batch_formatted_inputs, max_length=max_length)
                else:
                    inputs = tokenizer(batch_formatted_inputs, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)
                    outputs = model(**inputs, return_dict=True)
                    scores = outputs.logits.view(-1).float().cpu().numpy()
            for j, doc_d in enumerate(original_docs_for_batch):
                score_val = float(scores[j]) if j < len(scores) else 0.0
                res = {"text_id":doc_d.get("text_id",""), "title":doc_d.get("title",""), "text":doc_d.get("text",""), "score":score_val}
                if "full_paper" in doc_d and doc_d.get("full_paper"): res["full_paper"] = doc_d["full_paper"]
                results.append(res)
        except Exception as e:
            logger.error(f"Batch rerank error (ablation, type {reranker_type}): {e}", exc_info=True)
            for doc_d in original_docs_for_batch: 
                 res = {"text_id":doc_d.get("text_id",""), "title":doc_d.get("title",""), "text":doc_d.get("text",""), "score":doc_d.get("score",0.0)}
                 if "full_paper" in doc_d and doc_d.get("full_paper"): res["full_paper"] = doc_d["full_paper"]
                 results.append(res)
    return sorted(results, key=lambda x: x['score'], reverse=True)

# --- Helper Functions (Filtering, Loading) ---
def filter_memory_features(all_features: List[str], exclude_memory_type: Optional[str]) -> List[str]:
    if exclude_memory_type is None or exclude_memory_type == "none" or not all_features:
        return all_features 
    
    filtered = []
    TAG_MAP = {
        "sequential": "[SEQUENTIAL_MEMORY]",
        "working": "[WORKING_MEMORY]",
        "long_explicit": "[LONG_EXPLICIT]", 
        "long_implicit": "[LONG_IMPLICIT]"  
    }
    
    tags_to_exclude_prefixes = []
    if exclude_memory_type == "long": # "long" implies both explicit and implicit LTM parts if they were distinct
        tags_to_exclude_prefixes.extend([TAG_MAP["long_explicit"], TAG_MAP["long_implicit"]])
    elif exclude_memory_type in TAG_MAP: # For "sequential" or "working"
        tags_to_exclude_prefixes.append(TAG_MAP[exclude_memory_type])
    else: 
        return all_features

    if not tags_to_exclude_prefixes: 
        return all_features

    for feature_str in all_features:
        is_excluded = any(feature_str.startswith(prefix) for prefix in tags_to_exclude_prefixes)
        if not is_excluded:
            filtered.append(feature_str)
            
    logger.debug(f"Ablation: Excluded '{exclude_memory_type}'. Kept {len(filtered)}/{len(all_features)} tagged features.")
    return filtered

def load_cognitive_and_retrieved_data(
    cognitive_features_path: str, 
    retrieved_results_path: str, 
    continuity_map: Optional[Dict[str, bool]] = None, 
    continuity_filter: str = "all"
) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    cognitive_data_loaded = {}
    retrieved_data_loaded = {}
    query_ids_to_keep = set()

    if not os.path.exists(cognitive_features_path):
        logger.error(f"Cognitive features file not found: {cognitive_features_path}"); return {}, {}
    try:
        with open(cognitive_features_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    if not query_id: continue
                    
                    keep = True
                    if continuity_filter != "all" and continuity_map is not None:
                        query_continuity = data.get("continuity", continuity_map.get(query_id, True))
                        if (continuity_filter == "continuous" and not query_continuity) or \
                           (continuity_filter == "non_continuous" and query_continuity):
                            keep = False
                    
                    if keep:
                        cognitive_data_loaded[query_id] = data 
                        query_ids_to_keep.add(query_id)
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON in {cognitive_features_path}: {line.strip()}")
    except Exception as e: logger.error(f"Error loading {cognitive_features_path}: {e}"); return {}, {}
    logger.info(f"Loaded {len(cognitive_data_loaded)} cognitive entries (filter: {continuity_filter}).")

    if not os.path.exists(retrieved_results_path):
        logger.error(f"Retrieved results file not found: {retrieved_results_path}"); return {}, {}
    try:
        with open(retrieved_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    if query_id in query_ids_to_keep: 
                         retrieved_data_loaded[query_id] = data.get("results", [])
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON in {retrieved_results_path}: {line.strip()}")
    except Exception as e: logger.error(f"Error loading {retrieved_results_path}: {e}"); return {}, {}
    
    final_cognitive_data = {qid: data for qid, data in cognitive_data_loaded.items() if qid in retrieved_data_loaded}
    if len(final_cognitive_data) < len(cognitive_data_loaded):
        logger.info(f"Final common queries for ablation after matching with retrieved: {len(final_cognitive_data)}")
    
    return final_cognitive_data, retrieved_data_loaded

def load_original_queries_continuity(file_path: str) -> Dict[str, bool]:
    continuity_map = {}
    if not os.path.exists(file_path):
        logger.warning(f"Original queries file for continuity info not found: {file_path}"); return continuity_map
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    is_continuous = data.get("continuity", False) 
                    if query_id: continuity_map[query_id] = is_continuous
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON in original queries file: {line.strip()}")
        logger.info(f"Loaded continuity info for {len(continuity_map)} queries from {file_path}")
    except Exception as e: logger.error(f"Error loading original queries {file_path}: {e}")
    return continuity_map

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Memory Ablation Experiment with Narrative Generation")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["CORAL", "LitSearch", "MedCorpus"])
    parser.add_argument("--reranker_type", type=str, default="minicpm", choices=["gte", "jina", "minicpm"])
    parser.add_argument("--reranker_path", type=str, help="Optional: Explicit path to the reranker model")
    parser.add_argument("--exclude_memory", type=str, default="none", choices=["sequential", "working", "long", "none"],
                        help="Memory component to exclude for narrative generation. 'none' uses all available features.")
    
    parser.add_argument("--cognitive_features_input_path", type=str, required=True, help="Path to cognitive_features_detailed.jsonl")
    parser.add_argument("--retrieved_results_input_path", type=str, required=True, help="Path to retrieved.jsonl")
    parser.add_argument("--original_queries_path", type=str, required=True, help="Path to original queries.jsonl (for continuity info)")
    
    parser.add_argument("--output_path", type=str, required=True, help="Output path for reranked results of this specific ablation run")
    
    parser.add_argument("--personalized_text_target_length", type=int, help="Target length for ablated narrative (uses global config if not set here)")
    parser.add_argument("--local_model_path", type=str, help="Path to LLM for PersonalizedGenerator (uses global config if not set)")

    parser.add_argument("--use_flash_attention", action="store_true", help="Enable Flash Attention 2 for MiniCPM")
    parser.add_argument("--batch_size", type=int, default=8, help="Reranking batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for reranker")
    parser.add_argument("--top_k", type=int, default=10, help="Number of final documents to save after reranking")
    parser.add_argument("--initial_top_k", type=int, default=50, help="Number of candidates from retrieval to rerank")
    parser.add_argument("--continuity_filter", type=str, default="all", choices=["all", "continuous", "non_continuous"],
                        help="Filter queries by continuity before processing")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for reranker. LLM uses config.llm_device.")
    args = parser.parse_args()

    # --- Setup Config and Device ---
    config = get_config() 
    # Update global config with relevant args from this script
    # This allows run_memory_ablation.sh or direct calls to override parts of the global config for this specific run
    # Note: args.dataset_name is already used directly, but other config aspects might be influenced
    if args.local_model_path: config.local_model_path = args.local_model_path
    if args.personalized_text_target_length:
        config.personalized_text_target_length = args.personalized_text_target_length
        config.length_suffix = f"_L{config.personalized_text_target_length}"
        config._update_text_length_constraints() 
    
    # Ensure the config's dataset_name is also updated if passed via args, as it's used by other components
    if args.dataset_name:
        config.dataset_name = args.dataset_name # This ensures PersonalizedGenerator also sees the correct dataset name if it needs it

    # --- LitSearch specific ablation logic ---
    # If the dataset is LitSearch, only proceed if exclude_memory is 'working' or 'long'.
    # For 'sequential' or 'none', skip the run.
    if args.dataset_name.lower() == "litsearch":
        if args.exclude_memory == "sequential":
            logger.info(f"LitSearch dataset: Skipping ablation for --exclude_memory sequential. "
                        f"The main LitSearch pipeline (via cognitive_retrieval.py) already excludes SM features "
                        f"from narrative generation. This run would be redundant.")
            return # Exit gracefully
        elif args.exclude_memory == "none":
            logger.info(f"LitSearch dataset: Skipping ablation for --exclude_memory none. "
                        f"The baseline for LitSearch effectively has SM excluded from narrative generation. "
                        f"Focus is on ablating WM or LM against this SM-excluded baseline.")
            return # Exit gracefully
        # If exclude_memory is "working" or "long", the script will proceed.
        logger.info(f"LitSearch dataset: Proceeding with ablation for --exclude_memory {args.exclude_memory}.")


    reranker_device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count() else "cpu"
    if "cuda" in reranker_device and args.gpu_id >= torch.cuda.device_count():
        logger.warning(f"Reranker GPU ID {args.gpu_id} invalid, using CPU for reranker.")
        reranker_device = "cpu"
    logger.info(f"Reranker will use device: {reranker_device}. LLM for narrative generation will use device: {config.llm_device}")

    logger.info(f"--- Memory Ablation Run ---")
    logger.info(f"Dataset: {args.dataset_name}, Reranker: {args.reranker_type}, Exclude: {args.exclude_memory}, Continuity: {args.continuity_filter}")
    logger.info(f"Output to: {args.output_path}, Ablated Narrative Target Length: {config.personalized_text_target_length}") # Use config value here

    continuity_map = load_original_queries_continuity(args.original_queries_path)
    cognitive_data, retrieved_results = load_cognitive_and_retrieved_data(
        args.cognitive_features_input_path, args.retrieved_results_input_path,
        continuity_map, args.continuity_filter
    )

    if not cognitive_data or not retrieved_results:
        logger.error("Failed to load input data. Exiting ablation run."); return

    try: 
        reranker_model, reranker_tokenizer = get_reranker_model_and_tokenizer(
            args.reranker_type, args.reranker_path, reranker_device, args.use_flash_attention
        )
    except Exception as e: logger.error(f"Failed to init reranker: {e}. Exiting."); return

    try: 
        narrative_generator = PersonalizedGenerator(config=config) 
        if narrative_generator.model is None: 
            raise RuntimeError("PersonalizedGenerator LLM failed to load. Check LLM path and config.")
    except Exception as e: logger.error(f"Failed to init PersonalizedGenerator: {e}. Exiting."); return

    final_output_for_file = []
    query_ids_to_process = list(cognitive_data.keys()) 

    for query_id in tqdm(query_ids_to_process, desc=f"Ablation ({args.reranker_type}, {args.exclude_memory}, {args.continuity_filter})"):
        if query_id not in retrieved_results: 
            logger.warning(f"Query {query_id} missing from retrieved results. Skipping.")
            continue

        current_cognitive_data = cognitive_data[query_id]
        original_query_text = current_cognitive_data["query"]
        # IMPORTANT: For LitSearch, cognitive_features_detailed.jsonl (loaded into current_cognitive_data)
        # will ALREADY have SM features excluded from 'tagged_memory_features' due to the change in cognitive_retrieval.py.
        # So, filter_memory_features will operate on this SM-pre-excluded list if it's LitSearch.
        all_tagged_features = current_cognitive_data.get("tagged_memory_features", [])
        
        # filter_memory_features will now correctly exclude WM or LM from the (already SM-excluded for LitSearch) features
        filtered_tagged_features = filter_memory_features(all_tagged_features, args.exclude_memory)
        
        memory_input_for_generator = {"tagged_memory_features": filtered_tagged_features}
        ablated_narrative = narrative_generator.generate_personalized_text(
            query=original_query_text,
            memory_results=memory_input_for_generator
            # excluded_memory argument in generate_personalized_text is not strictly needed
            # if memory_results already contains the correctly filtered features.
        )
        if "Error:" in ablated_narrative and "未就绪" not in ablated_narrative : # Handle generation errors, but not "not ready"
            logger.warning(f"Narrative generation error for {query_id}. Using empty narrative for reranking. Error: {ablated_narrative}")
            ablated_narrative = "" 

        candidate_docs_to_rerank = retrieved_results[query_id][:args.initial_top_k]

        reranked_docs_list = batch_rerank_documents_ablation(
            args.reranker_type, reranker_model, reranker_tokenizer,
            original_query_text, ablated_narrative, 
            candidate_docs_to_rerank, 
            reranker_device, args.batch_size, args.max_length
        )
        final_top_docs = reranked_docs_list[:args.top_k]

        result_entry = {
            "query_id": query_id, "query": original_query_text,
            "ablated_personalized_narrative": ablated_narrative, 
            "used_filtered_tagged_features_for_narrative": filtered_tagged_features, 
            "continuity": current_cognitive_data.get("continuity", True), 
            "ranked_results": final_top_docs
        }
        final_output_for_file.append(result_entry)

    logger.info(f"Saving {len(final_output_for_file)} results to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    try:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for res_item in final_output_for_file:
                f.write(json.dumps(res_item, ensure_ascii=False) + '\n')
        logger.info("Ablation run and saving completed successfully.")
    except IOError as e: logger.error(f"Failed to write results to {args.output_path}: {e}")

    del reranker_model, reranker_tokenizer, narrative_generator, cognitive_data, retrieved_results
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
