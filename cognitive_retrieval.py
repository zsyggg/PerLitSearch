# cognitive_retrieval.py (Stage 1: Cognitive Feature Extraction)
import json
import os
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import torch
import gc
from collections import defaultdict, OrderedDict
import time

try:
    from memory_system import CognitiveMemorySystem
except ImportError:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_fallback = logging.getLogger('CognitiveRetrieval_Fallback_MS')
    logger_fallback.error("Failed to import CognitiveMemorySystem.")
    class CognitiveMemorySystem: # Dummy
        def __init__(self, config=None): pass
        def process_query(self, query, user_id, clicks, topic_id=None): # Added topic_id for dummy
            return {
                "sequential_results_raw": {"sequential_continuity": {}, "sequential_terminology": {}},
                "working_memory_state_raw": {},
                "long_term_memory_results_raw": {}
            }
        def get_tagged_features(self, results, components, continuity_score_override=None): return []

try:
    from utils import Query, get_config, logger, load_queries
except ImportError: # Fallback
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_fallback = logging.getLogger('CognitiveRetrieval_Fallback_Utils')
    logger = logger_fallback; logger.error("Failed to import from utils.py.")
    from dataclasses import dataclass, field
    @dataclass
    class Query: query_id: str; query: str; topic_id: str = ""; turn_id: int = 0; continuity: bool = False;
    class DummyConfig: # Simplified
        device="cpu"; llm_device="cpu"; results_dir="."; dataset_name="dummy";
        cognitive_features_detailed_path = "results/dummy/cognitive_features_detailed.jsonl";
        memory_components=["sequential", "working", "long"]; continuity_threshold=0.3;
        feature_extractor='keybert'; memory_type='vector'; queries_path='queries.jsonl';
        dataset_type='unknown'; 
        test_query_limit: Optional[int] = None # Ensure dummy has this
        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()
    def load_queries(config): return []


class CognitiveFeatureExtractor:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.dataset_type = getattr(self.config, 'dataset_type', 'unknown').lower() # Ensure lowercase for comparison
        logger.info(f"CognitiveFeatureExtractor initializing for Stage 1.")
        try:
             self.memory_system = CognitiveMemorySystem(config=self.config)
             logger.info("CognitiveMemorySystem initialized for feature extraction.")
        except Exception as e: logger.error(f"Failed to init CognitiveMemorySystem: {e}", exc_info=True); raise
        logger.info(f"CognitiveFeatureExtractor ready. Extractor: {getattr(self.config, 'feature_extractor', 'N/A')}, Memory: {getattr(self.config, 'memory_type', 'N/A')}")
        logger.info(f"Current dataset type for feature extraction: {self.dataset_type}")


    def extract_features_for_query(self, query_obj: Query, user_id: str) -> Dict:
        """
        Processes a single query object to extract detailed cognitive features.
        For 'litsearch' dataset, sequential memory features will be excluded from tagged_memory_features.
        """
        query_id_str = getattr(query_obj, 'query_id', 'UNKNOWN_QUERY_ID')
        query_text = getattr(query_obj, 'query', '')
        topic_id = getattr(query_obj, 'topic_id', None) 
        turn_id = getattr(query_obj, 'turn_id', 0)

        logger.debug(f"Extracting cognitive features for query: {query_id_str} (Memory User ID: {user_id}, Topic: {topic_id}, Turn: {turn_id})")
        try:
            raw_memory_results = self.memory_system.process_query(
                query_text,
                user_id,
                [], 
                topic_id=topic_id
            )
            if not isinstance(raw_memory_results, dict):
                logger.warning(f"Memory system returned non-dict for {query_id_str}: {type(raw_memory_results)}. Using empty.")
                raw_memory_results = {
                    "sequential_results_raw": {}, "working_memory_state_raw": {}, "long_term_memory_results_raw": {}
                }


            active_mem_comp_for_narrative = getattr(self.config, 'memory_components', ["sequential", "working", "long"])
            if self.dataset_type == 'litsearch':
                if 'sequential' in active_mem_comp_for_narrative:
                    active_mem_comp_for_narrative = [comp for comp in active_mem_comp_for_narrative if comp != 'sequential']
                    logger.info(f"LitSearch dataset: Excluding 'sequential' memory from components for generating tagged features for query {query_id_str}. Effective components: {active_mem_comp_for_narrative}")
                else:
                    logger.debug(f"LitSearch dataset: 'sequential' memory already not in active_mem_comp_for_narrative for query {query_id_str}.")


            tagged_keywords_list = self.memory_system.get_tagged_features(
                raw_memory_results,
                active_mem_comp_for_narrative, 
            )

            detailed_features = {
                "query_id": query_id_str,
                "query": query_text,
                "topic_id": topic_id if topic_id else getattr(query_obj, 'topic_id', ''), 
                "turn_id": turn_id,
                "tagged_memory_features": tagged_keywords_list, 
                "sequential_results_raw": raw_memory_results.get("sequential_results_raw", {}),
                "working_memory_state_raw": raw_memory_results.get("working_memory_state_raw", {}),
                "long_term_memory_results_raw": raw_memory_results.get("long_term_memory_results_raw", {})
            }
            return detailed_features
        except Exception as e:
            logger.error(f"Error extracting cognitive features for query {query_id_str}: {e}", exc_info=True)
            return {
                "query_id": query_id_str, "query": query_text,
                "tagged_memory_features": [], "error": str(e),
                "topic_id": topic_id if topic_id else getattr(query_obj, 'topic_id', ''),
                "turn_id": turn_id,
            }

    def batch_extract_cognitive_features(self, queries: List[Query]) -> List[Dict]:
        """
        Processes a batch of queries to extract and save detailed cognitive features.
        The input 'queries' list is assumed to be already filtered by test_query_limit if applicable.
        """
        all_extracted_features_to_save = []
        output_file_path = self.config.cognitive_features_detailed_path
        
        logger.info(f"Starting batch cognitive feature extraction for {len(queries)} queries (already limited if applicable).")
        logger.info(f"Output detailed cognitive features to: {output_file_path}")

        if output_file_path:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f: pass 
                logger.info(f"Cleared cognitive features output file: {output_file_path}")
            except IOError as e:
                logger.error(f"Could not clear/create output file {output_file_path}: {e}")
                return [] 
        
        save_every_n = 50
        processed_for_saving_count = 0
        current_batch_to_save = []
        total_queries_memory_updated = 0 

        topic_groups = defaultdict(list)
        for q_obj in queries: 
            topic_id_for_group = getattr(q_obj, 'topic_id', None)
            if not topic_id_for_group: 
                topic_id_for_group = str(getattr(q_obj, 'query_id', 'unknown_topic')).split("_")[0]
            topic_groups[topic_id_for_group].append(q_obj)

        logger.info(f"Grouped into {len(topic_groups)} sessions/topics for feature extraction from the provided query list.")

        for topic_id_key, topic_queries_list in tqdm(topic_groups.items(), desc="Extracting Cognitive Features per Topic"):
            sorted_topic_queries = sorted(topic_queries_list, key=lambda q_sort: getattr(q_sort, 'turn_id', 0))
            if not sorted_topic_queries: continue

            user_memory_id_for_session = f"user_session_{topic_id_key}" 

            # 修改后：MedCorpus 现在处理所有轮次的查询
            for query_obj_to_process in sorted_topic_queries:
                total_queries_memory_updated += 1
                q_id_log = getattr(query_obj_to_process, 'query_id', 'N/A_ID')
                
                extracted_data = self.extract_features_for_query(query_obj_to_process, user_memory_id_for_session)

                # 现在所有查询都保存特征
                should_save_features = True
                
                if self.dataset_type == "medcorpus":
                    logger.info(f"MedCorpus Session {topic_id_key}, Turn {getattr(query_obj_to_process, 'turn_id', 'N/A')}: Preparing to save features for '{getattr(query_obj_to_process, 'query', '')[:30]}...'")
                else:
                    logger.debug(f"{self.dataset_type} Query {q_id_log}: Preparing to save features.")

                if should_save_features:
                    if extracted_data and not extracted_data.get("error"):
                        all_extracted_features_to_save.append(extracted_data)
                        current_batch_to_save.append(extracted_data)
                        processed_for_saving_count += 1
                        if processed_for_saving_count % save_every_n == 0 and current_batch_to_save:
                            self._save_features_batch(current_batch_to_save, output_file_path)
                            current_batch_to_save = []
                    else:
                        logger.warning(f"Skipping save for query {q_id_log} due to feature extraction error or no data.")
                
                self._cleanup_memory_iteration(q_id_log)

        if current_batch_to_save:
            self._save_features_batch(current_batch_to_save, output_file_path)

        logger.info(f"Cognitive feature extraction complete. Total queries processed for memory updates: {total_queries_memory_updated}. Features saved for: {processed_for_saving_count} queries to {output_file_path}.")
        return all_extracted_features_to_save


    def _save_features_batch(self, features_batch: List[Dict], output_file_path: str):
        if not features_batch or not output_file_path: return
        try:
            with open(output_file_path, 'a', encoding='utf-8') as f_out:
                for feature_set in features_batch:
                    f_out.write(json.dumps(feature_set, ensure_ascii=False) + "\n")
            logger.debug(f"Saved batch of {len(features_batch)} cognitive feature sets to {output_file_path}")
        except Exception as e:
            logger.error(f"Error writing cognitive features batch to {output_file_path}: {e}", exc_info=True)

    def _cleanup_memory_iteration(self, query_id_for_log: str):
        logger.debug(f"Post-query cleanup for {query_id_for_log}")
        gc.collect()
        if torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except: pass

def main():
    config = get_config()
    if not logger.handlers: 
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler('perslitrank_stage1.log')])
        if logger.name == 'CognitiveRetrieval_Fallback_Utils' or logger.name == 'PersLitRank':
             globals()['logger'] = logging.getLogger('PersLitRank_Stage1')


    logger.info("--- Running Cognitive Feature Extraction (Stage 1) Standalone ---")
    logger.info(f"Dataset: {config.dataset_name}, Type: {config.dataset_type}")
    logger.info(f"Output will be saved to: {config.cognitive_features_detailed_path}")

    queries_all = load_queries(config) 
    if not queries_all:
        logger.error(f"No queries loaded from {config.queries_path}. Exiting Stage 1.")
        return
    logger.info(f"Loaded {len(queries_all)} original queries from file.")

    queries_to_process = queries_all
    if config.test_query_limit is not None and config.test_query_limit > 0:
        if config.dataset_type == "medcorpus":
            # For MedCorpus, limit by number of unique topics
            unique_topic_ids = []
            seen_topic_ids = set()
            for q in queries_all:
                if q.topic_id not in seen_topic_ids:
                    unique_topic_ids.append(q.topic_id)
                    seen_topic_ids.add(q.topic_id)
            
            if len(unique_topic_ids) > config.test_query_limit:
                topics_to_keep = set(unique_topic_ids[:config.test_query_limit])
                queries_to_process = [q for q in queries_all if q.topic_id in topics_to_keep]
                logger.info(f"MedCorpus: Due to test_query_limit={config.test_query_limit}, "
                            f"processing all turns for the first {len(topics_to_keep)} unique topics. "
                            f"Total queries to process: {len(queries_to_process)} (out of {len(queries_all)} loaded).")
            else:
                logger.info(f"MedCorpus: test_query_limit={config.test_query_limit} is >= number of unique topics ({len(unique_topic_ids)}). "
                            f"Processing all loaded queries ({len(queries_all)}).")
        else: # For LitSearch or other single-turn datasets
            if len(queries_all) > config.test_query_limit:
                queries_to_process = queries_all[:config.test_query_limit]
                logger.info(f"{config.dataset_name}: Due to test_query_limit={config.test_query_limit}, processing the first {len(queries_to_process)} queries out of {len(queries_all)} loaded.")
            else:
                logger.info(f"{config.dataset_name}: test_query_limit={config.test_query_limit} is >= number of loaded queries ({len(queries_all)}). Processing all loaded queries.")
    
    if not queries_to_process:
        logger.error("No queries to process after applying test_query_limit. Exiting Stage 1.")
        return

    try:
        feature_extractor_instance = CognitiveFeatureExtractor(config)
    except Exception as e:
         logger.error(f"Failed to initialize CognitiveFeatureExtractor: {e}", exc_info=True)
         return

    start_time = time.time()
    feature_extractor_instance.batch_extract_cognitive_features(queries_to_process)
    end_time = time.time()

    logger.info(f"Stage 1 (Cognitive Feature Extraction) complete. Total time: {end_time - start_time:.2f}s.")
    logger.info(f"Detailed cognitive features (for {len(queries_to_process)} queries, potentially representing {len(set(q.topic_id for q in queries_to_process))} topics) saved to: {config.cognitive_features_detailed_path}")

if __name__ == "__main__":
    main()
