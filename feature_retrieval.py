# feature_retrieval.py
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel # Keep transformers import for potential future use or consistency
import torch
import torch.nn.functional as F
import gc # Import gc for cleanup
from typing import Optional # Added for type hinting

# --- Critical: Ensure utils can be imported ---
try:
    # Assuming utils.py is in the same directory or Python path
    from utils import get_config, Document, Query, logger, load_queries as load_queries_util
except ImportError as e:
    # Fallback if utils are not found (basic logging, dummy classes)
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('FeatureRetrieval_Fallback')
    from dataclasses import dataclass
    @dataclass
    class Document: text_id: str; title: str = ""; text: str = ""; full_paper: Optional[str] = None; full_text: Optional[str] = None; score: float = 0.0
    @dataclass
    class Query: query_id: str; query: str; topic_id: str = ""; turn_id: int = 0; personalized_features: str = ""
    class DummyConfig:
        device="cpu"; corpus_path="corpus.jsonl"; queries_path="queries.jsonl"; batch_size=256
        embedding_path=None; corpus_embeddings_path="corpus_embeddings.npy"; initial_top_k=50
        retrieved_results_path="retrieved.jsonl"; dataset_type="unknown"
        # Add dataset_type attribute to DummyConfig
        def __init__(self): self.dataset_type = "unknown"
    def get_config(): return DummyConfig()
    def load_queries_util(config): return []
    logger.error(f"Failed to import from utils: {e}. Using fallback. Retrieval might fail if config is needed.")
# --- End utils import ---

# --- Removed: from cognitive_retrieval import CognitiveRetrieval ---
# This import is not needed for the initial retrieval step and likely caused the circular import error.


# --- Loading Functions (Adapted from utils.py, kept local for clarity) ---

def load_corpus(config):
    """Load corpus documents from file (JSONL format)"""
    documents = {}
    corpus_path = config.corpus_path

    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found: {corpus_path}")
        return documents

    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    # Combine title, text, full_paper for full_text field
                    full_text_parts = [
                        data.get('title', ''),
                        data.get('text', ''),
                        data.get('full_paper', '')
                    ]
                    # Ensure parts are strings before joining
                    full_text_parts = [str(p) if p is not None else '' for p in full_text_parts]
                    full_text = " ".join(filter(None, full_text_parts)).strip()

                    doc = Document(
                        text_id=str(data['text_id']), # Ensure text_id is string
                        title=str(data.get('title', '')),
                        text=str(data.get('text', '')),
                        full_paper=str(data.get('full_paper', '')) if data.get('full_paper') is not None else None,
                        full_text=full_text # Use combined text
                    )
                    documents[doc.text_id] = doc
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in {corpus_path} at line {line_num}: {e}")
                    continue
                except KeyError as e:
                    logger.error(f"Missing key {e} in {corpus_path} at line {line_num}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing corpus line {line_num} in {corpus_path}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Failed to load corpus file {corpus_path}: {e}")

    logger.info(f"Successfully loaded {len(documents)} documents from {corpus_path}")
    return documents

def load_queries(config):
    """Load original queries from file (JSONL format) using the utility and apply filtering"""
    # Use the imported utility function
    queries = load_queries_util(config)
    
    # Apply test_query_limit if set
    if hasattr(config, 'test_query_limit') and config.test_query_limit is not None and config.test_query_limit > 0:
        if config.dataset_type == "medcorpus":
            # For MedCorpus, limit by number of topics/groups
            topic_queries = {}
            for q in queries:
                topic_id = getattr(q, 'topic_id', None)
                if not topic_id and hasattr(q, 'query_id'):
                    # Extract topic_id from query_id if not set
                    topic_id = '_'.join(str(q.query_id).split('_')[:-1])
                
                if topic_id:
                    if topic_id not in topic_queries:
                        topic_queries[topic_id] = []
                    topic_queries[topic_id].append(q)
            
            # Keep only the first test_query_limit topics
            sorted_topics = sorted(topic_queries.keys())
            topics_to_keep = sorted_topics[:config.test_query_limit]
            
            filtered_queries = []
            for topic in topics_to_keep:
                filtered_queries.extend(topic_queries[topic])
            
            # Sort by query_id to maintain order
            filtered_queries.sort(key=lambda q: str(q.query_id))
            
            logger.info(f"MedCorpus: Limited to first {config.test_query_limit} topics. "
                       f"Processing {len(filtered_queries)} queries (from {len(queries)} total).")
            queries = filtered_queries
        else:
            # For other datasets, limit by query count
            if len(queries) > config.test_query_limit:
                queries = queries[:config.test_query_limit]
                logger.info(f"{config.dataset_name}: Limited to first {config.test_query_limit} queries.")
    
    return queries

def save_embeddings(filepath, embeddings):
    """Saves embeddings to a .npy file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    logger.info(f"Saving embeddings to {filepath}...")
    try:
        np.save(filepath, embeddings)
        logger.info(f"Embeddings successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving embeddings file: {e}", exc_info=True)

def load_embeddings(filepath):
    """Loads embeddings from a .npy file, with validation."""
    if os.path.exists(filepath):
        logger.info(f"Loading embeddings from {filepath}...")
        try:
            embeddings = np.load(filepath)
            # Check if embeddings are a 2D numpy array with content
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2 and embeddings.shape[0] > 0 and embeddings.shape[1] > 0:
                logger.info(f"Successfully loaded embeddings with shape: {embeddings.shape}")
                return embeddings
            else:
                logger.warning(f"Loaded embeddings file {filepath} has invalid content or shape: {embeddings.shape if isinstance(embeddings, np.ndarray) else type(embeddings)}. Will regenerate.")
                return None
        except Exception as e:
            logger.error(f"Error loading embeddings file {filepath}: {e}")
            logger.warning("Will regenerate embeddings.")
            return None
    logger.info(f"Embeddings file {filepath} not found. Will generate new embeddings.")
    return None

def generate_embeddings(doc_texts, config):
    """Generates embeddings using SentenceTransformer."""
    logger.info("Generating embeddings for corpus...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("SentenceTransformer not installed. Please run: pip install sentence-transformers")
        raise

    device = config.device if torch.cuda.is_available() else "cpu"
    model_name_or_path = config.embedding_path
    # Determine if trust_remote_code is needed based on path
    use_trust_remote_code = False
    if model_name_or_path:
        model_name_lower = model_name_or_path.lower()
        use_trust_remote_code = "gte" in model_name_lower or "modelscope" in model_name_lower or "qwen" in model_name_lower # Add other potential cases
    else:
        logger.error("Embedding model path (config.embedding_path) is not set. Cannot generate embeddings.")
        raise ValueError("Embedding model path is required.")


    logger.info(f"Loading SentenceTransformer model: {model_name_or_path} on device {device}")
    model = SentenceTransformer(model_name_or_path, device=device, trust_remote_code=use_trust_remote_code)

    batch_size = getattr(config, 'batch_size', 128) # Use config batch size or default
    logger.info(f"Encoding {len(doc_texts)} documents with batch size {batch_size}...")

    try:
        embeddings = model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True, # Normalize for cosine similarity / IP search
            device=device
        )
        logger.info(f"Embeddings generated with shape: {embeddings.shape}")
        # Ensure embeddings are valid before saving
        if embeddings is not None and embeddings.ndim == 2 and embeddings.shape[0] == len(doc_texts):
             save_embeddings(config.corpus_embeddings_path, embeddings)
             return embeddings
        else:
             logger.error(f"Generated embeddings are invalid or shape mismatch. Expected ({len(doc_texts)}, dim), Got: {embeddings.shape if isinstance(embeddings, np.ndarray) else 'None'}")
             raise ValueError("Embedding generation failed or produced invalid output.")
    except Exception as e:
        logger.error(f"Error during SentenceTransformer encoding: {e}", exc_info=True)
        raise
    finally:
        # Clean up model and cache
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def build_faiss_index(doc_embeddings, config):
    """Builds a FAISS index (CPU only)."""
    if doc_embeddings is None or not isinstance(doc_embeddings, np.ndarray) or doc_embeddings.ndim != 2 or doc_embeddings.shape[0] == 0:
        logger.error("Cannot build FAISS index: No valid document embeddings provided.")
        raise ValueError("Document embeddings are invalid or empty.")

    dimension = doc_embeddings.shape[1]
    logger.info(f"Building FAISS CPU index (IndexFlatIP) with dimension {dimension}...")
    try:
        index = faiss.IndexFlatIP(dimension) # Using Inner Product (cosine similarity on normalized vectors)
        # Ensure embeddings are float32 for FAISS
        index.add(doc_embeddings.astype(np.float32))
        logger.info(f"FAISS index built successfully. Total vectors: {index.ntotal}")
        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        raise

def create_query_encoder(config):
    """Creates the query encoder model."""
    try:
        from sentence_transformers import SentenceTransformer
        model_name_or_path = config.embedding_path
        if not model_name_or_path:
             logger.error("Embedding model path (config.embedding_path) is not set. Cannot create query encoder.")
             raise ValueError("Embedding model path is required.")

        use_trust_remote_code = False
        model_name_lower = model_name_or_path.lower()
        use_trust_remote_code = "gte" in model_name_lower or "modelscope" in model_name_lower or "qwen" in model_name_lower

        logger.info(f"Creating query encoder: {model_name_or_path} on device {config.device}")
        query_encoder = SentenceTransformer(model_name_or_path, device=config.device, trust_remote_code=use_trust_remote_code)
        logger.info("Query encoder created successfully.")
        return query_encoder
    except ImportError:
        logger.error("SentenceTransformer not installed. Please run: pip install sentence-transformers")
        raise
    except Exception as e:
        logger.error(f"Failed to create query encoder: {e}", exc_info=True)
        raise # Query encoder is critical

def retrieve_topk(query: Query, query_encoder, index, doc_ids, corpus, config):
    """Retrieves top-k documents for a single query using the original query text."""
    query_text = query.query # Use the original query

    try:
        # Encode the query
        query_emb = query_encoder.encode(query_text, normalize_embeddings=True)
        # Ensure query_emb is a numpy array before reshape
        if not isinstance(query_emb, np.ndarray):
             query_emb = np.array(query_emb)
        query_emb = query_emb.reshape(1, -1).astype(np.float32) # Reshape and ensure float32 for FAISS

        # Search the FAISS index
        k = config.initial_top_k
        logger.debug(f"Searching index for top {k} documents for query ID {query.query_id}...")
        scores, indices = index.search(query_emb, k)

        # Process results
        candidates = []
        if scores.size > 0 and indices.size > 0:
            for score, idx in zip(scores[0], indices[0]):
                # FAISS can return -1 for invalid indices
                if idx != -1 and idx < len(doc_ids): # Check for valid index range
                    doc_id = doc_ids[idx]
                    if doc_id in corpus:
                        doc = corpus[doc_id]
                        # Create a new Document instance or copy to avoid modifying the original corpus dict
                        candidate_doc = Document(
                             text_id=doc.text_id,
                             title=doc.title,
                             text=doc.text,
                             full_paper=doc.full_paper,
                             full_text=doc.full_text,
                             score=float(score) # Store retrieval score
                        )
                        candidates.append(candidate_doc)
                    else:
                        logger.warning(f"Document ID {doc_id} found in index but not in loaded corpus.")
                # else: logger.debug(f"Invalid index {idx} returned by FAISS search for query {query.query_id}.") # Can be noisy

        # Sort by score (FAISS IP search returns higher scores for more similar items)
        candidates.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Retrieved {len(candidates)} candidate documents for query ID {query.query_id}.")
        return candidates[:k] # Return top K

    except Exception as e:
        logger.error(f"Error during retrieve_topk for query ID {query.query_id}: {e}", exc_info=True)
        return [] # Return empty list on error

# --- Main Function ---

def main():
    """Main function for initial retrieval step."""
    config = get_config()

    logger.info(f"--- Running Step 2: Initial Document Retrieval ---")
    logger.info(f"Dataset name: {config.dataset_name}")
    logger.info(f"Dataset type: {config.dataset_type}")
    logger.info(f"Queries path: {config.queries_path}")
    logger.info(f"Corpus path: {config.corpus_path}")
    logger.info(f"Will write retrieved results to: {config.retrieved_results_path}")

    # Load queries (applies MedCorpus filtering if needed)
    logger.info("Loading queries...")
    queries = load_queries(config) # This now includes MedCorpus filtering
    if not queries:
        logger.error("No queries loaded or remaining after filtering. Exiting retrieval.")
        return

    # Load corpus
    logger.info("Loading corpus...")
    corpus = load_corpus(config)
    if not corpus:
        logger.error("Corpus is empty. Exiting retrieval.")
        return
    logger.info(f"Successfully loaded {len(corpus)} documents.")
    doc_ids = list(corpus.keys())

    # Generate or load corpus embeddings
    doc_embeddings = load_embeddings(config.corpus_embeddings_path)
    if doc_embeddings is None or doc_embeddings.shape[0] != len(corpus):
         if doc_embeddings is not None:
             logger.warning(f"Embeddings shape mismatch ({doc_embeddings.shape[0]} vs {len(corpus)} docs). Regenerating.")
         doc_texts = [corpus[doc_id].full_text for doc_id in doc_ids]
         try:
             doc_embeddings = generate_embeddings(doc_texts, config)
         except Exception as e:
             logger.error(f"Failed to generate embeddings: {e}. Exiting retrieval.")
             return
         del doc_texts # Free memory
         gc.collect()
    else:
         logger.info("Using existing corpus embeddings.")

    # Build FAISS index
    try:
        index = build_faiss_index(doc_embeddings, config)
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}. Exiting retrieval.")
        return
    del doc_embeddings # Free memory
    gc.collect()

    # Create query encoder
    try:
        query_encoder = create_query_encoder(config)
    except Exception as e:
        logger.error(f"Failed to create query encoder: {e}. Exiting retrieval.")
        return

    # Retrieve for each query and save incrementally
    logger.info(f"Retrieving top {config.initial_top_k} documents for {len(queries)} queries...")
    os.makedirs(os.path.dirname(config.retrieved_results_path), exist_ok=True)

    # Clear output file before starting
    with open(config.retrieved_results_path, 'w', encoding='utf-8') as f_out:
        pass
    logger.info(f"Cleared output file: {config.retrieved_results_path}")

    save_every_n = 50 # Save results every N queries
    results_buffer = []

    for i, q in enumerate(tqdm(queries, desc="Retrieving documents")):
        candidates = retrieve_topk(q, query_encoder, index, doc_ids, corpus, config)

        # Prepare output data for this query
        out_data = {
            "query_id": q.query_id,
            "query": q.query,
            # Include personalized_features if it exists on the query object, else empty
            "personalized_features": getattr(q, 'personalized_features', ''),
            "results": [
                {
                    "text_id": c.text_id,
                    "title": c.title,
                    "text": c.text,
                    # Include full_paper only if it exists and is not None
                    **({"full_paper": c.full_paper} if c.full_paper else {}),
                    "score": c.score # Store the retrieval score
                } for c in candidates # candidates are already sorted and limited to top_k
            ]
        }
        results_buffer.append(out_data)

        # Save buffer periodically
        if (i + 1) % save_every_n == 0 or (i + 1) == len(queries):
            try:
                with open(config.retrieved_results_path, 'a', encoding='utf-8') as f_out:
                    for result in results_buffer:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                logger.debug(f"Saved batch of {len(results_buffer)} results to {config.retrieved_results_path}")
                results_buffer = [] # Clear buffer
            except IOError as e:
                logger.error(f"IOError saving results batch: {e}")
            # Perform cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"Retrieval complete. Results saved to {config.retrieved_results_path}")

    # Final cleanup
    del query_encoder, index, corpus, queries, results_buffer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("--- Initial Document Retrieval COMPLETED ---")


if __name__ == "__main__":
    # This allows running the retrieval step directly if needed
    # Example: python feature_retrieval.py --dataset_name MedCorpus --gpu_id 0
    import argparse # Need argparse here if running standalone
    parser = argparse.ArgumentParser(description="Run Initial Document Retrieval")
    parser.add_argument("--dataset_name", type=str, default="MedCorpus", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="/workspace/PerMed/data", help="Base data directory path")
    parser.add_argument("--results_dir", type=str, default="./results", help="Base results directory path")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for encoding")
    parser.add_argument("--initial_top_k", type=int, default=None, help="Number of candidates to retrieve")
    # Add embedding_path argument for standalone execution
    parser.add_argument("--embedding_path", type=str, default="/workspace/.cache/modelscope/hub/iic/gte_sentence-embedding_multilingual-base", help="Path to embedding model")


    cli_args = parser.parse_args()
    config = get_config()
    # Manually update config for standalone run if needed, especially embedding_path
    config.update(cli_args)
    if cli_args.embedding_path: # Ensure embedding_path from args is used
        config.embedding_path = cli_args.embedding_path
        logger.info(f"Using embedding path from args: {config.embedding_path}")


    main()
