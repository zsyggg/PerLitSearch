# utils.py - 支持Llama3 API和SiliconFlow API
import os
import logging
import torch
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('perslitrank.log')
    ]
)
logger = logging.getLogger('PersLitRank')

# 注册表类 - memory_system.py 需要这些
class FeatureExtractorRegistry:
    """特征提取器注册表"""
    _extractors = {}
    
    @classmethod
    def register(cls, name):
        def decorator(extractor_class):
            cls._extractors[name] = extractor_class
            return extractor_class
        return decorator
    
    @classmethod
    def get_extractor(cls, name, **kwargs):
        if name not in cls._extractors:
            raise ValueError(f"Unknown feature extractor: {name}")
        return cls._extractors[name](**kwargs)

class MemorySystemRegistry:
    """内存系统注册表"""
    _systems = {}
    
    @classmethod
    def register(cls, name):
        def decorator(system_class):
            cls._systems[name] = system_class
            return system_class
        return decorator
    
    @classmethod
    def get_system(cls, name, **kwargs):
        if name not in cls._systems:
            raise ValueError(f"Unknown memory system: {name}")
        return cls._systems[name](**kwargs)

@dataclass
class Document:
    text_id: str
    title: str = ""
    text: str = ""
    full_paper: Optional[str] = None
    full_text: Optional[str] = None
    score: float = 0.0

@dataclass
class Query:
    query_id: str
    query: str
    personalized_features: str = ""
    tagged_memory_features: List[str] = field(default_factory=list)
    sequential_results_raw: Optional[Dict] = field(default_factory=dict)
    working_memory_state_raw: Optional[Dict] = field(default_factory=dict)
    long_term_memory_results_raw: Optional[Dict] = field(default_factory=dict)
    topic_id: str = ""
    turn_id: int = 0

    def __post_init__(self):
        self.query_id = str(self.query_id)
        if "_" in self.query_id:
            parts = self.query_id.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                self.topic_id = "_".join(parts[:-1])
                try:
                    self.turn_id = int(parts[-1])
                except ValueError:
                    self.turn_id = 0
            else:
                self.topic_id = self.query_id
                self.turn_id = 0
        else:
            self.topic_id = self.query_id
            self.turn_id = 0


class Config:
    def __init__(self):
        # 核心路径和标识符
        self.dataset_name = "MedCorpus"
        self.base_data_dir = "/workspace/PerMed/data"
        self.results_dir = "./results"

        # 设备配置
        self.gpu_id = 0
        self.device = None
        self.llm_device = None
        self._setup_device()

        # 内存系统和特征提取
        self.memory_components = ["sequential", "working", "long"]
        self.feature_extractor = "keybert"
        self.keybert_embedder_device = "cpu"
        if self.device and "cuda" in self.device:
            self.keybert_embedder_device = self.device
        
        self.keybert_model = "/workspace/.cache/modelscope/hub/iic/gte_sentence-embedding_multilingual-base"
        self.embedding_path = "/workspace/all-MiniLM-L6-v2"
        
        self.memory_type = "vector"
        self.sentence_transformer_model = "/workspace/all-MiniLM-L6-v2"

        self.dataset_type = self._infer_dataset_type()
        self.batch_size = 256
        self.initial_top_k = 100
        self.final_top_k = 10

        self.personalized_text_target_length = 300
        self.personalized_text_max_length = 300
        self.length_suffix = f"_L{self.personalized_text_target_length}"
        self._update_text_length_constraints()

        self.max_tagged_features_for_llm = 10
        self.max_features_per_memory_module = 3
        self.max_phrases_per_tag = 5

        self._cognitive_features_detailed_base = "cognitive_features_detailed"
        self._personalized_queries_base = "personalized_queries"
        self._final_results_base = "ranked"
        self._initialize_data_paths()

        self.reranker_path = "/workspace/.cache/modelscope/hub/jinaai/jina-reranker-v2-base-multilingual"
        self.jina_reranker_path = "/workspace/.cache/modelscope/hub/jinaai/jina-reranker-v2-base-multilingual"
        self.minicpm_reranker_path = "/workspace/.cache/modelscope/hub/OpenBMB/MiniCPM-Reranker-Light"
        self.gte_reranker_path = "/workspace/.cache/modelscope/hub/iic/gte_passage-ranking_multilingual-base"
        self.reranker_type = "jina"
        
        self.rerank_input_type = "profile_only"
        
        self.reranker_max_length = 512

        # LLM API配置 - 支持多种模型
        self.llm_api_type = "ollama"  # 可选: "ollama", "siliconflow"
        
        # Ollama/Llama3 API配置
        self.llm_base_url = "http://172.18.147.77:11434"
        self.llm_model = "llama3:8b"
        
        # SiliconFlow API配置
        self.siliconflow_api_key = "sk-klnmpwfrfjowvolpblilseprcfwlalniumwxocgjrrcrtqib"
        self.siliconflow_api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.siliconflow_model = "deepseek-ai/DeepSeek-R1"

        # 添加模型后缀相关
        self.model_suffix = ""  # 模型后缀，用于文件名区分
        self._update_model_suffix()  # 初始化模型后缀

        # 通用LLM参数
        self.profile_generation_attempts = 1
        self.use_fixed_seed = True
        self.llm_seed = 42

        self.local_model_path = None
        self.local_model_tokenizer = None
        self.local_model_dtype = "float16"
        self.local_model_max_tokens = 350
        self.local_model_temperature = 0.4
        self.local_model_top_p = 0.9
        self.local_model_top_k = 15
        self.local_model_presence_penalty = None
        self.local_model_repetition_penalty = 1.0
        self.enable_thinking = False
        self.is_conversational = False
        self.continuity_threshold = 0.25
        
        self.test_query_limit: Optional[int] = None

        self.two_pass_rerank = False
        self.intermediate_top_k_two_pass = 20
        self.use_flash_attention = False

    def _initialize_data_paths(self):
        self.queries_path = self._get_data_path("queries.jsonl")
        self.corpus_path = self._get_data_path("corpus.jsonl")
        self.corpus_embeddings_path = self._get_data_path("corpus_embeddings.npy")
        self.retrieved_results_path = self._get_results_path_nosuffix("retrieved.jsonl")
        self.cognitive_features_detailed_path = self._get_results_path_nosuffix(f"{self._cognitive_features_detailed_base}.jsonl")

    def _update_text_length_constraints(self):
        pass

    def _update_model_suffix(self):
        """根据当前使用的模型更新文件名后缀"""
        if self.llm_api_type == "siliconflow":
            # SiliconFlow模型
            if "DeepSeek-R1" in self.siliconflow_model:
                self.model_suffix = "_deepseek-r1"
            elif "QwQ-32B" in self.siliconflow_model:
                self.model_suffix = "_qwq-32b"
            else:
                # 从模型名提取简短标识
                model_name = self.siliconflow_model.split("/")[-1].lower()
                self.model_suffix = f"_{model_name.replace(':', '-')}"
        else:
            # Ollama模型
            if "llama3:8b" in self.llm_model:
                self.model_suffix = "_llama3-8b"
            elif "llama3.3:72b-32k-context" in self.llm_model:
                self.model_suffix = "_llama3.3:72b-32k-context"
            else:
                # 通用处理
                model_name = self.llm_model.replace(":", "-").replace("/", "-")
                self.model_suffix = f"_{model_name}"

    @property
    def personalized_queries_path(self):
        """生成包含长度和模型后缀的画像文件路径"""
        base_filename = f"{self._personalized_queries_base}{self.length_suffix}{self.model_suffix}.jsonl"
        return self._get_results_path_with_suffix(base_filename)

    @property
    def final_results_path(self):
        """生成包含模型后缀的重排结果路径"""
        input_type_sfx = ""
        if self.rerank_input_type == "profile_and_query":
            input_type_sfx = "_profileQuery"
        elif self.rerank_input_type == "profile_only":
            input_type_sfx = "_profileOnly"
        elif self.rerank_input_type == "query_only":
            input_type_sfx = "_queryOnly"
       
        reranker_sfx = f"_{self.reranker_type}"
        k_sfx = f"_top{self.final_top_k}"
        
        # 只有在不是 query_only 模式时才添加 length_suffix 和 model_suffix
        current_length_suffix = self.length_suffix if self.rerank_input_type != "query_only" else ""
        current_model_suffix = self.model_suffix if self.rerank_input_type != "query_only" else ""
        
        two_pass_sfx = "_2pass" if self.two_pass_rerank else ""
        
        base_filename = f"{self._final_results_base}{reranker_sfx}{input_type_sfx}{two_pass_sfx}"
        return self._get_results_path_with_suffix(f"{base_filename}{current_length_suffix}{current_model_suffix}{k_sfx}.jsonl")

    def _setup_device(self):
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.llm_device = "cpu"
            print("WARNING: CUDA 不可用，所有操作将使用 CPU。")
            return
        
        num_gpus = torch.cuda.device_count()
        if self.gpu_id >= num_gpus:
            print(f"WARNING: 提供的 GPU ID {self.gpu_id} 对于 {num_gpus} 个可用 GPU 无效。默认为 GPU 0。")
            self.gpu_id = 0
        
        self.device = f"cuda:{self.gpu_id}"
        self.llm_device = self.device
        
        try:
            gpu_name = torch.cuda.get_device_name(self.gpu_id)
            print(f"INFO: GPU 设置: 主设备设置为 {self.device} ('{gpu_name}')。")
        except Exception as e:
            print(f"ERROR: 无法获取 GPU ID {self.gpu_id} 的名称。错误: {e}")
            print(f"INFO: GPU 设置: 主设备设置为 {self.device}。")

    def _infer_dataset_type(self):
        name_lower = self.dataset_name.lower()
        if "coral" in name_lower: return "coral"
        elif "medcorpus" in name_lower: return "medcorpus"
        elif "litsearch" in name_lower: return "litsearch"
        return "unknown"

    def _get_data_path(self, filename: str) -> str:
        return os.path.join(self.base_data_dir, self.dataset_name, filename)

    def _get_results_path_with_suffix(self, filename_with_all_suffixes_ext: str) -> str:
        dataset_results_dir = os.path.join(self.results_dir, self.dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        return os.path.join(dataset_results_dir, filename_with_all_suffixes_ext)

    def _get_results_path_nosuffix(self, filename_ext: str) -> str:
        dataset_results_dir = os.path.join(self.results_dir, self.dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        return os.path.join(dataset_results_dir, filename_ext)

    def update(self, args):
        # 数据集和路径更新
        if hasattr(args, 'dataset_name') and args.dataset_name:
            if self.dataset_name != args.dataset_name:
                self.dataset_name = args.dataset_name
                self.dataset_type = self._infer_dataset_type()
                self._initialize_data_paths()
                print(f"INFO: Config: 数据集已更改为 '{self.dataset_name}'。路径已重新初始化。")

        if hasattr(args, 'data_dir') and args.data_dir:
            if self.base_data_dir != args.data_dir:
                 self.base_data_dir = args.data_dir
                 self._initialize_data_paths()
        if hasattr(args, 'results_dir') and args.results_dir:
            if self.results_dir != args.results_dir:
                self.results_dir = args.results_dir
                self._initialize_data_paths()

        # GPU配置
        if hasattr(args, 'gpu_id') and args.gpu_id is not None:
            if self.gpu_id != args.gpu_id or self.device is None:
                self.gpu_id = args.gpu_id
                self._setup_device()
                if self.device and "cuda" in self.device:
                    self.keybert_embedder_device = self.device
                else:
                    self.keybert_embedder_device = "cpu"
                print(f"INFO: Config: GPU ID 已更改。主设备: {self.device}, KeyBERT embedder: {self.keybert_embedder_device}")

        # LLM API类型选择
        if hasattr(args, 'llm_api_type') and args.llm_api_type:
            self.llm_api_type = args.llm_api_type
            self._update_model_suffix()  # 更新模型后缀
            print(f"INFO: Config: LLM API类型已更新为: {self.llm_api_type}")

        # Llama3 API 相关配置
        if hasattr(args, 'llm_base_url') and args.llm_base_url:
            self.llm_base_url = args.llm_base_url
            print(f"INFO: Config: Llama3 API URL 已更新为: {self.llm_base_url}")
        
        if hasattr(args, 'llm_model') and args.llm_model:
            self.llm_model = args.llm_model
            self._update_model_suffix()  # 更新模型后缀
            print(f"INFO: Config: Llama3 模型已更新为: {self.llm_model}")
            print(f"INFO: Config: 模型后缀已更新为: {self.model_suffix}")

        # SiliconFlow API 相关配置
        if hasattr(args, 'siliconflow_api_key') and args.siliconflow_api_key:
            self.siliconflow_api_key = args.siliconflow_api_key
            print(f"INFO: Config: SiliconFlow API key已更新")
        
        if hasattr(args, 'siliconflow_model') and args.siliconflow_model:
            self.siliconflow_model = args.siliconflow_model
            self._update_model_suffix()  # 更新模型后缀
            print(f"INFO: Config: SiliconFlow模型已更新为: {self.siliconflow_model}")
            print(f"INFO: Config: 模型后缀已更新为: {self.model_suffix}")

        # 嵌入模型路径
        new_embedding_path = getattr(args, 'embedding_path', None)
        if new_embedding_path and self.embedding_path != new_embedding_path:
            self.embedding_path = new_embedding_path
            print(f"INFO: Config: 初始检索嵌入模型路径已更新为: {self.embedding_path}")
        
        new_keybert_model = getattr(args, 'keybert_model', None)
        if new_keybert_model and self.keybert_model != new_keybert_model:
            self.keybert_model = new_keybert_model
            print(f"INFO: Config: KeyBERT模型路径已更新为: {self.keybert_model}")

        # 文本长度配置
        if hasattr(args, 'personalized_text_target_length') and args.personalized_text_target_length is not None:
            if self.personalized_text_target_length != args.personalized_text_target_length:
                self.personalized_text_target_length = args.personalized_text_target_length
                self.length_suffix = f"_L{self.personalized_text_target_length}"
                self._update_text_length_constraints()
                print(f"INFO: Config: 个性化文本目标长度已设置为 {self.personalized_text_target_length}。")
        
        if hasattr(args, 'personalized_text_max_length') and args.personalized_text_max_length is not None:
            self.personalized_text_max_length = args.personalized_text_max_length
            print(f"INFO: Config: 个性化画像最大长度约束已设置为 {self.personalized_text_max_length}。")

        # 内存和特征配置
        if hasattr(args, 'max_tagged_features_for_llm') and args.max_tagged_features_for_llm is not None:
            self.max_tagged_features_for_llm = args.max_tagged_features_for_llm
        if hasattr(args, 'max_features_per_memory_module') and args.max_features_per_memory_module is not None:
            self.max_features_per_memory_module = args.max_features_per_memory_module
        if hasattr(args, 'max_phrases_per_tag') and args.max_phrases_per_tag is not None:
            self.max_phrases_per_tag = args.max_phrases_per_tag
        
        # 其他配置
        if hasattr(args, 'final_top_k') and args.final_top_k is not None:
            self.final_top_k = args.final_top_k
            print(f"INFO: Config: final_top_k 更新为 {self.final_top_k}。")
            
        if hasattr(args, 'test_query_limit') and args.test_query_limit is not None:
            try:
                self.test_query_limit = int(args.test_query_limit)
                if self.test_query_limit <= 0:
                    print(f"WARNING: Config: test_query_limit ({self.test_query_limit}) 无效，将被忽略。")
                    self.test_query_limit = None
                else:
                    print(f"INFO: Config: test_query_limit 设置为 {self.test_query_limit}。")
            except ValueError:
                print(f"WARNING: Config: 无法将 test_query_limit '{args.test_query_limit}' 解析为整数。将被忽略。")
                self.test_query_limit = None
        
        # 重排配置
        if hasattr(args, 'two_pass_rerank'):
            self.two_pass_rerank = args.two_pass_rerank
            print(f"INFO: Config: two_pass_rerank 设置为 {self.two_pass_rerank}。")
        if hasattr(args, 'intermediate_top_k_two_pass') and args.intermediate_top_k_two_pass is not None:
            self.intermediate_top_k_two_pass = args.intermediate_top_k_two_pass
            print(f"INFO: Config: intermediate_top_k_two_pass 设置为 {self.intermediate_top_k_two_pass}。")

        if hasattr(args, 'rerank_input_type') and args.rerank_input_type is not None:
            self.rerank_input_type = args.rerank_input_type
            print(f"INFO: Config: rerank_input_type 更新为 {self.rerank_input_type}。")
        
        if hasattr(args, 'use_flash_attention'):
            self.use_flash_attention = args.use_flash_attention
            print(f"INFO: Config: use_flash_attention 设置为 {self.use_flash_attention}。")

        # 直接赋值的属性
        direct_assign_attrs = [
            'feature_extractor', 'memory_type', 'sentence_transformer_model', 'batch_size',
            'initial_top_k', 'reranker_type', 'reranker_path', 'reranker_max_length',
            'local_model_dtype', 'local_model_max_tokens', 'local_model_temperature',
            'local_model_top_p', 'local_model_top_k', 'local_model_presence_penalty',
            'local_model_repetition_penalty', 'continuity_threshold'
        ]
        
        for attr_name in direct_assign_attrs:
            if hasattr(args, attr_name):
                val = getattr(args, attr_name)
                if val is not None:
                    if attr_name == 'reranker_path' and not val:
                        pass
                    elif attr_name == 'local_model_presence_penalty' and val == 0:
                        setattr(self, attr_name, None)
                        print(f"INFO: Config: {attr_name} 设置为 None (原为 0)。")
                    else:
                        setattr(self, attr_name, val)
        
        # 布尔属性
        for bool_attr in ['enable_thinking', 'conversational']:
            if hasattr(args, bool_attr):
                setattr(self, bool_attr, getattr(args, bool_attr))
                print(f"INFO: Config: {bool_attr} 从参数设置为 {getattr(self, bool_attr)}。")

        # 内存组件
        if hasattr(args, 'memory_components') and args.memory_components:
            valid_components = ["sequential", "working", "long"]
            self.memory_components = [
                c.strip().lower() for c in args.memory_components.split(',')
                if c.strip().lower() in valid_components
            ]
            print(f"INFO: Config: 内存组件已更新为: {self.memory_components}")

        # 重排器路径
        if self.reranker_type and not (hasattr(args, 'reranker_path') and args.reranker_path):
            if self.reranker_type == "jina": self.reranker_path = self.jina_reranker_path
            elif self.reranker_type == "minicpm": self.reranker_path = self.minicpm_reranker_path
            elif self.reranker_type == "gte": self.reranker_path = self.gte_reranker_path
            print(f"INFO: Config: 为 reranker_type '{self.reranker_type}' 使用路径: {self.reranker_path}")

        # 对话模式
        if hasattr(args, 'conversational') and args.conversational is not None:
            self.is_conversational = args.conversational
        else:
            self.is_conversational = self.dataset_type in ["coral", "medcorpus"]
        
        if self.is_conversational:
            print(f"INFO: Config: 运行模式设置为对话式 (dataset_type: {self.dataset_type})。")

_config = None
def get_config():
    global _config
    if _config is None:
        _config = Config()
        print("INFO: 全局 Config 对象已创建。")
    return _config

def load_queries(config: Config) -> List[Query]:
    queries = []
    if not os.path.exists(config.queries_path):
        print(f"ERROR: 查询文件未找到: {config.queries_path}")
        return queries
    try:
        with open(config.queries_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    query_obj = Query(query_id=data['query_id'], query=data['query'])
                    queries.append(query_obj)
                except KeyError as e:
                    print(f"ERROR: 查询文件 {config.queries_path} 第 {line_num} 行缺少键 {e}")
                except Exception as e:
                    print(f"ERROR: 解析查询文件 {config.queries_path} 第 {line_num} 行时出错: {e}")
    except Exception as e:
        print(f"ERROR: 从 {config.queries_path} 加载查询失败: {e}")
    print(f"INFO: 从 {config.queries_path} 加载了 {len(queries)} 个原始查询")
    return queries

def load_corpus(config: Config) -> Dict[str, Document]:
    documents = {}
    if not os.path.exists(config.corpus_path):
        print(f"ERROR: 语料库文件未找到: {config.corpus_path}")
        return documents
    try:
        with open(config.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    text_id_str = str(data['text_id'])
                    title = data.get('title','') or ""
                    text_content = data.get('text','') or ""
                    full_paper_content = data.get('full_paper','') or ""
                    text_parts = [title, text_content, full_paper_content]
                    full_text = " ".join(filter(None, text_parts)).strip()
                    documents[text_id_str] = Document(
                        text_id=text_id_str,
                        title=title,
                        text=text_content,
                        full_paper=full_paper_content if full_paper_content else None,
                        full_text=full_text
                    )
                except KeyError as e:
                    print(f"ERROR: 语料库文件 {config.corpus_path} 第 {line_num} 行缺少键 {e}")
                except Exception as e:
                    print(f"ERROR: 解析语料库文件 {config.corpus_path} 第 {line_num} 行时出错: {e}")
    except Exception as e:
        print(f"ERROR: 从 {config.corpus_path} 加载语料库失败: {e}")
    print(f"INFO: 从 {config.corpus_path} 加载了 {len(documents)} 个文档")
    return documents
