import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """RAG 系统配置类"""
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://10.172.10.103:11434/v1"))
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://10.172.10.100:19530"))
    data_path_glob: str = "milvus_docs/en/faq/*.md"
    collection_name: str = "qwen3_embedding_rag"
    embedding_model: str = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
    reranker_model: str = "qwen3:4b"
    llm_model: str = "qwen3:4b"
    embedding_dim: int = 1024
    metric_type: str = "IP"
    consistency_level: str = "Strong"
    search_limit: int = 10
    rerank_top_k: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 200
    cache_size: int = 1000
    batch_size: int = 32
    max_workers: int = 4
    max_retries: int = 3
    retry_delay: float = 1.0
    system_prompt: str = "你是一个专业的AI助手，请基于提供的上下文信息准确回答问题。"

    def __post_init__(self):
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY 未设置")
        if not self.openai_base_url:
            raise ValueError("OPENAI_BASE_URL 必须设置")
        if not self.milvus_uri:
            raise ValueError("MILVUS_URI 必须设置")

    def print_config(self):
        logger.info("=" * 50)
        logger.info("RAG 系统配置")
        logger.info("=" * 50)
        logger.info(f"嵌入模型: {self.embedding_model}")
        logger.info(f"重排序模型: {self.reranker_model}")
        logger.info(f"LLM模型: {self.llm_model}")
        logger.info(f"Milvus URI: {self.milvus_uri}")
        logger.info(f"数据路径: {self.data_path_glob}")
        logger.info(f"分块大小: {self.chunk_size}, 重叠: {self.chunk_overlap}")
        logger.info(f"批处理: {self.batch_size}, 缓存: {self.cache_size}")
        logger.info(f"并发数: {self.max_workers}")
        logger.info("=" * 50) 