import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
from openai import OpenAI
from .config import RAGConfig

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Embedding 服务"""
    def __init__(self, config: RAGConfig, openai_client):
        self.config = config
        self.openai_client = openai_client
        self.encode = lru_cache(maxsize=self.config.performance.cache_size)(self._encode_impl)
    def _encode_impl(self, text: str) -> List[float]:
        """编码实现"""
        for attempt in range(self.config.api.max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.config.models.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == self.config.api.max_retries - 1:
                    raise e
                logger.warning(f"Embedding生成失败，重试 {attempt + 1}/{self.config.api.max_retries}: {str(e)}")
                time.sleep(self.config.api.retry_delay * (attempt + 1))
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码"""
        with ThreadPoolExecutor(max_workers=self.config.performance.max_workers) as executor:
            return list(executor.map(self.encode, texts)) 