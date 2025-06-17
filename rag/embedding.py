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
    def __init__(self, config: RAGConfig, client: OpenAI):
        self.config = config
        self.client = client
        self._setup_cache()
    def _setup_cache(self):
        self.encode = lru_cache(maxsize=self.config.cache_size)(self._encode_impl)
    def _encode_impl(self, text: str, is_query: bool = False) -> List[float]:
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Embedding生成失败: {str(e)}")
                    raise
                logger.warning(f"Embedding生成失败，重试 {attempt + 1}/{self.config.max_retries}: {str(e)}")
                time.sleep(self.config.retry_delay * (attempt + 1))
    def encode_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        embeddings = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_text = {
                executor.submit(self.encode, text, is_query): text
                for text in texts
            }
            for future in as_completed(future_to_text):
                try:
                    embedding = future.result()
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"批量Embedding生成失败: {str(e)}")
                    raise
        return embeddings 