#!/usr/bin/env python3
"""
嵌入服务模块

提供高效的文本嵌入功能，包括：
- 批量处理
- 智能缓存
- 错误重试
- 性能监控
"""

import time
import hashlib
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import RAGConfig

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """嵌入结果数据类"""
    text: str
    embedding: List[float]
    model: str
    processing_time: float
    cached: bool

class EmbeddingService:
    """嵌入服务"""
    
    def __init__(self, config: RAGConfig, openai_client: OpenAI):
        self.config = config
        self.openai_client = openai_client
        self.logger = logging.getLogger(__name__)
        
        # 初始化缓存
        cache_size = self.config.models.embedding.cache_size
        self._setup_cache(cache_size)
        
        # 线程池用于批量处理
        self._executor = ThreadPoolExecutor(
            max_workers=min(self.config.models.embedding.batch_size, 8)
        )
        
        # 性能统计
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
    
    def _setup_cache(self, cache_size: int):
        """设置缓存"""
        if cache_size > 0:
            # 使用LRU缓存装饰器
            self._cached_encode = lru_cache(maxsize=cache_size)(self._encode_impl)
            self.logger.info(f"嵌入缓存已启用，大小: {cache_size}")
        else:
            self._cached_encode = self._encode_impl
            self.logger.info("嵌入缓存已禁用")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _encode_impl(self, text: str) -> List[float]:
        """编码实现（带重试）"""
        start_time = time.time()
        
        try:
            # 预处理文本
            processed_text = self._preprocess_text(text)
            
            # 调用API
            response = self.openai_client.embeddings.create(
                model=self.config.models.embedding.name,
                input=processed_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # 归一化（如果配置要求）
            if self.config.models.embedding.normalize:
                embedding = self._normalize_vector(embedding)
            
            processing_time = time.time() - start_time
            self._stats["total_processing_time"] += processing_time
            
            self.logger.debug(f"嵌入生成成功，耗时: {processing_time:.3f}秒")
            return embedding
            
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"嵌入生成失败: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 去除多余空白
        text = " ".join(text.split())
        
        # 截断到最大长度
        max_length = self.config.models.embedding.max_length
        if len(text) > max_length:
            text = text[:max_length]
            self.logger.warning(f"文本被截断到 {max_length} 字符")
        
        return text
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """归一化向量"""
        import math
        
        # 计算L2范数
        norm = math.sqrt(sum(x * x for x in vector))
        
        if norm == 0:
            return vector
        
        # 归一化
        return [x / norm for x in vector]
    
    def encode(self, text: str) -> List[float]:
        """编码单个文本"""
        if not text.strip():
            raise ValueError("输入文本不能为空")
        
        self._stats["total_requests"] += 1
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(text)
            if hasattr(self._cached_encode, 'cache_info'):
                cache_info = self._cached_encode.cache_info()
                if cache_key in self._cached_encode.cache_info():
                    self._stats["cache_hits"] += 1
                    self.logger.debug("缓存命中")
                    return self._cached_encode(text)
            
            self._stats["cache_misses"] += 1
            return self._cached_encode(text)
            
        except Exception as e:
            self.logger.error(f"编码失败: {str(e)}")
            raise
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        if not texts:
            return []
        
        self.logger.info(f"开始批量编码 {len(texts)} 个文本")
        start_time = time.time()
        
        try:
            # 使用线程池并发处理
            futures = []
            for text in texts:
                future = self._executor.submit(self.encode, text)
                futures.append(future)
            
            # 收集结果
            embeddings = []
            for future in as_completed(futures):
                try:
                    embedding = future.result()
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.error(f"批量编码中的单个任务失败: {str(e)}")
                    # 对于失败的编码，使用零向量
                    embeddings.append([0.0] * self.config.models.embedding.dim)
            
            total_time = time.time() - start_time
            self.logger.info(f"批量编码完成，耗时: {total_time:.2f}秒")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"批量编码失败: {str(e)}")
            raise
    
    def encode_with_metadata(self, text: str) -> EmbeddingResult:
        """编码并返回元数据"""
        start_time = time.time()
        
        # 检查是否缓存
        cached = False
        if hasattr(self._cached_encode, 'cache_info'):
            cache_info = self._cached_encode.cache_info()
            if text in self._cached_encode.cache_info():
                cached = True
        
        embedding = self.encode(text)
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.config.models.embedding.name,
            processing_time=processing_time,
            cached=cached
        )
    
    def _generate_cache_key(self, text: str) -> str:
        """生成缓存键"""
        # 使用文本内容的哈希作为缓存键
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_info = {
            "enabled": self.config.models.embedding.cache_size > 0,
            "max_size": self.config.models.embedding.cache_size,
            "hits": self._stats["cache_hits"],
            "misses": self._stats["cache_misses"],
            "hit_rate": 0.0
        }
        
        total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
        if total_requests > 0:
            cache_info["hit_rate"] = self._stats["cache_hits"] / total_requests
        
        # 如果使用LRU缓存，获取详细信息
        if hasattr(self._cached_encode, 'cache_info'):
            lru_info = self._cached_encode.cache_info()
            cache_info.update({
                "current_size": lru_info.currsize,
                "max_size": lru_info.maxsize,
                "hits": lru_info.hits,
                "misses": lru_info.misses
            })
        
        return cache_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self._stats["total_requests"]
        avg_time = 0.0
        if total_requests > 0:
            avg_time = self._stats["total_processing_time"] / total_requests
        
        return {
            "total_requests": total_requests,
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "errors": self._stats["errors"],
            "average_processing_time": avg_time,
            "total_processing_time": self._stats["total_processing_time"],
            "cache_info": self.get_cache_info()
        }
    
    def clear_cache(self):
        """清空缓存"""
        if hasattr(self._cached_encode, 'cache_clear'):
            self._cached_encode.cache_clear()
            self.logger.info("嵌入缓存已清空")
    
    def resize_cache(self, new_size: int):
        """调整缓存大小"""
        if new_size < 0:
            raise ValueError("缓存大小不能为负数")
        
        # 重新设置缓存
        self._setup_cache(new_size)
        self.logger.info(f"缓存大小已调整为: {new_size}")
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """验证嵌入向量"""
        if not embedding:
            return False
        
        # 检查维度
        expected_dim = self.config.models.embedding.dim
        if len(embedding) != expected_dim:
            self.logger.warning(f"嵌入维度不匹配: 期望 {expected_dim}, 实际 {len(embedding)}")
            return False
        
        # 检查是否为数值
        try:
            for value in embedding:
                if not isinstance(value, (int, float)):
                    return False
        except Exception:
            return False
        
        return True
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            self.logger.info("嵌入服务资源清理完成")
        except Exception as e:
            self.logger.error(f"嵌入服务资源清理失败: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 