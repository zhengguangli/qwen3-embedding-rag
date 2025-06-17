#!/usr/bin/env python3
"""
嵌入服务模块

提供高效的文本嵌入功能，包括：
- 批量处理
- 智能缓存
- 错误重试
- 性能监控
- 统一服务接口
"""

import time
import hashlib
import math
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.rag.config import RAGConfig
from .base import CacheableService
from src.rag.exceptions import RAGException, handle_exception


@dataclass
class EmbeddingResult:
    """嵌入结果数据类"""
    text: str
    embedding: List[float]
    model: str
    processing_time: float
    cached: bool
    metadata: Dict[str, Any]


class EmbeddingService(CacheableService):
    """增强的嵌入服务
    
    提供高效、可靠的文本嵌入功能，支持：
    - 批量处理和并发执行
    - 智能缓存和性能优化
    - 自动重试和错误处理
    - 健康检查和监控
    """
    
    def __init__(self, config: RAGConfig, openai_client: OpenAI):
        # 先设置openai_client属性，因为_initialize()会用到它
        self.openai_client = openai_client
        self._lru_cache: Optional[Any] = None
        
        # 初始化缓存大小
        cache_size = config.models.embedding.cache_size
        
        # 调用父类初始化（会自动调用_initialize()）
        super().__init__(config, "embedding", cache_size)
    
    def _initialize(self) -> None:
        """初始化嵌入服务"""
        try:
            # 验证必需配置
            self._validate_required_config()
            
            # 设置LRU缓存
            if self.cache_size > 0:
                self._lru_cache = lru_cache(maxsize=self.cache_size)(self._encode_impl)
                self.logger.info(f"LRU缓存已启用，大小: {self.cache_size}")
            else:
                self._lru_cache = self._encode_impl
                self.logger.info("缓存已禁用")
            
            # 测试API连接
            self._test_api_connection()
            
        except Exception as e:
            raise RAGException(
                f"嵌入服务初始化失败: {str(e)}",
                model_name=self.config.models.embedding.name
            )
    
    def _validate_required_config(self) -> None:
        """验证必需的配置项"""
        required_configs = [
            ("models.embedding.name", self.config.models.embedding.name),
            ("models.embedding.dim", self.config.models.embedding.dim),
            ("models.embedding.max_length", self.config.models.embedding.max_length)
        ]
        
        missing_configs = []
        for key, value in required_configs:
            if value is None:
                missing_configs.append(key)
        
        if missing_configs:
            raise RAGException(
                f"缺少必需的配置项: {', '.join(missing_configs)}"
            )
    
    def _test_api_connection(self) -> None:
        """测试API连接"""
        try:
            # 使用简单文本测试连接
            test_response = self.openai_client.embeddings.create(
                model=self.config.models.embedding.name,
                input="test",
                encoding_format="float"
            )
            
            if not test_response.data:
                raise RAGException("API响应为空")
            
            self.logger.debug("API连接测试成功")
            
        except Exception as e:
            raise RAGException(
                f"API连接测试失败: {str(e)}",
                endpoint=self.config.api.openai_base_url
            )
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 简单的API健康检查
            self._test_api_connection()
            return True
        except Exception as e:
            self.logger.warning(f"健康检查失败: {str(e)}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _encode_impl(self, text: str) -> List[float]:
        """编码实现（带重试）
        
        Args:
            text: 要编码的文本
            
        Returns:
            嵌入向量
            
        Raises:
            RAGException: 编码失败
        """
        with self._measure_time("encode_single"):
            try:
                # 预处理文本
                processed_text = self._preprocess_text(text)
                
                # 调用API
                response = self.openai_client.embeddings.create(
                    model=self.config.models.embedding.name,
                    input=processed_text,
                    encoding_format="float"
                )
                
                if not response.data:
                    raise RAGException("API返回空数据")
                
                embedding = response.data[0].embedding
                
                # 归一化（如果配置要求）
                if self.config.models.embedding.normalize:
                    embedding = self._normalize_vector(embedding)
                
                # 验证嵌入向量
                if not self._validate_embedding(embedding):
                    raise RAGException("生成的嵌入向量无效")
                
                return embedding
                
            except Exception as e:
                if isinstance(e, RAGException):
                    raise
                
                rag_exception = handle_exception(e)
                raise RAGException(
                    f"嵌入生成失败: {str(e)}",
                    model_name=self.config.models.embedding.name
                ) from e
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text or not text.strip():
            raise RAGException("输入文本不能为空")
        
        # 去除多余空白
        text = " ".join(text.split())
        
        # 截断到最大长度
        max_length = self.config.models.embedding.max_length
        if len(text) > max_length:
            text = text[:max_length]
            self.logger.warning(f"文本被截断到 {max_length} 字符")
        
        return text
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """归一化向量
        
        Args:
            vector: 原始向量
            
        Returns:
            归一化后的向量
        """
        # 计算L2范数
        norm = math.sqrt(sum(x * x for x in vector))
        
        if norm == 0:
            self.logger.warning("向量范数为0，无法归一化")
            return vector
        
        # 归一化
        return [x / norm for x in vector]
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """验证嵌入向量
        
        Args:
            embedding: 嵌入向量
            
        Returns:
            是否有效
        """
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
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    return False
        except Exception:
            return False
        
        return True
    
    def _get_cache_key(self, text: str, **kwargs) -> str:
        """生成缓存键
        
        Args:
            text: 文本内容
            **kwargs: 其他参数
            
        Returns:
            缓存键
        """
        # 包含模型信息和配置参数
        key_data = {
            "text": text,
            "model": self.config.models.embedding.name,
            "normalize": self.config.models.embedding.normalize,
            "max_length": self.config.models.embedding.max_length
        }
        key_data.update(kwargs)
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def encode(self, text: str) -> List[float]:
        """编码单个文本
        
        Args:
            text: 要编码的文本
            
        Returns:
            嵌入向量
            
        Raises:
            RAGException: 输入验证失败
            RAGException: 编码失败
        """
        if not isinstance(text, str):
            raise RAGException("输入必须是字符串类型")
        
        # 检查缓存
        cache_key = self._get_cache_key(text)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # 生成嵌入
        embedding = self._lru_cache(text)
        
        # 存入缓存
        self._put_to_cache(cache_key, embedding)
        
        return embedding
    
    def encode_batch(self, texts: List[str], max_workers: Optional[int] = None) -> List[List[float]]:
        """批量编码文本
        
        Args:
            texts: 文本列表
            max_workers: 最大工作线程数
            
        Returns:
            嵌入向量列表
            
        Raises:
            RAGException: 输入验证失败
            RAGException: 批量编码失败
        """
        if not texts:
            return []
        
        if not isinstance(texts, list):
            raise RAGException("texts必须是列表类型")
        
        with self._measure_time("encode_batch"):
            self.logger.info(f"开始批量编码 {len(texts)} 个文本")
            
            # 确定工作线程数
            if max_workers is None:
                max_workers = min(self.config.models.embedding.batch_size, 8)
            
            try:
                # 使用线程池并发处理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self.encode, text) for text in texts]
                    
                    # 收集结果
                    embeddings = []
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            embedding = future.result()
                            embeddings.append(embedding)
                        except Exception as e:
                            self.logger.error(f"批量编码中的文本 {i} 失败: {str(e)}")
                            # 对于失败的编码，使用零向量
                            zero_vector = [0.0] * self.config.models.embedding.dim
                            embeddings.append(zero_vector)
                
                self.logger.info(f"批量编码完成，成功处理 {len(embeddings)} 个文本")
                return embeddings
                
            except Exception as e:
                raise RAGException(f"批量编码失败: {str(e)}")
    
    def encode_with_metadata(self, text: str) -> EmbeddingResult:
        """编码并返回详细元数据
        
        Args:
            text: 要编码的文本
            
        Returns:
            包含元数据的编码结果
        """
        start_time = time.time()
        
        # 检查是否来自缓存
        cache_key = self._get_cache_key(text)
        cached = cache_key in self._cache
        
        # 生成嵌入
        embedding = self.encode(text)
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.config.models.embedding.name,
            processing_time=processing_time,
            cached=cached,
            metadata={
                "text_length": len(text),
                "embedding_dim": len(embedding),
                "normalized": self.config.models.embedding.normalize,
                "cache_key": cache_key
            }
        )
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """获取嵌入服务统计信息
        
        Returns:
            统计信息字典
        """
        base_metrics = self.get_metrics()
        
        # 添加LRU缓存信息
        lru_info = {}
        if hasattr(self._lru_cache, 'cache_info'):
            lru_cache_info = self._lru_cache.cache_info()
            lru_info = {
                "lru_hits": lru_cache_info.hits,
                "lru_misses": lru_cache_info.misses,
                "lru_current_size": lru_cache_info.currsize,
                "lru_max_size": lru_cache_info.maxsize
            }
        
        return {
            **base_metrics,
            "model_info": {
                "name": self.config.models.embedding.name,
                "dimension": self.config.models.embedding.dim,
                "max_length": self.config.models.embedding.max_length,
                "normalize": self.config.models.embedding.normalize
            },
            "lru_cache_info": lru_info
        }
    
    def clear_all_caches(self) -> None:
        """清空所有缓存"""
        # 清空基类缓存
        self.clear_cache()
        
        # 清空LRU缓存
        if hasattr(self._lru_cache, 'cache_clear'):
            self._lru_cache.cache_clear()
            self.logger.info("LRU缓存已清空")
    
    def resize_cache(self, new_size: int) -> None:
        """调整缓存大小
        
        Args:
            new_size: 新的缓存大小
            
        Raises:
            RAGException: 无效的缓存大小
        """
        if new_size < 0:
            raise RAGException("缓存大小不能为负数")
        
        # 更新配置
        self.cache_size = new_size
        
        # 重新初始化LRU缓存
        if new_size > 0:
            self._lru_cache = lru_cache(maxsize=new_size)(self._encode_impl)
        else:
            self._lru_cache = self._encode_impl
        
        # 清空基类缓存
        self.clear_cache()
        
        self.logger.info(f"缓存大小已调整为: {new_size}")
    
    def validate_model_config(self) -> bool:
        """验证模型配置
        
        Returns:
            配置是否有效
        """
        try:
            # 检查必需参数
            required_attrs = [
                self.config.models.embedding.name,
                self.config.models.embedding.dim,
                self.config.models.embedding.max_length
            ]
            
            for attr in required_attrs:
                if attr is None:
                    return False
            
            # 验证维度
            if self.config.models.embedding.dim <= 0:
                return False
            
            # 验证最大长度
            if self.config.models.embedding.max_length <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {str(e)}")
            return False 