#!/usr/bin/env python3
"""
RAG系统基础模块

提供统一的服务基类和通用接口
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import asyncio

from src.rag.config import RAGConfig
from src.rag.exceptions import RAGException, handle_exception


@dataclass
class ServiceMetrics:
    """服务度量指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    last_request_time: Optional[float] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_request(self, processing_time: float, success: bool = True, error: Optional[str] = None):
        """添加请求记录"""
        self.total_requests += 1
        self.total_processing_time += processing_time
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                self.errors.append({
                    "timestamp": self.last_request_time,
                    "error": error,
                    "processing_time": processing_time
                })
        
        # 计算平均处理时间
        if self.total_requests > 0:
            self.average_processing_time = self.total_processing_time / self.total_requests
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def reset(self):
        """重置指标"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        self.last_request_time = None
        self.errors.clear()


class BaseService(ABC):
    """服务基类
    
    提供统一的服务管理功能：
    - 配置管理
    - 日志记录
    - 性能监控
    - 错误处理
    - 健康检查
    """
    
    def __init__(self, config: RAGConfig, service_name: str):
        self.config = config
        self.service_name = service_name
        self.logger = logging.getLogger(f"rag.{service_name}")
        self.metrics = ServiceMetrics()
        self._is_initialized = False
        self._is_healthy = True
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{service_name}")
        
        # 初始化服务
        try:
            self._initialize()
            self._is_initialized = True
            self.logger.info(f"{service_name} 服务初始化成功")
        except Exception as e:
            self.logger.error(f"{service_name} 服务初始化失败: {str(e)}")
            self._is_healthy = False
            raise
    
    @abstractmethod
    def _initialize(self) -> None:
        """初始化服务（子类实现）"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查（子类实现）"""
        pass
    
    def is_healthy(self) -> bool:
        """检查服务是否健康"""
        if not self._is_initialized:
            return False
        
        try:
            self._is_healthy = self.health_check()
        except Exception as e:
            self.logger.error(f"{self.service_name} 健康检查失败: {str(e)}")
            self._is_healthy = False
        
        return self._is_healthy
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        return {
            "service_name": self.service_name,
            "is_initialized": self._is_initialized,
            "is_healthy": self._is_healthy,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": self.metrics.get_success_rate(),
                "error_rate": self.metrics.get_error_rate(),
                "average_processing_time": self.metrics.average_processing_time,
                "last_request_time": self.metrics.last_request_time
            }
        }
    
    def reset_metrics(self) -> None:
        """重置服务指标"""
        self.metrics.reset()
        self.logger.info(f"{self.service_name} 指标已重置")
    
    def _record_request(
        self, 
        operation: str,
        processing_time: float, 
        success: bool = True, 
        error: Optional[Exception] = None
    ) -> None:
        """记录请求"""
        error_msg = str(error) if error else None
        self.metrics.add_request(processing_time, success, error_msg)
        
        if success:
            self.logger.debug(
                f"{self.service_name}.{operation} 成功，"
                f"耗时: {processing_time:.3f}s"
            )
        else:
            self.logger.error(
                f"{self.service_name}.{operation} 失败，"
                f"耗时: {processing_time:.3f}s，错误: {error_msg}"
            )
    
    def _measure_time(self, operation: str):
        """时间测量装饰器上下文管理器"""
        return _TimeMeasurer(self, operation)
    
    async def _async_execute(self, func, *args, **kwargs):
        """异步执行函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, *args, **kwargs)
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            self.logger.info(f"{self.service_name} 资源清理完成")
        except Exception as e:
            self.logger.error(f"{self.service_name} 资源清理失败: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class _TimeMeasurer:
    """时间测量上下文管理器"""
    
    def __init__(self, service: BaseService, operation: str):
        self.service = service
        self.operation = operation
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        success = exc_type is None
        error = exc_val if exc_val else None
        
        self.service._record_request(
            self.operation, 
            processing_time, 
            success, 
            error
        )


class CacheableService(BaseService):
    """支持缓存的服务基类"""
    
    def __init__(self, config: RAGConfig, service_name: str, cache_size: int = 1000):
        self.cache_size = cache_size
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        super().__init__(config, service_name)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键（子类可重写）"""
        import hashlib
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if key in self._cache:
            self._cache_hits += 1
            self.logger.debug(f"缓存命中: {key}")
            return self._cache[key]
        
        self._cache_misses += 1
        return None
    
    def _put_to_cache(self, key: str, value: Any) -> None:
        """放入缓存"""
        # 简单的LRU实现：如果缓存满了，删除第一个元素
        if len(self._cache) >= self.cache_size:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[key] = value
        self.logger.debug(f"数据已缓存: {key}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "miss_rate": 1 - hit_rate
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info(f"{self.service_name} 缓存已清空")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标（包含缓存信息）"""
        metrics = super().get_metrics()
        metrics["cache_info"] = self.get_cache_info()
        return metrics


class AsyncService(BaseService):
    """异步服务基类"""
    
    def __init__(self, config: RAGConfig, service_name: str):
        super().__init__(config, service_name)
        self._semaphore = asyncio.Semaphore(10)  # 限制并发数
    
    async def _async_operation(self, operation_name: str, coro):
        """执行异步操作"""
        async with self._semaphore:
            start_time = time.time()
            try:
                result = await coro
                processing_time = time.time() - start_time
                self._record_request(operation_name, processing_time, True)
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                rag_exception = handle_exception(e)
                self._record_request(operation_name, processing_time, False, rag_exception)
                raise rag_exception


class ConfigurableService(BaseService):
    """可配置的服务基类"""
    
    def __init__(self, config: RAGConfig, service_name: str):
        super().__init__(config, service_name)
        self._config_cache = {}
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点符号路径）"""
        if key in self._config_cache:
            return self._config_cache[key]
        
        # 支持嵌套配置键，如 "models.embedding.name"
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = getattr(value, k)
            self._config_cache[key] = value
            return value
        except AttributeError:
            self.logger.warning(f"配置键不存在: {key}，使用默认值: {default}")
            return default
    
    def validate_config(self, required_keys: List[str]) -> None:
        """验证必需的配置项"""
        missing_keys = []
        for key in required_keys:
            if self.get_config_value(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise RAGException(
                f"{self.service_name} 缺少必需的配置项: {', '.join(missing_keys)}",
                error_code="CONFIG_ERROR",
                context={"missing_keys": missing_keys}
            ) 