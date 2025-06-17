#!/usr/bin/env python3
"""
RAG系统自定义异常模块

定义了完整的异常层次结构，提供详细的错误信息和上下文
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """RAG系统基础异常类"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class ConfigurationError(RAGException):
    """配置相关错误"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key
        if config_key:
            self.context["config_key"] = config_key


class ValidationError(RAGException):
    """数据验证错误"""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field_name = field_name
        if field_name:
            self.context["field_name"] = field_name


class DatabaseError(RAGException):
    """数据库相关错误"""
    pass


class MilvusConnectionError(DatabaseError):
    """Milvus连接错误"""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MILVUS_CONNECTION_ERROR", **kwargs)
        self.endpoint = endpoint
        if endpoint:
            self.context["endpoint"] = endpoint


class MilvusCollectionError(DatabaseError):
    """Milvus集合操作错误"""
    
    def __init__(self, message: str, collection_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MILVUS_COLLECTION_ERROR", **kwargs)
        self.collection_name = collection_name
        if collection_name:
            self.context["collection_name"] = collection_name


class MilvusIndexError(DatabaseError):
    """Milvus索引错误"""
    
    def __init__(self, message: str, index_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MILVUS_INDEX_ERROR", **kwargs)
        self.index_type = index_type
        if index_type:
            self.context["index_type"] = index_type


class ProcessingError(RAGException):
    """处理相关错误"""
    pass


class DocumentProcessingError(ProcessingError):
    """文档处理错误"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DOCUMENT_PROCESSING_ERROR", **kwargs)
        self.file_path = file_path
        if file_path:
            self.context["file_path"] = file_path


class EmbeddingError(ProcessingError):
    """嵌入生成错误"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EMBEDDING_ERROR", **kwargs)
        self.model_name = model_name
        if model_name:
            self.context["model_name"] = model_name


class RerankingError(ProcessingError):
    """重排序错误"""
    
    def __init__(self, message: str, algorithm: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RERANKING_ERROR", **kwargs)
        self.algorithm = algorithm
        if algorithm:
            self.context["algorithm"] = algorithm


class ModelError(RAGException):
    """模型相关错误"""
    pass


class LLMError(ModelError):
    """大语言模型错误"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="LLM_ERROR", **kwargs)
        self.model_name = model_name
        if model_name:
            self.context["model_name"] = model_name


class APIError(ModelError):
    """API调用错误"""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="API_ERROR", **kwargs)
        self.status_code = status_code
        self.endpoint = endpoint
        if status_code:
            self.context["status_code"] = status_code
        if endpoint:
            self.context["endpoint"] = endpoint


class RateLimitError(APIError):
    """API速率限制错误"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="RATE_LIMIT_ERROR", **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.context["retry_after"] = retry_after


class AuthenticationError(APIError):
    """认证错误"""
    
    def __init__(self, message: str = "API认证失败", **kwargs):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", **kwargs)


class PipelineError(RAGException):
    """管道执行错误"""
    
    def __init__(self, message: str, stage: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="PIPELINE_ERROR", **kwargs)
        self.stage = stage
        if stage:
            self.context["stage"] = stage


class TimeoutError(RAGException):
    """超时错误"""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            self.context["timeout_seconds"] = timeout_seconds


class ResourceError(RAGException):
    """资源相关错误"""
    pass


class InsufficientMemoryError(ResourceError):
    """内存不足错误"""
    
    def __init__(self, message: str, required_memory: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="INSUFFICIENT_MEMORY_ERROR", **kwargs)
        self.required_memory = required_memory
        if required_memory:
            self.context["required_memory"] = required_memory


class DiskSpaceError(ResourceError):
    """磁盘空间不足错误"""
    
    def __init__(self, message: str, required_space: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DISK_SPACE_ERROR", **kwargs)
        self.required_space = required_space
        if required_space:
            self.context["required_space"] = required_space


def handle_exception(exception: Exception) -> RAGException:
    """将标准异常转换为RAG异常
    
    Args:
        exception: 原始异常
        
    Returns:
        转换后的RAG异常
    """
    if isinstance(exception, RAGException):
        return exception
    
    # 根据异常类型进行转换
    error_mappings = {
        ConnectionError: MilvusConnectionError,
        TimeoutError: TimeoutError,
        ValueError: ValidationError,
        FileNotFoundError: DocumentProcessingError,
        MemoryError: InsufficientMemoryError,
        PermissionError: ConfigurationError
    }
    
    exception_type = type(exception)
    rag_exception_class = error_mappings.get(exception_type, RAGException)
    
    return rag_exception_class(
        message=str(exception),
        context={"original_exception": exception_type.__name__}
    )


def format_error_message(exception: RAGException) -> str:
    """格式化错误消息为用户友好的格式
    
    Args:
        exception: RAG异常对象
        
    Returns:
        格式化的错误消息
    """
    message_parts = []
    
    # 添加主要错误信息
    message_parts.append(f"❌ {exception.message}")
    
    # 添加错误代码
    if exception.error_code:
        message_parts.append(f"错误代码: {exception.error_code}")
    
    # 添加上下文信息
    if exception.context:
        context_info = []
        for key, value in exception.context.items():
            if key != "original_exception":  # 跳过内部异常信息
                context_info.append(f"{key}: {value}")
        
        if context_info:
            message_parts.append(f"详细信息: {', '.join(context_info)}")
    
    return "\n".join(message_parts)


def get_error_suggestions(exception: RAGException) -> str:
    """根据异常类型提供解决建议
    
    Args:
        exception: RAG异常对象
        
    Returns:
        解决建议
    """
    suggestions = {
        "CONFIG_ERROR": "请检查配置文件格式和必需参数是否正确设置",
        "MILVUS_CONNECTION_ERROR": "请检查Milvus服务是否运行，网络连接是否正常",
        "MILVUS_COLLECTION_ERROR": "请检查集合名称是否正确，或尝试重新创建集合",
        "EMBEDDING_ERROR": "请检查嵌入模型是否可用，API密钥是否正确",
        "LLM_ERROR": "请检查LLM服务是否可用，模型名称是否正确",
        "API_ERROR": "请检查API服务状态和网络连接",
        "RATE_LIMIT_ERROR": "API请求过于频繁，请稍后重试",
        "AUTHENTICATION_ERROR": "请检查API密钥是否正确配置",
        "TIMEOUT_ERROR": "请求超时，请检查网络连接或增加超时时间",
        "INSUFFICIENT_MEMORY_ERROR": "内存不足，请减少批处理大小或增加系统内存",
        "DOCUMENT_PROCESSING_ERROR": "文档处理失败，请检查文件格式和权限"
    }
    
    suggestion = suggestions.get(exception.error_code, "请查看日志获取更多详细信息")
    return f"💡 建议: {suggestion}" 