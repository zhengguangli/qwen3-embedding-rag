#!/usr/bin/env python3
"""
RAG系统配置管理模块

提供完整的配置管理功能，包括：
- 分层配置模型
- 类型安全验证
- 环境变量支持
- 配置热重载
- 配置缓存
- 配置迁移
"""

import os
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
import yaml
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """距离度量类型枚举"""
    IP = "IP"
    L2 = "L2"
    COSINE = "COSINE"

class ConsistencyLevel(str, Enum):
    """一致性级别枚举"""
    STRONG = "Strong"
    BOUNDED = "Bounded"
    EVENTUALLY = "Eventually"
    SESSION = "Session"

class IndexType(str, Enum):
    """索引类型枚举"""
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    FLAT = "FLAT"

class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class APIConfig(BaseModel):
    """API配置模型"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    openai_api_key: str = Field(default="", description="OpenAI API密钥")
    openai_base_url: str = Field(default="http://10.172.10.103:11434/v1", description="OpenAI兼容API基础URL")
    timeout: int = Field(default=30, ge=1, le=300, description="API超时时间(秒)")
    max_retries: int = Field(default=3, ge=0, le=10, description="最大重试次数")
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="重试延迟(秒)")
    
    @field_validator('openai_base_url')
    def validate_url(cls, v, info):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL必须以http://或https://开头")
        return v

class DatabaseConfig(BaseModel):
    """数据库配置模型"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    milvus_uri: str = Field(default="http://10.172.10.100:19530", description="Milvus服务URI")
    collection_name: str = Field(default="qwen3_embedding_rag", description="集合名称")
    embedding_dim: int = Field(default=1024, ge=1, le=8192, description="嵌入向量维度")
    metric_type: MetricType = Field(default=MetricType.IP, description="距离度量类型")
    consistency_level: ConsistencyLevel = Field(default=ConsistencyLevel.STRONG, description="一致性级别")
    index_type: IndexType = Field(default=IndexType.IVF_FLAT, description="索引类型")
    nlist: int = Field(default=1024, ge=1, description="聚类数量")
    nprobe: int = Field(default=16, ge=1, description="搜索聚类数")
    
    @field_validator('milvus_uri')
    def validate_milvus_uri(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Milvus URI必须以http://或https://开头")
        return v
    
    @field_validator('collection_name')
    def validate_collection_name(cls, v):
        if not v or len(v) > 64:
            raise ValueError("集合名称不能为空且长度不能超过64个字符")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("集合名称只能包含字母、数字、下划线和连字符")
        return v

class ModelsConfig(BaseModel):
    """模型配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    embedding_model: str = Field(default="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0", description="嵌入模型")
    reranker_model: str = Field(default="qwen3:4b", description="重排序模型")
    llm_model: str = Field(default="qwen3:4b", description="大语言模型")
    embedding_batch_size: int = Field(default=32, ge=1, le=128, description="嵌入批处理大小")
    llm_max_tokens: int = Field(default=2048, ge=1, le=8192, description="LLM最大token数")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM温度参数")
    
    @field_validator('embedding_model', 'reranker_model', 'llm_model')
    def validate_model_name(cls, v):
        if not v or len(v) > 256:
            raise ValueError("模型名称不能为空且长度不能超过256个字符")
        return v

class DataConfig(BaseModel):
    """数据处理配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    data_path_glob: str = Field(default="milvus_docs/en/faq/*.md", description="数据文件路径模式")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="文本分块大小")
    chunk_overlap: int = Field(default=200, ge=0, le=5000, description="分块重叠大小")
    supported_formats: List[str] = Field(default=[".md", ".txt", ".pdf"], description="支持的文件格式")
    encoding: str = Field(default="utf-8", description="文件编码")
    
    @field_validator('supported_formats')
    def validate_formats(cls, v):
        if not v:
            raise ValueError("支持的文件格式列表不能为空")
        for fmt in v:
            if not fmt.startswith('.'):
                raise ValueError(f"文件格式必须以.开头: {fmt}")
        return v
    
    @field_validator('encoding')
    def validate_encoding(cls, v):
        try:
            "test".encode(v)
            return v
        except LookupError:
            raise ValueError(f"不支持的编码格式: {v}")

class SearchConfig(BaseModel):
    """搜索配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    search_limit: int = Field(default=10, ge=1, le=100, description="搜索结果数量限制")
    rerank_top_k: int = Field(default=3, ge=1, le=20, description="重排序top-k")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    enable_rerank: bool = Field(default=True, description="是否启用重排序")
    enable_hybrid_search: bool = Field(default=False, description="是否启用混合搜索")

class PerformanceConfig(BaseModel):
    """性能配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    cache_size: int = Field(default=1000, ge=100, le=10000, description="缓存大小")
    max_workers: int = Field(default=4, ge=1, le=32, description="最大工作线程数")
    batch_size: int = Field(default=32, ge=1, le=256, description="批处理大小")
    enable_gpu: bool = Field(default=False, description="是否启用GPU")
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="GPU内存使用比例")
    
    @field_validator('max_workers')
    def validate_max_workers(cls, v):
        import multiprocessing
        max_cpus = multiprocessing.cpu_count()
        if v > max_cpus * 2:
            logger.warning(f"工作线程数({v})超过CPU核心数的2倍({max_cpus * 2})，可能影响性能")
        return v

class LoggingConfig(BaseModel):
    """日志配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    log_file: str = Field(default="logs/rag_system.log", description="日志文件路径")
    max_log_size: str = Field(default="10MB", description="最大日志文件大小")
    backup_count: int = Field(default=5, ge=0, le=20, description="备份文件数量")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")
    
    @field_validator('max_log_size')
    def validate_log_size(cls, v):
        import re
        pattern = r'^(\d+)(KB|MB|GB)$'
        if not re.match(pattern, v.upper()):
            raise ValueError("日志大小格式错误，应为数字+单位(KB/MB/GB)")
        return v.upper()

class OutputConfig(BaseModel):
    """输出配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    output_dir: str = Field(default="answers", description="输出目录")
    save_intermediate_results: bool = Field(default=True, description="是否保存中间结果")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    output_format: str = Field(default="txt", description="输出格式")
    
    @field_validator('output_format')
    def validate_output_format(cls, v):
        supported_formats = ['txt', 'json', 'yaml', 'csv']
        if v.lower() not in supported_formats:
            raise ValueError(f"不支持的输出格式: {v}，支持: {supported_formats}")
        return v.lower()

class PromptsConfig(BaseModel):
    """提示词配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    system_prompt: str = Field(
        default="你是一个专业的AI助手，请基于提供的上下文信息准确回答问题。如果上下文中没有相关信息，请明确说明。",
        description="系统提示词"
    )
    query_prompt: str = Field(
        default="基于以下上下文信息回答问题：\n\n{context}\n\n问题：{question}\n\n答案：",
        description="查询提示词"
    )
    rerank_prompt: str = Field(
        default="请对以下候选答案进行重新排序，选择最相关的答案：\n\n{answers}\n\n问题：{question}",
        description="重排序提示词"
    )
    
    @field_validator('system_prompt', 'query_prompt', 'rerank_prompt')
    def validate_prompt_length(cls, v):
        if len(v) > 10000:
            raise ValueError("提示词长度不能超过10000个字符")
        return v

class SecurityConfig(BaseModel):
    """安全配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    enable_ssl_verification: bool = Field(default=True, description="是否启用SSL验证")
    api_key_rotation_days: int = Field(default=30, ge=1, le=365, description="API密钥轮换天数")
    max_request_size: str = Field(default="10MB", description="最大请求大小")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000, description="每分钟请求限制")
    
    @field_validator('max_request_size')
    def validate_request_size(cls, v):
        import re
        pattern = r'^(\d+)(KB|MB|GB)$'
        if not re.match(pattern, v.upper()):
            raise ValueError("请求大小格式错误，应为数字+单位(KB/MB/GB)")
        return v.upper()

class MonitoringConfig(BaseModel):
    """监控配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    enable_metrics: bool = Field(default=True, description="是否启用指标收集")
    metrics_port: int = Field(default=8000, ge=1024, le=65535, description="指标服务端口")
    health_check_interval: int = Field(default=30, ge=5, le=300, description="健康检查间隔(秒)")
    performance_monitoring: bool = Field(default=True, description="是否启用性能监控")

class RAGConfigModel(BaseModel):
    """RAG系统完整配置模型"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @field_validator('data')
    def validate_chunk_overlap(cls, v):
        if v.chunk_overlap >= v.chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        return v

    @field_validator('search')
    def validate_rerank_top_k(cls, v):
        if v.rerank_top_k > v.search_limit:
            raise ValueError("rerank_top_k不能大于search_limit")
        return v

@dataclass
class ConfigCache:
    """配置缓存"""
    data: Dict[str, Any]
    hash: str
    timestamp: float
    file_path: str

class RAGConfig:
    """RAG系统配置类 - 支持配置验证、环境变量和热重载"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化配置"""
        self._config: RAGConfigModel = RAGConfigModel()
        self._config_file: Optional[str] = config_file
        self._last_modified: float = 0
        self._lock: threading.RLock = threading.RLock()
        self._cache: Optional[ConfigCache] = None
        self._env_mappings: Dict[str, str] = {
            'api.openai_api_key': 'OPENAI_API_KEY',
            'api.openai_base_url': 'OPENAI_BASE_URL',
            'database.milvus_uri': 'MILVUS_URI',
            'database.collection_name': 'MILVUS_COLLECTION',
            'logging.log_level': 'LOG_LEVEL',
        }
        
        # 初始化后处理
        self._load_from_env()
        self._validate_config()
        
        # 如果指定了配置文件，则加载
        if config_file:
            self.load_from_file(config_file)
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        for config_path, env_var in self._env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config_path, env_value)
    
    def _set_nested_value(self, path: str, value: Any) -> None:
        """设置嵌套配置值"""
        keys = path.split('.')
        obj = self._config
        for key in keys[:-1]:
            obj = getattr(obj, key)
        setattr(obj, keys[-1], value)
    
    def _validate_config(self) -> None:
        """验证配置"""
        try:
            self._config = RAGConfigModel.model_validate(self._config.model_dump())
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _should_reload(self, config_path: Path) -> bool:
        """判断是否需要重新加载配置"""
        if not config_path.exists():
            return False
        
        current_mtime = config_path.stat().st_mtime
        current_hash = self._calculate_file_hash(config_path)
        
        # 检查文件修改时间和哈希值
        if (current_mtime > self._last_modified or 
            (self._cache and current_hash != self._cache.hash)):
            return True
        
        return False
    
    @lru_cache(maxsize=128)
    def _parse_config_file(self, file_path: Path) -> Dict[str, Any]:
        """解析配置文件（带缓存）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def load_from_file(self, config_file: str) -> bool:
        """从文件加载配置"""
        try:
            with self._lock:
                config_path = Path(config_file)
                if not config_path.exists():
                    logger.warning(f"配置文件不存在: {config_file}")
                    return False
                
                # 检查是否需要重新加载
                if not self._should_reload(config_path):
                    return True
                
                # 解析配置文件
                config_data = self._parse_config_file(config_path)
                
                # 移除注释字段
                config_data = self._remove_comments(config_data)
                
                # 更新配置
                self._config = RAGConfigModel.model_validate(config_data)
                self._config_file = config_file
                self._last_modified = config_path.stat().st_mtime
                
                # 更新缓存
                self._cache = ConfigCache(
                    data=config_data,
                    hash=self._calculate_file_hash(config_path),
                    timestamp=time.time(),
                    file_path=str(config_path)
                )
                
                # 重新加载环境变量
                self._load_from_env()
                
                logger.info(f"配置已从文件加载: {config_file}")
                return True
                
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return False
    
    def _remove_comments(self, data: Any) -> Any:
        """移除配置中的注释字段"""
        if isinstance(data, dict):
            return {k: self._remove_comments(v) for k, v in data.items() 
                   if not k.startswith('_')}
        elif isinstance(data, list):
            return [self._remove_comments(item) for item in data]
        else:
            return data
    
    def save_to_file(self, config_file: str, include_comments: bool = True) -> bool:
        """保存配置到文件"""
        try:
            with self._lock:
                config_path = Path(config_file)
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                config_data = self._config.model_dump()
                
                # 添加注释
                if include_comments:
                    config_data = self._add_comments(config_data)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    if config_path.suffix.lower() in ('.yaml', '.yml'):
                        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                    else:
                        json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"配置已保存到文件: {config_file}")
                return True
                
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
            return False
    
    def _add_comments(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """添加配置注释"""
        comments = {
            "_comment": "RAG系统配置文件 - 基于Qwen3 Embedding和Milvus向量数据库",
            "_version": "1.0.0",
            "_description": "检索增强生成系统配置"
        }
        
        result = {**comments, **data}
        
        # 为每个部分添加注释
        section_comments = {
            "api": "API相关配置",
            "database": "Milvus向量数据库配置",
            "models": "AI模型配置",
            "data": "数据处理配置",
            "search": "搜索相关配置",
            "performance": "性能优化配置",
            "logging": "日志配置",
            "output": "输出配置",
            "prompts": "提示词配置",
            "security": "安全配置",
            "monitoring": "监控配置"
        }
        
        for section, comment in section_comments.items():
            if section in result:
                result[section] = {"_comment": comment, **result[section]}
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        try:
            keys = key.split('.')
            obj = self._config
            for k in keys:
                obj = getattr(obj, k)
            return obj
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        try:
            with self._lock:
                self._set_nested_value(key, value)
                self._validate_config()
                return True
        except Exception as e:
            logger.error(f"设置配置失败: {e}")
            return False
    
    def reload(self) -> bool:
        """重新加载配置文件"""
        if self._config_file:
            return self.load_from_file(self._config_file)
        return False
    
    @contextmanager
    def temporary_config(self, **kwargs):
        """临时配置上下文管理器"""
        original_values = {}
        try:
            # 保存原始值
            for key, value in kwargs.items():
                # 将双下划线转换为点分隔符
                config_key = key.replace('__', '.')
                original_values[config_key] = self.get(config_key)
                self.set(config_key, value)
            yield self
        finally:
            # 恢复原始值
            for key, value in original_values.items():
                self.set(key, value)
    
    def print_config(self, detailed: bool = False) -> None:
        """打印配置信息"""
        logger.info("=" * 60)
        logger.info("RAG 系统配置")
        logger.info("=" * 60)
        logger.info(f"配置文件: {self._config_file or '默认配置'}")
        
        if detailed:
            # 详细配置信息
            config_dict = self._config.model_dump()
            for section, section_data in config_dict.items():
                logger.info(f"\n🔧 {section.upper()}配置:")
                for key, value in section_data.items():
                    logger.info(f"  {key}: {value}")
        else:
            # 简要配置信息
            logger.info(f"嵌入模型: {self._config.models.embedding_model}")
            logger.info(f"重排序模型: {self._config.models.reranker_model}")
            logger.info(f"LLM模型: {self._config.models.llm_model}")
            logger.info(f"Milvus URI: {self._config.database.milvus_uri}")
            logger.info(f"数据路径: {self._config.data.data_path_glob}")
            logger.info(f"分块大小: {self._config.data.chunk_size}, 重叠: {self._config.data.chunk_overlap}")
            logger.info(f"批处理: {self._config.performance.batch_size}, 缓存: {self._config.performance.cache_size}")
            logger.info(f"并发数: {self._config.performance.max_workers}")
            logger.info(f"搜索限制: {self._config.search.search_limit}, 重排序: {self._config.search.rerank_top_k}")
        
        logger.info("=" * 60)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.model_dump()
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            self._validate_config()
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def get_diff(self, other_config: 'RAGConfig') -> Dict[str, Tuple[Any, Any]]:
        """获取与另一个配置的差异"""
        diff = {}
        current_dict = self.to_dict()
        other_dict = other_config.to_dict()
        
        def compare_dicts(dict1: Dict, dict2: Dict, prefix: str = ""):
            for key in set(dict1.keys()) | set(dict2.keys()):
                full_key = f"{prefix}.{key}" if prefix else key
                val1 = dict1.get(key)
                val2 = dict2.get(key)
                
                if isinstance(val1, dict) and isinstance(val2, dict):
                    compare_dicts(val1, val2, full_key)
                elif val1 != val2:
                    diff[full_key] = (val1, val2)
        
        compare_dicts(current_dict, other_dict)
        return diff
    
    def migrate_from_old_format(self, old_config: Dict[str, Any]) -> bool:
        """从旧格式迁移配置"""
        try:
            migration_map = {
                'embedding_model': 'models.embedding_model',
                'reranker_model': 'models.reranker_model',
                'llm_model': 'models.llm_model',
                'milvus_uri': 'database.milvus_uri',
                'collection_name': 'database.collection_name',
                'embedding_dim': 'database.embedding_dim',
                'data_path_glob': 'data.data_path_glob',
                'chunk_size': 'data.chunk_size',
                'chunk_overlap': 'data.chunk_overlap',
                'search_limit': 'search.search_limit',
                'rerank_top_k': 'search.rerank_top_k',
                'batch_size': 'performance.batch_size',
                'max_workers': 'performance.max_workers',
                'cache_size': 'performance.cache_size',
                'system_prompt': 'prompts.system_prompt',
            }
            
            for old_key, new_key in migration_map.items():
                if old_key in old_config:
                    self.set(new_key, old_config[old_key])
            
            logger.info("配置迁移完成")
            return True
            
        except Exception as e:
            logger.error(f"配置迁移失败: {e}")
            return False
    
    # 便捷属性访问
    @property
    def api(self) -> APIConfig:
        return self._config.api
    
    @property
    def database(self) -> DatabaseConfig:
        return self._config.database
    
    @property
    def models(self) -> ModelsConfig:
        return self._config.models
    
    @property
    def data(self) -> DataConfig:
        return self._config.data
    
    @property
    def search(self) -> SearchConfig:
        return self._config.search
    
    @property
    def performance(self) -> PerformanceConfig:
        return self._config.performance
    
    @property
    def logging(self) -> LoggingConfig:
        return self._config.logging
    
    @property
    def output(self) -> OutputConfig:
        return self._config.output
    
    @property
    def prompts(self) -> PromptsConfig:
        return self._config.prompts
    
    @property
    def security(self) -> SecurityConfig:
        return self._config.security
    
    @property
    def monitoring(self) -> MonitoringConfig:
        return self._config.monitoring
    
    def __repr__(self) -> str:
        return f"RAGConfig(config_file='{self._config_file}', valid={self.validate()})"
    
    def __str__(self) -> str:
        return f"RAGConfig(valid={self.validate()})" 