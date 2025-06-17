#!/usr/bin/env python3
"""
RAG系统配置管理模块

提供现代化的配置管理功能，包括：
- 类型安全的配置模型
- 环境变量支持
- 配置验证和热重载
- 多格式配置文件支持
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
from functools import lru_cache
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
import yaml

logger = logging.getLogger(__name__)

class APIConfig(BaseModel):
    """API配置模型"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    openai_api_key: str = Field(default="", description="OpenAI API密钥")
    openai_base_url: str = Field(default="http://10.172.10.103:11434/v1", description="OpenAI兼容API基础URL")
    timeout: int = Field(default=30, ge=1, le=300, description="API超时时间(秒)")
    max_retries: int = Field(default=3, ge=0, le=10, description="最大重试次数")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="重试延迟(秒)")
    
    @field_validator('openai_base_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL必须以http://或https://开头")
        return v.rstrip('/')
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v and os.getenv('OPENAI_API_KEY'):
            return os.getenv('OPENAI_API_KEY', '')
        return v

class DatabaseConfig(BaseModel):
    """数据库配置模型"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    endpoint: str = Field(default="http://localhost:19530", description="Milvus服务Endpoint，格式如http://host:port")
    collection_name: str = Field(default="qwen3_embedding_rag", description="集合名称")
    username: str = Field(default="", description="Milvus用户名")
    password: str = Field(default="", description="Milvus密码")
    timeout: int = Field(default=30, ge=1, le=300, description="连接超时时间(秒)")
    embedding_dim: int = Field(default=1024, description="嵌入向量维度")
    metric_type: str = Field(default="IP", description="距离度量类型")
    index_type: str = Field(default="IVF_FLAT", description="索引类型")
    nlist: int = Field(default=1024, description="聚类数量")
    consistency_level: str = Field(default="Strong", description="一致性级别")
    batch_size: int = Field(default=1000, description="批处理大小")
    ttl_seconds: int = Field(default=0, description="TTL时间(秒)")
    nprobe: int = Field(default=16, description="搜索聚类数")
    m: int = Field(default=4, description="HNSW参数M")
    nbits: int = Field(default=8, description="量化位数")
    # 向后兼容
    milvus_uri: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None

    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://', 'tcp://', 'unix://')):
            raise ValueError("Endpoint必须以http://、https://、tcp://或unix://开头")
        return v.rstrip('/')

    @field_validator('collection_name')
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not v or len(v) > 255:
            raise ValueError("集合名称不能为空且长度不能超过255")
        return v

class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    name: str = Field(default="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0", description="嵌入模型名称")
    dim: int = Field(default=1024, ge=1, le=8192, description="嵌入向量维度")
    batch_size: int = Field(default=32, ge=1, le=128, description="批处理大小")
    max_length: int = Field(default=512, ge=64, le=2048, description="最大文本长度")
    normalize: bool = Field(default=True, description="是否归一化向量")
    cache_size: int = Field(default=1000, ge=0, le=10000, description="缓存大小")
    
    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or len(v) > 256:
            raise ValueError("模型名称不能为空且长度不能超过256个字符")
        return v
    
    @field_validator('dim')
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        common_dims = [384, 512, 768, 1024, 1536, 2048, 4096]
        if v not in common_dims:
            logger.warning(f"嵌入维度 {v} 不是常见值，请确认模型支持")
        return v

class LLMConfig(BaseModel):
    """大语言模型配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    name: str = Field(default="qwen3:4b", description="LLM模型名称")
    temperature: float = Field(default=0.6, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-P参数")
    top_k: int = Field(default=20, ge=1, le=100, description="Top-K参数")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-P参数")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="最大生成token数")
    stop: Optional[List[str]] = Field(default=None, description="停止词列表")
    system_prompt: str = Field(default="你是一个有用的AI助手。", description="系统提示词")
    
    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or len(v) > 256:
            raise ValueError("模型名称不能为空且长度不能超过256个字符")
        return v

class ModelsConfig(BaseModel):
    """模型配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

class DataConfig(BaseModel):
    """数据处理配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    data_path_glob: str = Field(default="milvus_docs/en/faq/*.md", description="数据文件路径模式")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="文本分块大小")
    chunk_overlap: int = Field(default=200, ge=0, le=5000, description="分块重叠大小")
    chunk_strategy: str = Field(default="fixed", description="分块策略")
    file_extensions: List[str] = Field(default=[".md", ".txt", ".json"], description="支持的文件扩展名")
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError("分块重叠大小必须小于分块大小")
        return v
    
    @field_validator('chunk_strategy')
    @classmethod
    def validate_chunk_strategy(cls, v: str) -> str:
        valid_strategies = ["fixed", "semantic", "sentence"]
        if v not in valid_strategies:
            raise ValueError(f"分块策略必须是以下之一: {valid_strategies}")
        return v

class SearchConfig(BaseModel):
    """搜索配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    search_limit: int = Field(default=10, ge=1, le=100, description="搜索结果数量限制")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    rerank_top_k: int = Field(default=5, ge=1, le=20, description="重排序后保留的文档数量")
    use_reranker: bool = Field(default=True, description="是否使用重排序")
    rerank_algorithm: str = Field(default="llm", description="重排序算法")
    
    @field_validator('rerank_top_k')
    @classmethod
    def validate_rerank_top_k(cls, v: int, info) -> int:
        search_limit = info.data.get('search_limit', 10)
        if v > search_limit:
            raise ValueError(f"rerank_top_k不能大于search_limit: {v} > {search_limit}")
        return v
    
    @field_validator('rerank_algorithm')
    @classmethod
    def validate_rerank_algorithm(cls, v: str) -> str:
        valid_algorithms = ['llm', 'keyword', 'semantic', 'hybrid']
        if v not in valid_algorithms:
            raise ValueError(f"重排序算法必须是以下之一: {valid_algorithms}")
        return v

class LoggingConfig(BaseModel):
    """日志配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: str = Field(default="logs/rag_system.log", description="日志文件路径")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="最大日志文件大小(字节)")
    backup_count: int = Field(default=5, description="备份文件数量")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是以下之一: {valid_levels}")
        return v.upper()

class OutputConfig(BaseModel):
    """输出配置"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    output_dir: str = Field(default="answers", description="输出目录")
    output_format: str = Field(default="txt", description="输出格式")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    timestamp_format: str = Field(default="%Y%m%d_%H%M%S", description="时间戳格式")
    
    @field_validator('output_format')
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        valid_formats = ["txt", "json", "md"]
        if v not in valid_formats:
            raise ValueError(f"输出格式必须是以下之一: {valid_formats}")
        return v

class RAGConfigModel(BaseModel):
    """RAG系统完整配置模型"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

class RAGConfig:
    """RAG系统配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None, env: Optional[str] = None):
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        self.config_model: Optional[RAGConfigModel] = None
        self._last_modified: Optional[float] = None
        
        # 检测环境变量或参数
        self.env = env or os.getenv("RAG_ENV", "dev")
        if not config_file:
            # 优先查找rag_config.{env}.json
            env_config = Path(f"rag_config.{self.env}.json")
            if env_config.exists():
                config_file = str(env_config)
                self.config_file = config_file
                logger.info(f"检测到环境变量RAG_ENV={self.env}，加载环境配置文件: {config_file}")
            else:
                default_config = Path("rag_config.json")
                if default_config.exists():
                    config_file = str(default_config)
                    self.config_file = config_file
                    logger.info(f"加载默认配置文件: {config_file}")
        
        if config_file:
            self.load_from_file(config_file)
        else:
            self.config_model = RAGConfigModel()
            self.config_data = self.config_model.model_dump()
        
        # 从环境变量加载配置（优先级最高）
        self._load_from_env()
        
        # 验证配置
        self._validate_config()
        logger.info(f"当前环境: {self.env}，配置文件: {self.config_file}")
    
    @classmethod
    def from_file(cls, config_file: str) -> 'RAGConfig':
        """从文件创建配置实例"""
        return cls(config_file)
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        env_mapping = {
            'OPENAI_API_KEY': 'api.openai_api_key',
            'OPENAI_BASE_URL': 'api.openai_base_url',
            'OPENAI_TIMEOUT': 'api.timeout',
            'MILVUS_URI': 'database.milvus_uri',
            'MILVUS_COLLECTION': 'database.collection_name',
            'MILVUS_USERNAME': 'database.username',
            'MILVUS_PASSWORD': 'database.password',
            'EMBEDDING_MODEL': 'models.embedding.name',
            'LLM_MODEL': 'models.llm.name',
            'LOG_LEVEL': 'logging.log_level',
            'DATA_PATH': 'data.data_path_glob',
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(config_path, value)
    
    def _set_nested_value(self, path: str, value: Any) -> None:
        """设置嵌套配置值"""
        keys = path.split('.')
        current = self.config_data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 尝试类型转换
        try:
            if isinstance(value, str):
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit() and value.count('.') == 1:
                    value = float(value)
        except (ValueError, TypeError):
            pass
        
        current[keys[-1]] = value
    
    def _validate_config(self) -> None:
        """验证配置"""
        try:
            self.config_model = RAGConfigModel(**self.config_data)
        except ValidationError as e:
            logger.error(f"配置验证失败: {e}")
            raise ValueError(f"配置验证失败: {e}")
    
    @lru_cache(maxsize=128)
    def _parse_config_file(self, file_path: Path) -> Dict[str, Any]:
        """解析配置文件"""
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
    
    def load_from_file(self, config_file: str) -> bool:
        """从文件加载配置"""
        try:
            file_path = Path(config_file)
            self.config_data = self._parse_config_file(file_path)
            self._last_modified = file_path.stat().st_mtime
            logger.info(f"配置文件加载成功: {config_file}")
            return True
        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
            raise
    
    def save_to_file(self, config_file: str) -> bool:
        """保存配置到文件"""
        try:
            file_path = Path(config_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
                elif file_path.suffix.lower() == '.json':
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
            
            logger.info(f"配置文件保存成功: {config_file}")
            return True
        except Exception as e:
            logger.error(f"配置文件保存失败: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        current = self.config_data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current
    
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        try:
            self._set_nested_value(key, value)
            self._validate_config()
            return True
        except Exception as e:
            logger.error(f"设置配置值失败: {str(e)}")
            return False
    
    def reload(self) -> bool:
        """重新加载配置"""
        if self.config_file:
            return self.load_from_file(self.config_file)
        return False
    
    def has_changed(self) -> bool:
        """检查配置文件是否已更改"""
        if not self.config_file:
            return False
        
        file_path = Path(self.config_file)
        if not file_path.exists():
            return False
        
        current_mtime = file_path.stat().st_mtime
        return self._last_modified != current_mtime
    
    def print_config(self, detailed: bool = False) -> None:
        """打印配置信息"""
        if detailed:
            config_str = json.dumps(self.config_data, indent=2, ensure_ascii=False)
            print("详细配置:")
            print(config_str)
        else:
            print("配置摘要:")
            print(f"  API基础URL: {self.api.openai_base_url}")
            print(f"  Milvus URI: {self.database.milvus_uri}")
            print(f"  集合名称: {self.database.collection_name}")
            print(f"  嵌入模型: {self.models.embedding.name}")
            print(f"  LLM模型: {self.models.llm.name}")
            print(f"  数据路径: {self.data.data_path_glob}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config_data.copy()
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            self._validate_config()
            return True
        except Exception:
            return False
    
    # 属性访问器
    @property
    def api(self) -> APIConfig:
        return self.config_model.api
    
    @property
    def database(self) -> DatabaseConfig:
        return self.config_model.database
    
    @property
    def models(self) -> ModelsConfig:
        return self.config_model.models
    
    @property
    def data(self) -> DataConfig:
        return self.config_model.data
    
    @property
    def search(self) -> SearchConfig:
        return self.config_model.search
    
    @property
    def logging(self) -> LoggingConfig:
        return self.config_model.logging
    
    @property
    def output(self) -> OutputConfig:
        return self.config_model.output
    
    def __repr__(self) -> str:
        return f"RAGConfig(config_file='{self.config_file}')"
    
    def __str__(self) -> str:
        return f"RAGConfig(config_file='{self.config_file}')" 