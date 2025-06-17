# RAG系统配置指南

## 概述

RAG系统使用分层配置管理，支持JSON和YAML格式的配置文件，同时支持环境变量覆盖。配置系统提供了完整的验证、类型检查和热重载功能。

## 配置文件结构

### 1. API配置 (`api`)

```yaml
api:
  openai_api_key: ""  # OpenAI API密钥
  openai_base_url: "http://10.172.10.103:11434/v1"  # API基础URL
  timeout: 30  # 超时时间(秒)
  max_retries: 3  # 最大重试次数
  retry_delay: 1.0  # 重试延迟(秒)
```

### 2. 数据库配置 (`database`)

```yaml
database:
  milvus_uri: "http://10.172.10.100:19530"  # Milvus服务URI
  collection_name: "qwen3_embedding_rag"  # 集合名称
  embedding_dim: 1024  # 嵌入向量维度
  metric_type: "IP"  # 距离度量类型
  consistency_level: "Strong"  # 一致性级别
  index_type: "IVF_FLAT"  # 索引类型
  nlist: 1024  # 聚类数量
  nprobe: 16  # 搜索聚类数
```

### 3. 模型配置 (`models`)

```yaml
models:
  embedding_model: "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"  # 嵌入模型
  reranker_model: "qwen3:4b"  # 重排序模型
  llm_model: "qwen3:4b"  # 大语言模型
  embedding_batch_size: 32  # 嵌入批处理大小
  llm_max_tokens: 2048  # LLM最大token数
  llm_temperature: 0.7  # LLM温度参数
```

### 4. 数据处理配置 (`data`)

```yaml
data:
  data_path_glob: "milvus_docs/en/faq/*.md"  # 数据文件路径模式
  chunk_size: 1000  # 文本分块大小
  chunk_overlap: 200  # 分块重叠大小
  supported_formats: [".md", ".txt", ".pdf"]  # 支持的文件格式
  encoding: "utf-8"  # 文件编码
```

### 5. 搜索配置 (`search`)

```yaml
search:
  search_limit: 10  # 搜索结果数量限制
  rerank_top_k: 3  # 重排序top-k
  similarity_threshold: 0.7  # 相似度阈值
  enable_rerank: true  # 是否启用重排序
  enable_hybrid_search: false  # 是否启用混合搜索
```

### 6. 性能配置 (`performance`)

```yaml
performance:
  cache_size: 1000  # 缓存大小
  max_workers: 4  # 最大工作线程数
  batch_size: 32  # 批处理大小
  enable_gpu: false  # 是否启用GPU
  gpu_memory_fraction: 0.8  # GPU内存使用比例
```

### 7. 日志配置 (`logging`)

```yaml
logging:
  log_level: "INFO"  # 日志级别
  log_file: "logs/rag_system.log"  # 日志文件路径
  max_log_size: "10MB"  # 最大日志文件大小
  backup_count: 5  # 备份文件数量
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # 日志格式
```

### 8. 输出配置 (`output`)

```yaml
output:
  output_dir: "answers"  # 输出目录
  save_intermediate_results: true  # 是否保存中间结果
  include_metadata: true  # 是否包含元数据
  output_format: "txt"  # 输出格式
```

### 9. 提示词配置 (`prompts`)

```yaml
prompts:
  system_prompt: "你是一个专业的AI助手，请基于提供的上下文信息准确回答问题。"
  query_prompt: "基于以下上下文信息回答问题：\n\n{context}\n\n问题：{question}\n\n答案："
  rerank_prompt: "请对以下候选答案进行重新排序，选择最相关的答案：\n\n{answers}\n\n问题：{question}"
```

### 10. 安全配置 (`security`)

```yaml
security:
  enable_ssl_verification: true  # 是否启用SSL验证
  api_key_rotation_days: 30  # API密钥轮换天数
  max_request_size: "10MB"  # 最大请求大小
  rate_limit_per_minute: 60  # 每分钟请求限制
```

### 11. 监控配置 (`monitoring`)

```yaml
monitoring:
  enable_metrics: true  # 是否启用指标收集
  metrics_port: 8000  # 指标服务端口
  health_check_interval: 30  # 健康检查间隔(秒)
  performance_monitoring: true  # 是否启用性能监控
```

## 环境变量支持

系统支持以下环境变量来覆盖配置文件中的设置：

| 环境变量 | 配置路径 | 描述 |
|---------|---------|------|
| `OPENAI_API_KEY` | `api.openai_api_key` | OpenAI API密钥 |
| `OPENAI_BASE_URL` | `api.openai_base_url` | OpenAI兼容API基础URL |
| `MILVUS_URI` | `database.milvus_uri` | Milvus服务URI |
| `MILVUS_COLLECTION` | `database.collection_name` | Milvus集合名称 |
| `LOG_LEVEL` | `logging.log_level` | 日志级别 |

## 配置管理工具

### 1. 验证配置

```bash
python scripts/config_manager.py validate config.json
```

### 2. 转换配置格式

```bash
# JSON转YAML
python scripts/config_manager.py convert config.json config.yaml

# YAML转JSON
python scripts/config_manager.py convert config.yaml config.json
```

### 3. 生成配置模板

```bash
# 生成JSON模板
python scripts/config_manager.py template config.json

# 生成YAML模板
python scripts/config_manager.py template config.yaml --format yaml
```

### 4. 显示配置信息

```bash
python scripts/config_manager.py info config.json
```

### 5. 生成环境变量模板

```bash
python scripts/config_manager.py env .env.template
```

## 配置验证规则

### 数值范围验证

- `api.timeout`: 1-300秒
- `api.max_retries`: 0-10次
- `api.retry_delay`: 0.1-60.0秒
- `database.embedding_dim`: 1-8192
- `database.nlist`: ≥1
- `database.nprobe`: ≥1
- `models.embedding_batch_size`: 1-128
- `models.llm_max_tokens`: 1-8192
- `models.llm_temperature`: 0.0-2.0
- `data.chunk_size`: 100-10000
- `data.chunk_overlap`: 0-5000
- `search.search_limit`: 1-100
- `search.rerank_top_k`: 1-20
- `search.similarity_threshold`: 0.0-1.0
- `performance.cache_size`: 100-10000
- `performance.max_workers`: 1-32
- `performance.batch_size`: 1-256
- `performance.gpu_memory_fraction`: 0.1-1.0
- `logging.backup_count`: 0-20
- `security.api_key_rotation_days`: 1-365
- `security.rate_limit_per_minute`: 1-1000
- `monitoring.metrics_port`: 1024-65535
- `monitoring.health_check_interval`: 5-300

### 逻辑验证

- `data.chunk_overlap` 必须小于 `data.chunk_size`
- `search.rerank_top_k` 不能大于 `search.search_limit`

## 使用示例

### 1. 基本使用

```python
from rag.config import RAGConfig

# 使用默认配置
config = RAGConfig()

# 从文件加载配置
config.load_from_file("config.json")

# 获取配置值
api_key = config.get("api.openai_api_key")
milvus_uri = config.database.milvus_uri
```

### 2. 动态修改配置

```python
# 设置配置值
config.set("api.timeout", 60)
config.set("models.llm_temperature", 0.5)

# 验证配置
if config.validate():
    print("配置有效")
```

### 3. 配置热重载

```python
# 重新加载配置文件
if config.reload():
    print("配置已重新加载")
```

### 4. 保存配置

```python
# 保存到文件
config.save_to_file("config_backup.json")
```

## 最佳实践

### 1. 配置文件管理

- 使用版本控制管理配置文件
- 为不同环境创建不同的配置文件
- 敏感信息使用环境变量而不是配置文件

### 2. 配置验证

- 在部署前验证配置文件
- 使用配置管理工具检查配置有效性
- 定期检查配置是否符合最佳实践

### 3. 性能优化

- 根据硬件资源调整 `performance` 配置
- 监控系统性能并相应调整配置
- 使用GPU加速时注意内存使用

### 4. 安全考虑

- 定期轮换API密钥
- 启用SSL验证
- 设置合理的请求限制

## 故障排除

### 常见问题

1. **配置验证失败**
   - 检查数值是否在有效范围内
   - 确认逻辑约束是否满足
   - 查看错误信息中的具体字段

2. **环境变量不生效**
   - 确认环境变量名称正确
   - 检查环境变量是否已设置
   - 重启应用程序

3. **配置文件加载失败**
   - 检查文件路径是否正确
   - 确认文件格式是否支持
   - 验证文件权限

4. **配置热重载不工作**
   - 确认配置文件路径已设置
   - 检查文件修改时间
   - 查看日志中的错误信息

### 调试技巧

- 使用 `config.print_config()` 查看当前配置
- 启用详细日志记录
- 使用配置管理工具验证配置
- 检查环境变量设置 