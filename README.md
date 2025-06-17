# Qwen3 Embedding RAG 系统

基于 Qwen3 嵌入模型和 Milvus 向量数据库的检索增强生成（RAG）系统。

## 🚀 功能特性

- **多模型支持**: 支持 Qwen3 嵌入模型、重排序模型和 LLM
- **高效检索**: 基于 Milvus 向量数据库的相似性搜索
- **智能重排序**: 使用重排序模型优化检索结果
- **批量处理**: 支持并发处理和批量操作
- **缓存机制**: 内置缓存提升性能
- **模块化设计**: 清晰的代码结构和易于扩展
- **高级配置管理**: 支持JSON/YAML格式、环境变量、配置验证和热重载

## 📁 项目结构

```
qwen3-embedding-rag/
├── rag/                    # 主程序包
│   ├── __init__.py
│   ├── config.py          # 配置管理（已优化）
│   ├── document.py        # 文档处理
│   ├── embedding.py       # 嵌入服务
│   ├── llm.py            # LLM 服务
│   ├── main.py           # 主程序入口
│   ├── milvus_service.py # Milvus 服务
│   ├── pipeline.py       # RAG 管道
│   ├── reranker.py       # 重排序服务
│   └── utils.py          # 工具函数
├── scripts/               # 脚本目录
│   ├── config_manager.py # 配置管理工具
│   ├── config_example.py # 配置使用示例
│   ├── example_usage.py  # 使用示例
│   └── performance_monitor.py # 性能监控
├── docs/                  # 文档目录
│   └── configuration.md  # 配置指南
├── milvus_docs/          # 文档数据
├── answers/              # 答案输出目录
├── logs/                 # 日志输出目录
├── main.py              # 根目录启动脚本
├── config.json          # 配置文件（已优化）
├── config.template.yaml # YAML配置模板
├── requirements.txt     # 依赖包
└── README.md           # 项目说明
```

## 🛠️ 安装配置

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd qwen3-embedding-rag

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

#### 方法一：使用配置模板

```bash
# 生成配置模板
python scripts/config_manager.py template config.yaml --format yaml

# 编辑配置文件
vim config.yaml
```

#### 方法二：使用环境变量

```bash
# 生成环境变量模板
python scripts/config_manager.py env .env.template

# 复制并编辑环境变量文件
cp .env.template .env
vim .env
```

#### 方法三：直接编辑配置文件

编辑 `config.json` 文件：

```json
{
  "api": {
    "openai_api_key": "",
    "openai_base_url": "http://10.172.10.103:11434/v1"
  },
  "database": {
    "milvus_uri": "http://your-milvus-server:19530",
    "collection_name": "qwen3_embedding_rag"
  },
  "models": {
    "embedding_model": "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
    "reranker_model": "qwen3:4b",
    "llm_model": "qwen3:4b"
  }
}
```

### 3. 配置验证

```bash
# 验证配置文件
python scripts/config_manager.py validate config.json

# 显示配置信息
python scripts/config_manager.py info config.json
```

### 4. 启动服务

确保以下服务正在运行：
- **Ollama**: 运行 Qwen3 模型
- **Milvus**: 向量数据库服务

## 🚀 使用方法

### 命令行模式

```bash
# 单次提问
python main.py --question "你的问题" --output-file "答案文件.txt"

# 强制重建 Milvus 集合
python main.py --force-recreate --question "你的问题"

# 指定配置文件
python main.py --config custom_config.json --question "你的问题"
```

### 交互模式

```bash
python main.py
# 然后输入问题，答案会自动保存到 answers/ 目录
```

### 参数说明

- `--question, -q`: 要回答的问题
- `--force-recreate, -f`: 强制重建 Milvus 集合
- `--output-file, -o`: 答案输出文件路径（默认: answer.txt）
- `--config`: 配置文件路径
- `--log-level`: 日志级别（默认: INFO）

## ⚙️ 配置管理

### 配置管理工具

```bash
# 验证配置
python scripts/config_manager.py validate config.json

# 转换配置格式（JSON ↔ YAML）
python scripts/config_manager.py convert config.json config.yaml

# 生成配置模板
python scripts/config_manager.py template config.yaml --format yaml

# 显示配置信息
python scripts/config_manager.py info config.json

# 生成环境变量模板
python scripts/config_manager.py env .env.template
```

### 配置使用示例

```bash
# 运行配置示例
python scripts/config_example.py
```

### 环境变量支持

| 环境变量 | 配置路径 | 描述 |
|---------|---------|------|
| `OPENAI_API_KEY` | `api.openai_api_key` | OpenAI API密钥 |
| `OPENAI_BASE_URL` | `api.openai_base_url` | OpenAI兼容API基础URL |
| `MILVUS_URI` | `database.milvus_uri` | Milvus服务URI |
| `MILVUS_COLLECTION` | `database.collection_name` | Milvus集合名称 |
| `LOG_LEVEL` | `logging.log_level` | 日志级别 |

## 📊 输出文件

### 答案文件
- **位置**: `answers/` 目录
- **格式**: 
  ```
  问题: [用户问题]
  答案: [完整答案内容]
  
  生成时间: 2025-06-17 13:08:03
  ```

### 日志文件
- **位置**: `logs/` 目录
- **文件**: `rag_system.log`
- **内容**: 系统运行日志，包括配置信息、处理进度等

## 🔧 配置选项

### API配置 (`api`)
| 配置项 | 说明 | 默认值 | 范围 |
|--------|------|--------|------|
| `openai_api_key` | OpenAI API密钥 | `""` | - |
| `openai_base_url` | API基础URL | `http://10.172.10.103:11434/v1` | - |
| `timeout` | API超时时间(秒) | `30` | 1-300 |
| `max_retries` | 最大重试次数 | `3` | 0-10 |
| `retry_delay` | 重试延迟(秒) | `1.0` | 0.1-60.0 |

### 数据库配置 (`database`)
| 配置项 | 说明 | 默认值 | 范围 |
|--------|------|--------|------|
| `milvus_uri` | Milvus服务URI | `http://10.172.10.100:19530` | - |
| `collection_name` | 集合名称 | `qwen3_embedding_rag` | - |
| `embedding_dim` | 嵌入向量维度 | `1024` | 1-8192 |
| `metric_type` | 距离度量类型 | `IP` | IP/L2/COSINE |
| `consistency_level` | 一致性级别 | `Strong` | Strong/Bounded/Eventually/Session |
| `index_type` | 索引类型 | `IVF_FLAT` | IVF_FLAT/IVF_SQ8/HNSW等 |
| `nlist` | 聚类数量 | `1024` | ≥1 |
| `nprobe` | 搜索聚类数 | `16` | ≥1 |

### 模型配置 (`models`)
| 配置项 | 说明 | 默认值 | 范围 |
|--------|------|--------|------|
| `embedding_model` | 嵌入模型 | `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` | - |
| `reranker_model` | 重排序模型 | `qwen3:4b` | - |
| `llm_model` | 大语言模型 | `qwen3:4b` | - |
| `embedding_batch_size` | 嵌入批处理大小 | `32` | 1-128 |
| `llm_max_tokens` | LLM最大token数 | `2048` | 1-8192 |
| `llm_temperature` | LLM温度参数 | `0.7` | 0.0-2.0 |

### 数据处理配置 (`data`)
| 配置项 | 说明 | 默认值 | 范围 |
|--------|------|--------|------|
| `data_path_glob` | 数据文件路径模式 | `milvus_docs/en/faq/*.md` | - |
| `chunk_size` | 文本分块大小 | `1000` | 100-10000 |
| `chunk_overlap` | 分块重叠大小 | `200` | 0-5000 |
| `supported_formats` | 支持的文件格式 | `[".md", ".txt", ".pdf"]` | - |
| `encoding` | 文件编码 | `utf-8` | - |

### 搜索配置 (`search`)
| 配置项 | 说明 | 默认值 | 范围 |
|--------|------|--------|------|
| `search_limit` | 搜索结果数量限制 | `10` | 1-100 |
| `rerank_top_k` | 重排序top-k | `3` | 1-20 |
| `similarity_threshold` | 相似度阈值 | `0.7` | 0.0-1.0 |
| `enable_rerank` | 是否启用重排序 | `true` | true/false |
| `enable_hybrid_search` | 是否启用混合搜索 | `false` | true/false |

### 性能配置 (`performance`)
| 配置项 | 说明 | 默认值 | 范围 |
|--------|------|--------|------|
| `cache_size` | 缓存大小 | `1000` | 100-10000 |
| `max_workers` | 最大工作线程数 | `4` | 1-32 |
| `batch_size` | 批处理大小 | `32` | 1-256 |
| `enable_gpu` | 是否启用GPU | `false` | true/false |
| `gpu_memory_fraction` | GPU内存使用比例 | `0.8` | 0.1-1.0 |

## 🎯 性能优化

- **并发处理**: 支持多线程并发处理
- **批量操作**: 批量处理文档和嵌入
- **缓存机制**: 内置缓存减少重复计算
- **索引优化**: 自动创建和优化 Milvus 索引
- **配置热重载**: 支持运行时重新加载配置
- **GPU加速**: 支持GPU加速计算

## 🔍 故障排除

### 常见问题

1. **Milvus 连接失败**
   - 检查 Milvus 服务是否启动
   - 验证连接地址和端口

2. **模型加载失败**
   - 确保 Ollama 服务运行
   - 检查模型名称是否正确

3. **内存不足**
   - 减少批处理大小
   - 调整并发数量

4. **配置验证失败**
   - 使用配置管理工具验证配置
   - 检查数值是否在有效范围内
   - 确认逻辑约束是否满足

### 日志查看

```bash
# 查看最新日志
tail -f logs/rag_system.log

# 查看错误日志
grep ERROR logs/rag_system.log
```

### 配置调试

```bash
# 验证配置文件
python scripts/config_manager.py validate config.json

# 显示配置信息
python scripts/config_manager.py info config.json

# 运行配置示例
python scripts/config_example.py
```

## 📝 开发说明

### 代码结构

- **模块化设计**: 每个功能模块独立封装
- **配置驱动**: 通过配置文件控制行为
- **错误处理**: 完善的异常处理机制
- **日志记录**: 详细的日志记录便于调试
- **配置验证**: 完整的配置验证和类型检查

### 扩展开发

1. **添加新模型**: 在对应服务类中添加模型支持
2. **自定义处理器**: 继承基础类实现自定义逻辑
3. **配置扩展**: 在配置类中添加新的配置项
4. **配置验证**: 添加自定义验证规则

### 配置系统特性

- **类型安全**: 使用Pydantic进行类型验证
- **环境变量支持**: 支持环境变量覆盖配置
- **热重载**: 支持运行时重新加载配置
- **多格式支持**: 支持JSON和YAML格式
- **配置验证**: 完整的配置验证和约束检查
- **配置管理工具**: 提供配置验证、转换、生成工具

## 📄 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请查看：
- [项目 Issues](../../issues)
- [配置指南](./docs/configuration.md)
- [文档说明](./docs/)