# Qwen3 Embedding RAG 系统

基于 Qwen3 Embedding 和 Milvus 向量数据库的检索增强生成 (RAG) 系统。

## 功能特性

- 🔍 **智能文档检索**: 使用向量相似度搜索相关文档
- 🎯 **重排序优化**: 通过重排序模型提高检索精度
- 🤖 **智能问答**: 基于检索结果生成准确答案
- ⚡ **高性能**: 支持并发处理和批量操作
- 🔧 **易于配置**: 灵活的配置系统
- 📊 **性能监控**: 内置性能统计和监控

## 系统架构

```
用户问题 → Embedding生成 → 向量检索 → 重排序 → 答案生成
    ↓           ↓           ↓         ↓         ↓
  输入处理    OpenAI API   Milvus   重排序模型   LLM模型
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- Milvus 向量数据库
- OpenAI 兼容的 API 服务

### 2. 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 3. 配置环境

创建 `.env` 文件或设置环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://your-ollama-server:11434/v1"
export MILVUS_URI="http://your-milvus-server:19530"
```

### 4. 运行系统

```bash
# 基本使用
python main.py --question "你的问题"

# 重建集合
python main.py --rebuild --question "你的问题"

# 查看帮助
python main.py --help
```

## 配置说明

主要配置项 (`config.json`):

```json
{
  "embedding_model": "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
  "reranker_model": "qwen3:4b",
  "llm_model": "qwen3:4b",
  "embedding_dim": 1024,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "search_limit": 10,
  "rerank_top_k": 3
}
```

## 使用示例

### 基本使用

```python
from rag.config import RAGConfig
from rag.pipeline import RAGPipeline

# 创建配置
config = RAGConfig()

# 创建管道
pipeline = RAGPipeline(config)

# 设置集合
pipeline.setup_collection()

# 运行查询
answer = pipeline.run("你的问题")
print(answer)
```

### 自定义配置

```python
config = RAGConfig(
    embedding_model="your-embedding-model",
    reranker_model="your-reranker-model",
    llm_model="your-llm-model",
    embedding_dim=1024,
    max_workers=4,
    cache_size=1000
)
```

## 性能监控

使用性能监控脚本查看系统性能：

```bash
# 生成性能报告
python scripts/performance_monitor.py --report

# 清除性能记录
python scripts/performance_monitor.py --clear
```

## 项目结构

```
qwen3-embedding-rag/
├── rag/                    # 主程序包
│   ├── __init__.py
│   ├── main.py             # 主入口
│   ├── config.py           # 配置管理
│   ├── document.py         # 文档处理
│   ├── embedding.py        # Embedding服务
│   ├── reranker.py         # 重排序服务
│   ├── llm.py              # LLM服务
│   ├── milvus_service.py   # Milvus服务
│   ├── pipeline.py         # RAG管道
│   └── utils.py            # 工具函数
├── scripts/                # 脚本工具
│   ├── example_usage.py    # 使用示例
│   └── performance_monitor.py # 性能监控
├── main.py                 # 启动脚本
├── config.json            # 配置文件
├── requirements.txt        # 依赖文件
├── pyproject.toml         # 项目配置
├── README.md              # 说明文档
└── milvus_docs/           # 示例文档
```

## 核心组件

### DocumentProcessor
- 文档加载和分块
- 智能文本分割

### EmbeddingService
- 向量嵌入生成
- 批量处理和缓存

### RerankerService
- 文档重排序
- 相关性评分

### LLMService
- 答案生成
- 上下文处理

### MilvusService
- 向量数据库操作
- 集合管理

### RAGPipeline
- 统一调度
- 流程管理

## 故障排除

### 常见问题

1. **维度不匹配错误**
   - 检查 `embedding_dim` 配置
   - 重建 Milvus 集合

2. **API 连接失败**
   - 检查 API 地址和密钥
   - 确认服务可用性

3. **性能问题**
   - 调整并发数和批处理大小
   - 检查网络延迟

## 开发

### 代码风格

项目使用 Black 进行代码格式化：

```bash
black .
```

### 测试

```bash
pytest
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！