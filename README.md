# Qwen3 Embedding RAG 系统

一个基于Qwen3嵌入模型和Milvus向量数据库的高性能RAG（检索增强生成）系统。

## 🚀 主要特性

### 核心功能
- **智能文档处理**: 支持多种文件格式，智能分块策略（固定长度、语义分块、句子级分块）
- **高性能向量搜索**: 基于Milvus的快速相似度搜索
- **多种重排序算法**: LLM重排序、关键词匹配、语义相似度、混合算法
- **流式输出支持**: 实时生成答案，提升用户体验
- **批量处理**: 支持批量文档导入和查询处理

### 技术特性
- **异步处理**: 全异步架构，提高并发性能
- **错误处理和重试**: 智能重试机制，提高系统稳定性
- **性能监控**: 实时性能指标监控和报告生成
- **缓存机制**: 智能缓存，减少重复计算
- **资源管理**: 自动资源清理和连接池管理

### 开发体验
- **现代化CLI**: 基于Click和Rich的友好命令行界面
- **配置管理**: 支持多种配置格式，环境变量支持
- **代码质量**: 完整的代码质量工具链（flake8、mypy、black、isort）
- **测试框架**: 单元测试和集成测试支持

## 📦 安装

### 环境要求
- Python 3.8+
- Milvus 2.3+
- OpenAI API密钥

### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd qwen3-embedding-rag
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境**
```bash
cp config.template.yaml config.yaml
# 编辑配置文件，填入你的API密钥和Milvus连接信息
```

## ⚙️ 配置

### 配置文件结构
```yaml
api:
  openai_api_key: "your-api-key"
  openai_base_url: "https://api.openai.com/v1"
  timeout: 30

models:
  embedding:
    name: "text-embedding-3-small"
    dim: 1536
    batch_size: 100
  llm:
    name: "gpt-3.5-turbo"
    temperature: 0.7
    max_tokens: 1000
    system_prompt: "你是一个专业的AI助手..."

database:
  host: "localhost"
  port: 19530
  collection_name: "rag_documents"
  embedding_dim: 1536
  metric_type: "COSINE"
  index_type: "IVF_FLAT"
  nlist: 1024

data:
  data_path_glob: "docs/**/*.md"
  file_extensions: [".md", ".txt", ".py", ".js"]
  chunk_size: 1000
  chunk_overlap: 200
  chunk_strategy: "semantic"  # fixed, semantic, sentence

search:
  search_limit: 10
  use_reranker: true
  rerank_top_k: 5
  rerank_algorithm: "hybrid"  # llm, keyword, semantic, hybrid

logging:
  log_level: "INFO"
  log_file: "logs/rag.log"
```

### 环境变量支持
```bash
export OPENAI_API_KEY="your-api-key"
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
```

## 🚀 使用

### 基本使用

1. **初始化系统**
```bash
python -m rag.cli setup --force-recreate
```

2. **交互式问答**
```bash
python -m rag.cli chat
```

3. **单次查询**
```bash
python -m rag.cli query "什么是RAG系统？"
```

### 高级功能

1. **批量导入文档**
```bash
python -m rag.cli import --path "docs/" --strategy semantic
```

2. **性能监控**
```bash
# 实时监控
python scripts/performance_monitor.py --mode monitor --interval 30

# 负载测试
python scripts/performance_monitor.py --mode load-test --duration 300
```

3. **配置管理**
```bash
# 验证配置
python -m rag.cli config validate

# 生成配置模板
python -m rag.cli config generate
```

### 编程接口

```python
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig

# 初始化
config = RAGConfig.from_file("config.yaml")
pipeline = RAGPipeline(config)

# 设置集合
pipeline.setup_collection()

# 查询
response = pipeline.run("什么是RAG系统？")
print(response.answer)
print(f"置信度: {response.confidence}")
print(f"处理时间: {response.processing_time:.2f}秒")
```

## 🔧 开发

### 代码质量检查
```bash
# 格式化代码
make format

# 类型检查
make type-check

# 代码质量检查
make lint

# 运行测试
make test
```

### 添加新功能

1. **添加新的重排序算法**
```python
def custom_rerank(question: str, candidates: List[str]) -> List[RerankResult]:
    # 实现自定义重排序逻辑
    pass

reranker_service.add_rerank_algorithm("custom", custom_rerank)
```

2. **添加新的提示词模板**
```python
from rag.llm import PromptTemplate

template = PromptTemplate(
    name="custom",
    template="自定义提示词模板: {question}",
    variables=["question"],
    description="自定义模板"
)

llm_service.add_prompt_template(template)
```

## 📊 性能优化

### 系统调优建议

1. **Milvus配置优化**
   - 根据数据量调整`nlist`参数
   - 选择合适的索引类型（IVF_FLAT、HNSW等）
   - 调整`nprobe`参数平衡精度和速度

2. **分块策略选择**
   - `fixed`: 适合结构化文档
   - `semantic`: 适合长文档和段落
   - `sentence`: 适合需要精确句子级别的场景

3. **重排序算法选择**
   - `llm`: 精度最高，速度较慢
   - `keyword`: 速度快，适合关键词匹配
   - `semantic`: 平衡精度和速度
   - `hybrid`: 综合多种算法

### 监控指标

- **响应时间**: 平均查询响应时间
- **成功率**: 查询成功率
- **资源使用**: CPU和内存使用率
- **向量搜索性能**: 搜索延迟和召回率

## 🐛 故障排除

### 常见问题

1. **Milvus连接失败**
   - 检查Milvus服务是否启动
   - 验证连接参数（host、port）
   - 检查网络连接

2. **API调用失败**
   - 验证OpenAI API密钥
   - 检查网络连接
   - 查看API配额限制

3. **文档导入失败**
   - 检查文件路径和权限
   - 验证文件格式支持
   - 查看日志文件

### 日志分析
```bash
# 查看实时日志
tail -f logs/rag.log

# 搜索错误日志
grep "ERROR" logs/rag.log

# 查看性能日志
python scripts/performance_monitor.py --report
```

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 运行测试
5. 提交Pull Request

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装预提交钩子
pre-commit install

# 运行所有检查
make all
```

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 强大的大语言模型
- [Milvus](https://milvus.io/) - 高性能向量数据库
- [OpenAI](https://openai.com/) - API服务提供商

## 📞 支持

如有问题或建议，请：

1. 查看[文档](docs/)
2. 搜索[Issues](../../issues)
3. 创建新的Issue
4. 联系维护者

---

**注意**: 这是一个开发中的项目，API可能会发生变化。请查看[CHANGELOG](CHANGELOG.md)了解最新更新。

## 配置文件命名规范

- **主配置文件**：`rag_config.json`
  - 推荐用于生产或默认环境。
- **远程Milvus配置模板**：`rag_config.remote_milvus.json`
  - 推荐用于远程Milvus部署场景，可复制为主配置后使用。
- **本地/开发/测试环境**：可命名为 `rag_config.dev.json`、`rag_config.test.json`、`rag_config.local.json` 等。
  - 便于多环境切换和管理。

> 启动时可通过 `-c` 或 `--config` 参数指定配置文件路径，例如：
> ```bash
> python main.py --config rag_config.remote_milvus.json status
> ```

## 其他说明
- 所有配置模板均可根据实际需求进行扩展和调整。
- 配置项详细说明请参考 `docs/configuration.md` 或内嵌注释。