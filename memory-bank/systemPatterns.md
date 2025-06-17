# 系统模式 (System Patterns)

## 架构概览
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档处理      │    │   嵌入服务      │    │   Milvus服务    │
│  Document       │───▶│  Embedding      │───▶│  MilvusService  │
│  Processor      │    │  Service        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   重排序服务    │    │   LLM服务       │    │   主管道        │
│  Reranker       │    │  LLMService     │    │  Pipeline       │
│  Service        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   CLI界面       │
                       │  Click + Rich   │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   主入口点      │
                       │   main.py       │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   配置管理      │
                       │  rag_config.json│
                       └─────────────────┘
```

## 核心设计模式

### 1. 管道模式 (Pipeline Pattern)
- **目的**: 将复杂的RAG流程分解为可组合的步骤
- **实现**: `Pipeline`类协调各个服务组件
- **优势**: 易于测试、扩展和维护

### 2. 服务层模式 (Service Layer Pattern)
- **目的**: 封装外部依赖和业务逻辑
- **实现**: 各个Service类（EmbeddingService、LLMService等）
- **优势**: 松耦合、可替换、易于测试

### 3. 配置驱动模式 (Configuration-Driven Pattern)
- **目的**: 通过配置文件控制系统行为
- **实现**: Pydantic配置模型和热重载机制
- **优势**: 灵活性、可维护性、环境隔离

### 4. 异步模式 (Async Pattern)
- **目的**: 提高并发性能和响应速度
- **实现**: 异步方法和协程
- **优势**: 高并发、非阻塞、资源效率

### 5. 单一入口点模式 (Single Entry Point Pattern)
- **目的**: 确保系统有清晰的启动路径
- **实现**: main.py作为唯一主入口点
- **优势**: 避免混乱、便于维护、清晰的错误处理

### 6. 自动配置加载模式 (Auto-Config Loading Pattern)
- **目的**: 简化用户使用，自动加载默认配置
- **实现**: 自动查找rag_config.json
- **优势**: 用户体验友好、减少配置复杂度

### 7. 远程连接模式 (Remote Connection Pattern)
- **目的**: 支持分布式部署和远程服务
- **实现**: endpoint统一管理连接配置
- **优势**: 生产就绪、支持云部署

## 组件关系

### 数据流
1. **文档输入** → DocumentProcessor
2. **文本分块** → 嵌入向量化 → Milvus存储
3. **查询处理** → 向量检索 → 重排序 → LLM生成
4. **用户交互** → CLI界面 → 主入口点
5. **配置管理** → 自动加载 → 服务初始化

### 依赖关系
- Pipeline 依赖所有Service组件
- Service组件相互独立
- 配置系统贯穿所有组件
- CLI界面依赖Pipeline
- 主入口点统一管理所有功能
- 配置管理支持自动加载和手动指定

## 扩展点

### 1. 嵌入模型扩展
```python
class CustomEmbeddingService(EmbeddingService):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # 自定义嵌入逻辑
        pass
```

### 2. 重排序算法扩展
```python
class CustomReranker(RerankerService):
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        # 自定义重排序逻辑
        pass
```

### 3. LLM提供商扩展
```python
class CustomLLMService(LLMService):
    def generate_answer(self, query: str, context: str) -> str:
        # 自定义LLM调用逻辑
        pass
```

### 4. CLI命令扩展
```python
@cli.command()
def custom_command():
    """自定义CLI命令"""
    pass
```

### 5. 配置扩展
```python
# 支持新的配置格式和环境
class CustomConfig(RAGConfig):
    def load_from_custom_source(self):
        # 自定义配置加载逻辑
        pass
```

## 错误处理模式

### 1. 重试机制
- 网络请求失败自动重试
- 指数退避策略
- 最大重试次数限制

### 2. 降级策略
- 服务不可用时使用备用方案
- 缓存机制减少依赖
- 优雅的错误响应

### 3. 监控和日志
- 结构化日志记录
- 性能指标收集
- 错误追踪和报警

### 4. 入口点错误处理
- 导入错误检测和用户友好提示
- 模块依赖验证
- 清晰的错误信息显示

### 5. 配置错误处理
- 配置验证和类型检查
- 默认值回退机制
- 用户友好的错误提示

## 性能优化模式

### 1. 缓存策略
- LRU缓存减少重复计算
- 向量缓存避免重复嵌入
- 结果缓存提升响应速度

### 2. 批处理
- 批量嵌入处理
- 批量数据库操作
- 并发处理提升吞吐量

### 3. 资源管理
- 连接池管理
- 内存使用优化
- 异步IO减少阻塞

## 入口点管理

### 1. 单一入口点原则
- main.py作为唯一主入口点
- 移除其他模块的重复入口点
- 统一的错误处理和用户界面

### 2. 模块导入管理
- 清晰的导入路径
- 避免循环导入
- 导入错误检测和处理

### 3. CLI界面统一
- Click + Rich提供现代化CLI体验
- 统一的命令结构和帮助信息
- 用户友好的错误提示

## 配置管理模式

### 1. 自动配置加载
- 默认查找rag_config.json
- 支持手动指定配置文件
- 向后兼容旧配置格式

### 2. 配置验证
- Pydantic模型验证
- 类型安全和范围检查
- 用户友好的错误信息

### 3. 环境适配
- 支持多环境配置
- 环境变量集成
- 配置热重载

### 4. 远程连接支持
- endpoint统一管理
- 支持多种协议
- 连接状态检测 