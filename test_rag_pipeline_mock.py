#!/usr/bin/env python3
"""
RAG主流程mock集成测试
"""
import sys
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# 替换Milvus为mock
sys.modules['pymilvus'] = __import__('test_mock_milvus').sys.modules['pymilvus']

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline

# Mock嵌入服务
class MockEmbeddingService:
    def __init__(self, *args, **kwargs): pass
    def encode_batch(self, texts):
        # 返回固定向量
        return [[0.1] * 1536 for _ in texts]
    def encode(self, text):
        return [0.1] * 1536

# Mock LLM服务
class MockLLMService:
    def __init__(self, *args, **kwargs): pass
    def generate_answer(self, question, context, template_name="rag"):
        # 返回固定答案
        return SimpleNamespace(content=f"[MOCK答案] 问题: {question} | 上下文: {context}")

# Mock重排序服务
class MockRerankerService:
    def __init__(self, *args, **kwargs): pass
    def rerank(self, question, candidates, algorithm="llm", top_k=None):
        # 简单分数排序
        from rag.reranker import RerankResult
        return [RerankResult(content=c, score=1.0-0.1*i, algorithm=algorithm, model="mock", processing_time=0.01, metadata={}) for i, c in enumerate(candidates)]

# Patch RAGPipeline依赖
@patch('rag.pipeline.EmbeddingService', MockEmbeddingService)
@patch('rag.pipeline.LLMService', MockLLMService)
@patch('rag.pipeline.RerankerService', MockRerankerService)
def main(*_):
    config = RAGConfig.from_file('test_config.yaml')
    pipeline = RAGPipeline(config)
    pipeline.setup_collection(force_recreate=True)
    # 导入数据
    pipeline._import_data()
    # 测试主流程
    question = "什么是RAG系统？"
    answer = pipeline.run(question)
    print(f"\n最终答案: {answer}")

if __name__ == "__main__":
    main() 