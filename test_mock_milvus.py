#!/usr/bin/env python3
"""
模拟Milvus服务测试脚本
"""

import time
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock
import types

# 模拟Milvus客户端
class MockMilvusClient:
    def __init__(self, uri=None, token=None, **kwargs):
        self.collections = {}
        self.indexes = {}
        self.data = {}
        self.uri = uri
        self.token = token
        print(f"模拟Milvus客户端初始化: {uri}")
    
    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections
    
    def create_collection(self, collection_name: str, **kwargs):
        self.collections[collection_name] = kwargs
        self.data[collection_name] = []
        print(f"模拟创建集合: {collection_name}")
    
    def drop_collection(self, collection_name: str):
        if collection_name in self.collections:
            del self.collections[collection_name]
            del self.data[collection_name]
            print(f"模拟删除集合: {collection_name}")
    
    def has_index(self, collection_name: str) -> bool:
        return collection_name in self.indexes
    
    def create_index(self, collection_name: str, **kwargs):
        self.indexes[collection_name] = kwargs
        print(f"模拟创建索引: {collection_name}")
    
    def insert(self, collection_name: str, data: List[Dict]):
        if collection_name not in self.data:
            self.data[collection_name] = []
        self.data[collection_name].extend(data)
        print(f"模拟插入 {len(data)} 条数据到集合: {collection_name}")
    
    def search(self, collection_name: str, **kwargs):
        # 模拟搜索结果
        mock_results = []
        for i in range(min(5, len(self.data.get(collection_name, [])))):
            entity = types.SimpleNamespace(
                get=lambda k, default=None: {'text': f"模拟文档内容 {i}", 'metadata': {'source': f'doc_{i}.md'}}.get(k, default)
            )
            hit = types.SimpleNamespace(
                entity=entity,
                score=0.9 - i * 0.1
            )
            mock_results.append(hit)
        return [mock_results]
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        return {
            'row_count': len(self.data.get(collection_name, [])),
            'collection_name': collection_name
        }

# 模拟连接
class MockConnections:
    @staticmethod
    def connect(**kwargs):
        print(f"模拟连接到Milvus: {kwargs}")
    
    @staticmethod
    def disconnect(alias: str):
        print(f"模拟断开连接: {alias}")

# 替换真实的Milvus模块
import sys
sys.modules['pymilvus'] = Mock()
sys.modules['pymilvus'].MilvusClient = MockMilvusClient
sys.modules['pymilvus'].connections = MockConnections()

# 测试Milvus服务
if __name__ == "__main__":
    from rag.config import RAGConfig
    from rag.milvus_service import MilvusService
    
    # 加载配置
    config = RAGConfig.from_file('test_config.yaml')
    
    # 创建Milvus服务
    service = MilvusService(config)
    print("Milvus服务初始化成功")
    
    # 测试集合设置
    service.setup_collection(force_recreate=True)
    
    # 测试数据插入
    chunks = ["文档1内容", "文档2内容", "文档3内容"]
    embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
    service.insert_data(chunks, embeddings)
    
    # 测试搜索
    query_embedding = [0.1] * 1536
    results = service.search(query_embedding, limit=3)
    print(f"搜索结果: {len(results)} 条")
    
    # 获取统计信息
    stats = service.get_statistics()
    print(f"统计信息: {stats}")
    
    # 清理资源
    service.cleanup()
    print("测试完成") 