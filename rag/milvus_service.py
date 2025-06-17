#!/usr/bin/env python3
"""
Milvus服务模块

提供高性能的向量数据库服务，包括：
- 连接池管理
- 批量操作优化
- 错误处理和重试
- 性能监控
- 数据一致性保证
"""

import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from pymilvus import MilvusClient, connections
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import RAGConfig

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果数据类"""
    content: str
    score: float
    metadata: Dict[str, Any]
    id: str

class MilvusService:
    """增强的Milvus服务"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self._stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_inserts": 0,
            "successful_inserts": 0,
            "failed_inserts": 0,
            "total_processing_time": 0.0,
            "average_search_time": 0.0
        }
        
        # 连接池
        self._connection_pool = {}
        self._max_connections = 5
        
        # 初始化客户端
        self._init_client()
        
        # 线程池用于并发操作
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _init_client(self):
        """初始化Milvus客户端"""
        try:
            # 建立连接
            connections.connect(
                alias="default",
                host=self.config.database.host,
                port=self.config.database.port
            )
            
            self.client = MilvusClient(
                uri=f"{self.config.database.host}:{self.config.database.port}",
                token=self.config.database.token if hasattr(self.config.database, 'token') else None
            )
            
            self.logger.info(f"Milvus客户端初始化成功: {self.config.database.host}:{self.config.database.port}")
            
        except Exception as e:
            self.logger.error(f"Milvus客户端初始化失败: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def setup_collection(self, force_recreate: bool = False) -> bool:
        """设置集合"""
        try:
            collection_name = self.config.database.collection_name
            
            # 强制重建
            if force_recreate:
                try:
                    self.client.drop_collection(collection_name)
                    self.logger.info(f"删除集合: {collection_name}")
                except Exception as e:
                    self.logger.warning(f"删除集合失败（可能不存在）: {str(e)}")
            
            # 检查集合是否存在
            if not self.client.has_collection(collection_name):
                # 创建集合
                self.client.create_collection(
                    collection_name=collection_name,
                    dimension=self.config.database.embedding_dim,
                    metric_type=self.config.database.metric_type,
                    consistency_level=self.config.database.consistency_level,
                    properties={
                        "collection.ttl.seconds": self.config.database.ttl_seconds if hasattr(self.config.database, 'ttl_seconds') else 0
                    }
                )
                self.logger.info(f"创建集合: {collection_name}")
                
                # 创建索引
                self._create_index(collection_name)
            else:
                self.logger.info(f"集合已存在: {collection_name}")
                
                # 检查索引是否存在
                if not self.client.has_index(collection_name):
                    self._create_index(collection_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置集合失败: {str(e)}")
            raise
    
    def _create_index(self, collection_name: str):
        """创建索引"""
        try:
            index_params = {
                "index_type": self.config.database.index_type,
                "metric_type": self.config.database.metric_type,
                "params": {
                    "nlist": self.config.database.nlist,
                    "m": self.config.database.m if hasattr(self.config.database, 'm') else 4,
                    "nbits": self.config.database.nbits if hasattr(self.config.database, 'nbits') else 8
                }
            }
            
            self.client.create_index(
                collection_name=collection_name,
                field_name="embedding",
                index_params=index_params
            )
            self.logger.info(f"创建索引: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"创建索引失败: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def insert_data(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """插入数据"""
        if not chunks or not embeddings:
            self.logger.warning("没有数据需要插入")
            return False
        
        if len(chunks) != len(embeddings):
            raise ValueError("chunks和embeddings长度不匹配")
        
        start_time = time.time()
        
        try:
            self._stats["total_inserts"] += 1
            collection_name = self.config.database.collection_name
            
            # 准备数据
            data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 生成唯一ID
                chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
                doc_id = f"{chunk_hash}_{i}"
                
                # 准备元数据
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                doc_metadata.update({
                    "chunk_id": doc_id,
                    "chunk_index": i,
                    "content_length": len(chunk),
                    "word_count": len(chunk.split()),
                    "insert_time": time.time()
                })
                
                data.append({
                    "id": doc_id,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": doc_metadata
                })
            
            # 批量插入
            batch_size = self.config.database.batch_size if hasattr(self.config.database, 'batch_size') else 1000
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                self.client.insert(collection_name, batch)
                self.logger.debug(f"插入批次 {i//batch_size + 1}: {len(batch)} 条数据")
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._stats["successful_inserts"] += 1
            self._stats["total_processing_time"] += processing_time
            
            self.logger.info(f"插入 {len(data)} 条数据到集合: {collection_name}，耗时: {processing_time:.2f}秒")
            return True
            
        except Exception as e:
            self._stats["failed_inserts"] += 1
            self.logger.error(f"插入数据失败: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def search(
        self, 
        query_embedding: List[float], 
        limit: Optional[int] = None,
        filter_expr: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """搜索相似文档"""
        start_time = time.time()
        
        try:
            self._stats["total_searches"] += 1
            collection_name = self.config.database.collection_name
            
            # 搜索参数
            search_params = {
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": {
                    "metric_type": self.config.database.metric_type,
                    "params": {
                        "nprobe": self.config.database.nprobe if hasattr(self.config.database, 'nprobe') else 10
                    }
                },
                "limit": limit or self.config.search.search_limit,
                "output_fields": ["text", "metadata"]
            }
            
            # 添加过滤条件
            if filter_expr:
                search_params["expr"] = filter_expr
            
            # 执行搜索
            results = self.client.search(
                collection_name=collection_name,
                **search_params
            )
            
            # 处理结果
            search_results = []
            for hit in results[0]:
                content = hit.entity.get("text", "")
                score = hit.score
                metadata = hit.entity.get("metadata", {})
                
                search_results.append((content, score, metadata))
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._stats["successful_searches"] += 1
            self._stats["total_processing_time"] += processing_time
            self._stats["average_search_time"] = (
                (self._stats["average_search_time"] * (self._stats["successful_searches"] - 1) + processing_time) /
                self._stats["successful_searches"]
            )
            
            self.logger.debug(f"搜索完成，返回 {len(search_results)} 个结果，耗时: {processing_time:.3f}秒")
            return search_results
            
        except Exception as e:
            self._stats["failed_searches"] += 1
            self.logger.error(f"搜索失败: {str(e)}")
            raise
    
    def batch_search(self, query_embeddings: List[List[float]], limit: Optional[int] = None) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """批量搜索"""
        try:
            self.logger.info(f"开始批量搜索 {len(query_embeddings)} 个查询")
            
            # 并发执行搜索
            futures = []
            for query_embedding in query_embeddings:
                future = self._executor.submit(self.search, query_embedding, limit)
                futures.append(future)
            
            # 收集结果
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"批量搜索中的单个查询失败: {str(e)}")
                    results.append([])
            
            self.logger.info(f"批量搜索完成，成功处理 {len([r for r in results if r])} 个查询")
            return results
            
        except Exception as e:
            self.logger.error(f"批量搜索失败: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            collection_name = self.config.database.collection_name
            
            if not self.client.has_collection(collection_name):
                return {"error": "集合不存在"}
            
            # 获取集合统计信息
            stats = self.client.get_collection_stats(collection_name)
            
            return {
                "collection_name": collection_name,
                "row_count": stats.get("row_count", 0),
                "index_status": "已创建" if self.client.has_index(collection_name) else "未创建",
                "dimension": self.config.database.embedding_dim,
                "metric_type": self.config.database.metric_type,
                "index_type": self.config.database.index_type
            }
            
        except Exception as e:
            self.logger.error(f"获取集合信息失败: {str(e)}")
            return {"error": str(e)}
    
    def delete_data(self, filter_expr: str) -> bool:
        """删除数据"""
        try:
            collection_name = self.config.database.collection_name
            
            # 先查询要删除的数据数量
            results = self.client.query(
                collection_name=collection_name,
                filter_=filter_expr,
                output_fields=["id"]
            )
            
            delete_count = len(results)
            
            # 执行删除
            self.client.delete(collection_name, filter_expr)
            
            self.logger.info(f"删除 {delete_count} 条数据，过滤条件: {filter_expr}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除数据失败: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_searches = self._stats["total_searches"]
        total_inserts = self._stats["total_inserts"]
        
        search_success_rate = 0.0
        insert_success_rate = 0.0
        
        if total_searches > 0:
            search_success_rate = self._stats["successful_searches"] / total_searches
        
        if total_inserts > 0:
            insert_success_rate = self._stats["successful_inserts"] / total_inserts
        
        return {
            "searches": {
                "total": total_searches,
                "successful": self._stats["successful_searches"],
                "failed": self._stats["failed_searches"],
                "success_rate": search_success_rate,
                "average_time": self._stats["average_search_time"]
            },
            "inserts": {
                "total": total_inserts,
                "successful": self._stats["successful_inserts"],
                "failed": self._stats["failed_inserts"],
                "success_rate": insert_success_rate
            },
            "collection_info": self.get_collection_info()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_inserts": 0,
            "successful_inserts": 0,
            "failed_inserts": 0,
            "total_processing_time": 0.0,
            "average_search_time": 0.0
        }
        self.logger.info("Milvus服务统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            
            # 关闭连接
            connections.disconnect("default")
            
            self.logger.info("Milvus服务资源清理完成")
        except Exception as e:
            self.logger.error(f"Milvus服务资源清理失败: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 