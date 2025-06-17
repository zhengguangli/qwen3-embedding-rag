#!/usr/bin/env python3
"""
Milvus服务模块

提供高性能的向量数据库服务，包括：
- 连接池管理
- 批量操作优化
- 错误处理和重试
- 性能监控
- 数据一致性保证
- 统一服务接口
"""

import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from pymilvus import MilvusClient, connections
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pymilvus.milvus_client.index import IndexParams

from src.rag.config import RAGConfig
from src.rag.exceptions import RAGException, handle_exception
from .base import BaseService
from .exceptions import (
    MilvusConnectionError,
    MilvusCollectionError, 
    MilvusIndexError,
    ValidationError,
)


@dataclass
class SearchResult:
    """搜索结果数据类"""
    content: str
    score: float
    metadata: Dict[str, Any]
    id: str
    rank: int
    processing_time: float


@dataclass
class InsertResult:
    """插入结果数据类"""
    success: bool
    inserted_count: int
    failed_count: int
    processing_time: float
    errors: List[str]


class MilvusService(BaseService):
    """增强的Milvus服务
    
    提供企业级向量数据库功能，支持：
    - 自动连接管理和重连
    - 批量操作和性能优化
    - 健康检查和故障检测
    - 详细的监控和统计
    """
    
    def __init__(self, config: RAGConfig):
        # 先设置必要的属性
        self.client: Optional[MilvusClient] = None
        self._connection_alias = "rag_default"
        
        # 调用父类初始化（会自动调用_initialize()）
        super().__init__(config, "milvus")
    
    def _initialize(self) -> None:
        """初始化Milvus服务"""
        try:
            # 验证必需配置
            self._validate_required_config()
            
            # 初始化客户端连接
            self._init_client()
            
            # 测试连接
            self._test_connection()
            
            self.logger.info("Milvus服务初始化成功")
            
        except Exception as e:
            raise MilvusConnectionError(
                f"Milvus服务初始化失败: {str(e)}",
                endpoint=self._get_endpoint()
            )
    
    def _validate_required_config(self) -> None:
        """验证必需的配置项"""
        required_configs = [
            ("database.collection_name", self.config.database.collection_name),
            ("database.embedding_dim", self.config.database.embedding_dim)
        ]
        
        # 检查连接配置（至少要有一个）
        endpoint_configs = [
            getattr(self.config.database, 'endpoint', None),
            getattr(self.config.database, 'milvus_uri', None),
            getattr(self.config.database, 'host', None)
        ]
        
        if not any(endpoint_configs):
            required_configs.append(("database.endpoint/milvus_uri/host", None))
        
        missing_configs = []
        for key, value in required_configs:
            if value is None:
                missing_configs.append(key)
        
        if missing_configs:
            raise ValidationError(
                f"缺少必需的配置项: {', '.join(missing_configs)}"
            )
    
    def _get_endpoint(self) -> str:
        """获取连接端点"""
        # 优先使用endpoint，其次milvus_uri，最后host+port
        if hasattr(self.config.database, 'endpoint') and self.config.database.endpoint:
            return self.config.database.endpoint
        elif hasattr(self.config.database, 'milvus_uri') and self.config.database.milvus_uri:
            return self.config.database.milvus_uri
        elif hasattr(self.config.database, 'host') and self.config.database.host:
            host = self.config.database.host
            port = getattr(self.config.database, 'port', 19530)
            return f"http://{host}:{port}"
        else:
            raise ValidationError("未找到有效的Milvus连接配置")
    
    def _parse_endpoint(self, endpoint: str) -> Tuple[str, int]:
        """解析端点为host和port"""
        try:
            # 移除协议前缀
            uri = endpoint
            for prefix in ['http://', 'https://', 'tcp://', 'unix://']:
                if uri.startswith(prefix):
                    uri = uri[len(prefix):]
                    break
            
            # 解析host和port
            if ':' in uri:
                host, port_str = uri.split(':', 1)
                port = int(port_str)
            else:
                host = uri
                port = 19530
            
            return host, port
            
        except Exception as e:
            raise ValidationError(f"无效的端点格式: {endpoint}")
    
    def _init_client(self) -> None:
        """初始化Milvus客户端"""
        try:
            endpoint = self._get_endpoint()
            host, port = self._parse_endpoint(endpoint)
            
            self.logger.info(f"连接Milvus服务器: {host}:{port}")
            
            # 建立连接
            connections.connect(
                alias=self._connection_alias,
                host=host,
                port=port
            )
            
            # 创建客户端
            self.client = MilvusClient(
                uri=endpoint,
                token=getattr(self.config.database, 'token', None)
            )
            
            self.logger.info(f"Milvus客户端连接成功: {endpoint}")
            
        except Exception as e:
            raise MilvusConnectionError(
                f"无法连接到Milvus服务器: {str(e)}",
                endpoint=self._get_endpoint()
            )
    
    def _test_connection(self) -> None:
        """测试连接"""
        try:
            if not self.client:
                raise MilvusConnectionError("客户端未初始化")
            
            # 尝试列出集合
            collections = self.client.list_collections()
            self.logger.debug(f"连接测试成功，发现 {len(collections)} 个集合")
            
        except Exception as e:
            raise MilvusConnectionError(f"连接测试失败: {str(e)}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.client:
                return False
            
            # 简单的连接测试
            self._test_connection()
            return True
            
        except Exception as e:
            self.logger.warning(f"健康检查失败: {str(e)}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def setup_collection(self, force_recreate: bool = False) -> bool:
        """设置集合
        
        Args:
            force_recreate: 是否强制重建集合
            
        Returns:
            设置是否成功
            
        Raises:
            MilvusCollectionError: 集合操作失败
        """
        with self._measure_time("setup_collection"):
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
                    self._create_collection(collection_name)
                    self._create_index(collection_name)
                else:
                    self.logger.info(f"集合已存在: {collection_name}")
                # 数据导入后，确保集合被加载到内存
                self.client.load_collection(collection_name=collection_name)
                return True
                
            except Exception as e:
                raise MilvusCollectionError(
                    f"设置集合失败: {str(e)}",
                    collection_name=self.config.database.collection_name
                )
    
    def _create_collection(self, collection_name: str) -> None:
        """创建集合（修正版，auto_id=False，允许手动传入id）"""
        try:
            from pymilvus import DataType
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.config.database.embedding_dim)
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                consistency_level=self.config.database.consistency_level,
                properties={
                    "collection.ttl.seconds": getattr(self.config.database, 'ttl_seconds', 0)
                }
            )
            self.logger.info(f"创建集合成功: {collection_name}")
        except Exception as e:
            raise MilvusCollectionError(
                f"创建集合失败: {str(e)}",
                collection_name=collection_name
            )
    
    def _create_index(self, collection_name: str) -> None:
        """创建索引（修正版，兼容新版Milvus API）"""
        try:
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type=self.config.database.metric_type,
                index_type=self.config.database.index_type,
                params={
                    "nlist": self.config.database.nlist,
                    "m": getattr(self.config.database, "m", 4),
                    "nbits": getattr(self.config.database, "nbits", 8)
                }
            )
            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            self.logger.info(f"创建索引成功: {collection_name}")
        except Exception as e:
            raise MilvusIndexError(
                f"创建索引失败: {str(e)}",
                index_type=self.config.database.index_type
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def insert_data(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> InsertResult:
        """插入数据
        
        Args:
            chunks: 文本块列表
            embeddings: 嵌入向量列表
            metadata: 元数据列表
            
        Returns:
            插入结果
            
        Raises:
            ValidationError: 输入验证失败
            MilvusCollectionError: 插入操作失败
        """
        # 输入验证
        if not chunks or not embeddings:
            raise ValidationError("chunks和embeddings不能为空")
        
        if len(chunks) != len(embeddings):
            raise ValidationError("chunks和embeddings长度不匹配")
        
        with self._measure_time("insert_data"):
            try:
                collection_name = self.config.database.collection_name
                
                # 准备数据
                data = self._prepare_insert_data(chunks, embeddings, metadata)
                
                # 批量插入
                inserted_count, errors = self._batch_insert(collection_name, data)
                
                result = InsertResult(
                    success=len(errors) == 0,
                    inserted_count=inserted_count,
                    failed_count=len(data) - inserted_count,
                    processing_time=0.0,  # 时间由_measure_time记录
                    errors=errors
                )
                
                self.logger.info(
                    f"插入完成: 成功 {result.inserted_count}, "
                    f"失败 {result.failed_count}, 总计 {len(data)}"
                )
                
                return result
                
            except Exception as e:
                if isinstance(e, (ValidationError, MilvusCollectionError)):
                    raise
                
                raise MilvusCollectionError(
                    f"插入数据失败: {str(e)}",
                    collection_name=self.config.database.collection_name
                )
    
    def _remove_reserved_keys(self, obj: Any, reserved_keys=None):
        if reserved_keys is None:
            reserved_keys = {"id", "embedding", "text"}
        if isinstance(obj, dict):
            return {k: self._remove_reserved_keys(v, reserved_keys) for k, v in obj.items() if k not in reserved_keys}
        elif isinstance(obj, list):
            return [self._remove_reserved_keys(i, reserved_keys) for i in obj]
        else:
            return obj

    def _prepare_insert_data(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadata: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """准备插入数据（递归移除metadata中的'id'等保留字段）"""
        data = []
        current_time = time.time()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # 生成唯一int型ID（hash+index，防止溢出）
            chunk_hash = int(hashlib.md5(chunk.encode('utf-8')).hexdigest(), 16) % (2**63)
            doc_id = chunk_hash + i
            doc_metadata = metadata[i].copy() if metadata and i < len(metadata) else {}
            doc_metadata = self._remove_reserved_keys(doc_metadata)
            print(f"[DEBUG] doc_metadata before insert: {doc_metadata}")
            doc_metadata.update({
                "chunk_id": doc_id,
                "chunk_index": i,
                "content_length": len(chunk),
                "word_count": len(chunk.split()),
                "insert_time": current_time
            })
            data.append({
                "id": doc_id,
                "text": chunk,
                "embedding": embedding,
                "metadata": doc_metadata
            })
        return data
    
    def _batch_insert(self, collection_name: str, data: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """批量插入数据"""
        batch_size = getattr(self.config.database, 'batch_size', 1000)
        inserted_count = 0
        errors = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                self.client.insert(collection_name, batch)
                inserted_count += len(batch)
                self.logger.debug(f"插入批次 {i//batch_size + 1}: {len(batch)} 条数据")
                
            except Exception as e:
                error_msg = f"批次 {i//batch_size + 1} 插入失败: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        return inserted_count, errors
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def search(
        self, 
        query_embedding: List[float], 
        limit: Optional[int] = None,
        filter_expr: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """搜索相似文档
        
        Args:
            query_embedding: 查询嵌入向量
            limit: 返回结果数量限制
            filter_expr: 过滤表达式
            include_metadata: 是否包含元数据
            
        Returns:
            搜索结果列表 (content, score, metadata)
            
        Raises:
            ValidationError: 输入验证失败
            MilvusCollectionError: 搜索操作失败
        """
        if not query_embedding:
            raise ValidationError("查询嵌入向量不能为空")
        
        with self._measure_time("search"):
            try:
                collection_name = self.config.database.collection_name
                
                # 构建搜索参数
                search_kwargs = {
                    "collection_name": collection_name,
                    "data": [query_embedding],
                    "anns_field": "embedding",
                    "search_params": {
                        "metric_type": self.config.database.metric_type,
                        "params": {
                            "nprobe": getattr(self.config.database, 'nprobe', 10)
                        }
                    },
                    "limit": limit or self.config.search.search_limit,
                    "output_fields": ["text", "metadata"] if include_metadata else ["text"]
                }
                
                if filter_expr:
                    search_kwargs["expr"] = filter_expr
                
                # 执行搜索
                results = self.client.search(**search_kwargs)
                
                # 处理结果
                search_results = []
                for hit in results[0]:
                    content = hit.entity.get("text", "")
                    score = hit.score
                    metadata = hit.entity.get("metadata", {}) if include_metadata else {}
                    
                    search_results.append((content, score, metadata))
                
                self.logger.debug(f"搜索完成，返回 {len(search_results)} 个结果")
                return search_results
                
            except Exception as e:
                if isinstance(e, ValidationError):
                    raise
                
                raise MilvusCollectionError(
                    f"搜索失败: {str(e)}",
                    collection_name=self.config.database.collection_name
                )
    
    def batch_search(
        self, 
        query_embeddings: List[List[float]], 
        limit: Optional[int] = None,
        max_workers: Optional[int] = None
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """批量搜索
        
        Args:
            query_embeddings: 查询嵌入向量列表
            limit: 每个查询的结果数量限制
            max_workers: 最大工作线程数
            
        Returns:
            批量搜索结果
        """
        if not query_embeddings:
            return []
        
        with self._measure_time("batch_search"):
            self.logger.info(f"开始批量搜索 {len(query_embeddings)} 个查询")
            
            # 确定工作线程数
            if max_workers is None:
                max_workers = min(len(query_embeddings), 4)
            
            try:
                # 并发执行搜索
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self.search, query_embedding, limit)
                        for query_embedding in query_embeddings
                    ]
                    
                    # 收集结果
                    results = []
                    for i, future in enumerate(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.logger.error(f"批量搜索中的查询 {i} 失败: {str(e)}")
                            results.append([])
                
                successful_count = len([r for r in results if r])
                self.logger.info(f"批量搜索完成，成功处理 {successful_count}/{len(query_embeddings)} 个查询")
                
                return results
                
            except Exception as e:
                raise MilvusCollectionError(f"批量搜索失败: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息
        
        Returns:
            集合信息字典
        """
        try:
            collection_name = self.config.database.collection_name
            
            if not self.client.has_collection(collection_name):
                return {"error": "集合不存在", "collection_name": collection_name}
            
            # 获取集合统计信息
            stats = self.client.get_collection_stats(collection_name)
            
            return {
                "collection_name": collection_name,
                "row_count": stats.get("row_count", 0),
                "index_status": "已创建" if self.client.has_index(collection_name) else "未创建",
                "dimension": self.config.database.embedding_dim,
                "metric_type": self.config.database.metric_type,
                "index_type": self.config.database.index_type,
                "consistency_level": self.config.database.consistency_level
            }
            
        except Exception as e:
            self.logger.error(f"获取集合信息失败: {str(e)}")
            return {"error": str(e)}
    
    def delete_data(self, filter_expr: str) -> int:
        """删除数据
        
        Args:
            filter_expr: 过滤表达式
            
        Returns:
            删除的数据条数
            
        Raises:
            MilvusCollectionError: 删除操作失败
        """
        with self._measure_time("delete_data"):
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
                return delete_count
                
            except Exception as e:
                raise MilvusCollectionError(
                    f"删除数据失败: {str(e)}",
                    collection_name=self.config.database.collection_name
                )
    
    def get_milvus_statistics(self) -> Dict[str, Any]:
        """获取Milvus服务统计信息
        
        Returns:
            详细的统计信息字典
        """
        base_metrics = self.get_metrics()
        
        return {
            **base_metrics,
            "connection_info": {
                "endpoint": self._get_endpoint(),
                "connection_alias": self._connection_alias,
                "is_connected": self.health_check()
            },
            "collection_info": self.get_collection_info(),
            "configuration": {
                "embedding_dim": self.config.database.embedding_dim,
                "metric_type": self.config.database.metric_type,
                "index_type": self.config.database.index_type,
                "consistency_level": self.config.database.consistency_level,
                "batch_size": getattr(self.config.database, 'batch_size', 1000)
            }
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.reset_metrics()
        self.logger.info("Milvus统计信息已重置")
    
    def check_connection(self) -> bool:
        """检测Milvus连接是否正常
        
        Returns:
            连接是否正常
        """
        try:
            if not self.client:
                return False
            
            # 尝试列出集合
            collections = self.client.list_collections()
            self.logger.info(f"Milvus连接正常，发现 {len(collections)} 个集合")
            return True
            
        except Exception as e:
            self.logger.error(f"Milvus连接检测失败: {str(e)}")
            return False
    
    def reconnect(self) -> bool:
        """重新连接
        
        Returns:
            重连是否成功
        """
        try:
            self.logger.info("尝试重新连接Milvus服务器...")
            
            # 关闭现有连接
            try:
                if hasattr(self, '_connection_alias'):
                    connections.disconnect(self._connection_alias)
            except Exception:
                pass
            
            # 重新初始化
            self._init_client()
            self._test_connection()
            
            self.logger.info("Milvus服务器重连成功")
            return True
            
        except Exception as e:
            self.logger.error(f"Milvus服务器重连失败: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            # 关闭连接
            if hasattr(self, '_connection_alias'):
                connections.disconnect(self._connection_alias)
            
            # 调用基类清理
            super().cleanup()
            
            self.logger.info("Milvus服务资源清理完成")
            
        except Exception as e:
            self.logger.error(f"Milvus服务资源清理失败: {str(e)}") 