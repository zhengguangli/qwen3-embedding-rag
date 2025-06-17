import logging
from pymilvus import MilvusClient
from .config import RAGConfig
from typing import List, Optional

logger = logging.getLogger(__name__)

class MilvusService:
    """Milvus 服务"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = MilvusClient(uri=self.config.database.milvus_uri)
        logger.info(f"Milvus客户端初始化成功: {self.config.database.milvus_uri}")
    def setup_collection(self, force_recreate: bool = False):
        """设置集合"""
        collection_name = self.config.database.collection_name
        if force_recreate:
            self.client.drop_collection(collection_name)
            logger.info(f"删除集合: {collection_name}")
        
        if not self.client.has_collection(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.config.database.embedding_dim,
                metric_type=self.config.database.metric_type,
                consistency_level=self.config.database.consistency_level
            )
            logger.info(f"创建集合: {collection_name}")
        else:
            logger.info(f"集合已存在: {collection_name}")
        
        # 创建索引
        collection_name = self.config.database.collection_name
        if not self.client.has_index(collection_name):
            self.client.create_index(
                collection_name=collection_name,
                index_type=self.config.database.index_type,
                metric_type=self.config.database.metric_type,
                extra_params={"nlist": self.config.database.nlist}
            )
            logger.info(f"创建索引: {collection_name}")
    def insert_data(self, chunks: List[str], embeddings: List[List[float]]):
        """插入数据"""
        collection_name = self.config.database.collection_name
        data = [{"id": i, "text": chunk, "embedding": embedding} for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
        self.client.insert(collection_name, data)
        logger.info(f"插入 {len(data)} 条数据到集合: {collection_name}")
    def search(self, query_embedding: List[float], limit: Optional[int] = None) -> List[str]:
        """搜索相似文档"""
        collection_name = self.config.database.collection_name
        results = self.client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=limit or self.config.search.search_limit,
            output_fields=["text"]
        )
        return [hit.entity.get("text") for hit in results[0]] 