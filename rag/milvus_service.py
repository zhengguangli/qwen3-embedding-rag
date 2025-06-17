import logging
from pymilvus import MilvusClient
from .config import RAGConfig

logger = logging.getLogger(__name__)

class MilvusService:
    """Milvus 服务"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self._connect()
    def _connect(self):
        try:
            self.client = MilvusClient(uri=self.config.milvus_uri)
            logger.info("Milvus 连接成功")
        except Exception as e:
            logger.error(f"Milvus 连接失败: {str(e)}")
            raise
    def setup_collection(self, force_recreate: bool = False) -> None:
        collection_name = self.config.collection_name
        if force_recreate and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"已删除集合: {collection_name}")
        if not self.client.has_collection(collection_name):
            # 使用简化的集合创建方式
            self.client.create_collection(
                collection_name,
                dimension=self.config.embedding_dim,
                metric_type=self.config.metric_type,
                consistency_level=self.config.consistency_level
            )
            logger.info(f"已创建集合: {collection_name}")
        else:
            logger.info("集合已存在")
    def insert_data(self, documents, embeddings):
        collection_name = self.config.collection_name
        try:
            # 准备数据，使用正确的格式
            data = []
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                data.append({
                    "id": i,
                    "vector": emb,
                    "text": doc
                })
            
            # 使用正确的插入方式
            self.client.insert(collection_name, data)
            logger.info(f"已插入 {len(documents)} 条数据")
        except Exception as e:
            logger.error(f"数据插入失败: {str(e)}")
            raise
    def search(self, query_embedding, limit=None):
        collection_name = self.config.collection_name
        try:
            results = self.client.search(
                collection_name, 
                [query_embedding], 
                limit=limit or self.config.search_limit,
                output_fields=["text"]
            )
            # 修正结果处理方式
            if results and len(results) > 0:
                return [hit.entity.get("text", "") for hit in results[0]]
            return []
        except Exception as e:
            logger.error(f"Milvus 检索失败: {str(e)}")
            return [] 