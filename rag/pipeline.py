import logging
from openai import OpenAI
from .config import RAGConfig
from .document import DocumentProcessor
from .embedding import EmbeddingService
from .reranker import RerankerService
from .llm import LLMService
from .milvus_service import MilvusService

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self._setup_services()
    
    def _setup_services(self):
        self.openai_client = OpenAI(
            api_key=self.config.api.openai_api_key,
            base_url=self.config.api.openai_base_url
        )
        self.doc_processor = DocumentProcessor(self.config)
        self.embedding_service = EmbeddingService(self.config, self.openai_client)
        self.reranker_service = RerankerService(self.config, self.openai_client)
        self.llm_service = LLMService(self.config, self.openai_client)
        self.milvus_service = MilvusService(self.config)
        logger.info("所有服务初始化成功")
    
    def setup_collection(self, force_recreate: bool = False):
        self.milvus_service.setup_collection(force_recreate)
        if force_recreate:
            self._import_data()
    
    def _import_data(self):
        documents = self.doc_processor.load_documents()
        chunks = self.doc_processor.split_documents(documents)
        embeddings = self.embedding_service.encode_batch(chunks)
        self.milvus_service.insert_data(chunks, embeddings)
    
    def run(self, question: str) -> str:
        logger.info(f"开始处理问题: '{question}'")
        query_embedding = self.embedding_service.encode(question)
        candidates = self.milvus_service.search(query_embedding)
        logger.info(f"检索到 {len(candidates)} 个候选文档")
        reranked = self.reranker_service.rerank(question, candidates)
        reranked_docs = [doc for doc, _ in reranked[:self.config.search.rerank_top_k]]
        logger.info(f"重排后选取前 {self.config.search.rerank_top_k} 个文档")
        context = "\n".join(reranked_docs)
        answer = self.llm_service.generate_answer(question, context)
        logger.info("处理完成")
        return answer 