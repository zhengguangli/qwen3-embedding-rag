#!/usr/bin/env python3
"""
RAG系统管道模块

提供完整的RAG处理流程，包括：
- 文档检索
- 重排序
- 答案生成
- 异步处理支持
- 错误处理和重试机制
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import RAGConfig
from .document import DocumentProcessor
from .embedding import EmbeddingService
from .reranker import RerankerService
from .llm import LLMService
from .milvus_service import MilvusService
from .utils import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果数据类"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str

@dataclass
class RAGResponse:
    """RAG响应数据类"""
    answer: str
    sources: List[SearchResult]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class RAGPipeline:
    """RAG处理管道"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.logging.log_level)
        self._setup_services()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _setup_services(self):
        """初始化所有服务"""
        try:
            self.openai_client = OpenAI(
                api_key=self.config.api.openai_api_key,
                base_url=self.config.api.openai_base_url,
                timeout=self.config.api.timeout
            )
            self.doc_processor = DocumentProcessor(self.config)
            self.embedding_service = EmbeddingService(self.config, self.openai_client)
            self.reranker_service = RerankerService(self.config, self.openai_client)
            self.llm_service = LLMService(self.config, self.openai_client)
            self.milvus_service = MilvusService(self.config)
            self.logger.info("所有服务初始化成功")
        except Exception as e:
            self.logger.error(f"服务初始化失败: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def setup_collection(self, force_recreate: bool = False) -> bool:
        """设置Milvus集合"""
        try:
            self.logger.info(f"设置Milvus集合: {self.config.database.collection_name}")
            self.milvus_service.setup_collection(force_recreate)
            
            if force_recreate:
                self.logger.info("强制重建集合，开始导入数据...")
                self._import_data()
            
            self.logger.info("Milvus集合设置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"设置Milvus集合失败: {str(e)}")
            raise
    
    def _import_data(self):
        """导入数据到Milvus"""
        try:
            self.logger.info("开始导入数据...")
            
            # 加载文档
            documents = self.doc_processor.load_documents()
            self.logger.info(f"加载了 {len(documents)} 个文档")
            
            # 分块处理
            chunks = self.doc_processor.split_documents(documents)
            self.logger.info(f"分块后共 {len(chunks)} 段")
            
            # 提取文本内容用于嵌入
            chunk_texts = [chunk.content for chunk in chunks]
            
            # 批量编码
            embeddings = self.embedding_service.encode_batch(chunk_texts)
            self.logger.info(f"生成了 {len(embeddings)} 个嵌入向量")
            
            # 插入数据
            self.milvus_service.insert_data(chunk_texts, embeddings)
            self.logger.info("数据导入完成")
            
        except Exception as e:
            self.logger.error(f"数据导入失败: {str(e)}")
            raise
    
    def run(self, question: str) -> str:
        """同步运行RAG流程"""
        try:
            response = asyncio.run(self.run_async(question))
            return response.answer
        except Exception as e:
            self.logger.error(f"RAG流程执行失败: {str(e)}")
            raise
    
    async def run_async(self, question: str) -> RAGResponse:
        """异步运行RAG流程"""
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"开始处理问题: '{question}'")
            
            # 1. 生成查询嵌入
            query_embedding = await self._encode_query_async(question)
            
            # 2. 检索相关文档
            candidates = await self._search_documents_async(query_embedding)
            self.logger.info(f"检索到 {len(candidates)} 个候选文档")
            
            # 3. 重排序（如果启用）
            if self.config.search.use_reranker and candidates:
                reranked = await self._rerank_documents_async(question, candidates)
                top_docs = reranked[:self.config.search.rerank_top_k]
                self.logger.info(f"重排序后选取前 {len(top_docs)} 个文档")
            else:
                top_docs = candidates[:self.config.search.search_limit]
            
            # 4. 生成答案
            context = self._build_context(top_docs)
            answer = await self._generate_answer_async(question, context)
            
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成，耗时: {processing_time:.2f}秒")
            
            return RAGResponse(
                answer=answer,
                sources=top_docs,
                confidence=self._calculate_confidence(top_docs),
                processing_time=processing_time,
                metadata={
                    "question": question,
                    "candidates_count": len(candidates),
                    "top_docs_count": len(top_docs),
                    "use_reranker": self.config.search.use_reranker
                }
            )
            
        except Exception as e:
            self.logger.error(f"RAG流程执行失败: {str(e)}")
            raise
    
    async def _encode_query_async(self, question: str) -> List[float]:
        """异步编码查询"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.embedding_service.encode, 
            question
        )
    
    async def _search_documents_async(self, query_embedding: List[float]) -> List[SearchResult]:
        """异步搜索文档"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.milvus_service.search,
            query_embedding
        )
        
        # 转换为SearchResult对象
        search_results = []
        for content, score, metadata in results:
            search_results.append(SearchResult(
                content=content,
                score=score,
                metadata=metadata or {},
                source=metadata.get('source', 'unknown') if metadata else 'unknown'
            ))
        
        return search_results
    
    async def _rerank_documents_async(self, question: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """异步重排序文档"""
        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            self._executor,
            self.reranker_service.rerank,
            question,
            [doc.content for doc in candidates]
        )
        
        # 重新组织结果
        reranked_results = []
        for rerank_result in reranked:
            # 找到对应的原始文档
            for candidate in candidates:
                if candidate.content == rerank_result.content:
                    reranked_results.append(SearchResult(
                        content=rerank_result.content,
                        score=rerank_result.score,
                        metadata=candidate.metadata,
                        source=candidate.source
                    ))
                    break
        
        return reranked_results
    
    async def _generate_answer_async(self, question: str, context: str) -> str:
        """异步生成答案"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor,
            self.llm_service.generate_answer,
            question,
            context
        )
        return response.content
    
    def _build_context(self, documents: List[SearchResult]) -> str:
        """构建上下文"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"文档{i} (相似度: {doc.score:.3f}):\n{doc.content}")
        
        return "\n\n".join(context_parts)
    
    def _calculate_confidence(self, documents: List[SearchResult]) -> float:
        """计算置信度"""
        if not documents:
            return 0.0
        
        # 基于最高相似度分数计算置信度
        max_score = max(doc.score for doc in documents)
        avg_score = sum(doc.score for doc in documents) / len(documents)
        
        # 综合考虑最高分和平均分
        confidence = (max_score * 0.7 + avg_score * 0.3)
        return min(confidence, 1.0)
    
    def batch_process(self, questions: List[str]) -> List[RAGResponse]:
        """批量处理多个问题"""
        try:
            self.logger.info(f"开始批量处理 {len(questions)} 个问题")
            
            # 使用异步并发处理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                tasks = [self.run_async(q) for q in questions]
                responses = loop.run_until_complete(asyncio.gather(*tasks))
                return responses
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"批量处理失败: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            stats = {
                "collection_info": self.milvus_service.get_collection_info(),
                "embedding_cache_size": self.embedding_service.get_cache_info(),
                "config_summary": {
                    "embedding_model": self.config.models.embedding.name,
                    "llm_model": self.config.models.llm.name,
                    "collection_name": self.config.database.collection_name,
                    "search_limit": self.config.search.search_limit,
                    "use_reranker": self.config.search.use_reranker
                }
            }
            return stats
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {str(e)}")
            return {}
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            self.logger.info("资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理失败: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 