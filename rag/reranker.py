#!/usr/bin/env python3
"""
重排序服务模块

提供多种重排序算法和策略，包括：
- 多种重排序算法
- 模型选择
- 性能优化
- 错误处理和重试
"""

import time
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import RAGConfig

logger = logging.getLogger(__name__)

@dataclass
class RerankResult:
    """重排序结果数据类"""
    content: str
    score: float
    algorithm: str
    model: str
    processing_time: float
    metadata: Dict[str, Any]

class RerankerService:
    """增强的重排序服务"""
    
    def __init__(self, config: RAGConfig, client: OpenAI):
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self._stats = {
            "total_reranks": 0,
            "successful_reranks": 0,
            "failed_reranks": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0
        }
        
        # 线程池用于并发处理
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # 重排序算法映射
        self.rerank_algorithms = {
            "llm": self._llm_rerank,
            "keyword": self._keyword_rerank,
            "semantic": self._semantic_rerank,
            "hybrid": self._hybrid_rerank
        }
        
        # 缓存相关性分数
        self._relevance_cache = {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def rerank(
        self, 
        question: str, 
        candidates: List[str], 
        algorithm: str = "llm",
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """重排序候选文档"""
        if not candidates:
            return []
        
        start_time = time.time()
        
        try:
            self._stats["total_reranks"] += 1
            
            # 确定top_k
            if top_k is None:
                top_k = self.config.search.rerank_top_k
            
            # 获取重排序算法
            rerank_func = self.rerank_algorithms.get(algorithm, self._llm_rerank)
            
            self.logger.info(f"开始重排序 {len(candidates)} 个候选文档，算法: {algorithm}")
            
            # 执行重排序
            results = rerank_func(question, candidates)
            
            # 按分数排序并限制数量
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._stats["successful_reranks"] += 1
            self._stats["total_processing_time"] += processing_time
            self._stats["average_batch_size"] = (
                (self._stats["average_batch_size"] * (self._stats["successful_reranks"] - 1) + len(candidates)) /
                self._stats["successful_reranks"]
            )
            
            self.logger.info(f"重排序完成，耗时: {processing_time:.2f}秒")
            return results
            
        except Exception as e:
            self._stats["failed_reranks"] += 1
            self.logger.error(f"重排序失败: {str(e)}")
            raise
    
    def _llm_rerank(self, question: str, candidates: List[str]) -> List[RerankResult]:
        """LLM重排序算法"""
        results = []
        
        # 并发处理
        futures = []
        for candidate in candidates:
            future = self._executor.submit(
                self._calculate_llm_relevance, 
                question, 
                candidate
            )
            futures.append((future, candidate))
        
        # 收集结果
        for future, candidate in futures:
            try:
                score = future.result()
                result = RerankResult(
                    content=candidate,
                    score=score,
                    algorithm="llm",
                    model=self.config.models.llm.name,
                    processing_time=0.0,  # 单个处理时间未记录
                    metadata={"method": "llm_scoring"}
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"LLM重排序失败: {str(e)}")
                # 使用默认分数
                result = RerankResult(
                    content=candidate,
                    score=0.5,
                    algorithm="llm",
                    model=self.config.models.llm.name,
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def _keyword_rerank(self, question: str, candidates: List[str]) -> List[RerankResult]:
        """关键词重排序算法"""
        results = []
        
        # 提取问题关键词
        question_keywords = self._extract_keywords(question)
        
        for candidate in candidates:
            try:
                # 提取候选文档关键词
                candidate_keywords = self._extract_keywords(candidate)
                
                # 计算关键词重叠度
                overlap = len(question_keywords.intersection(candidate_keywords))
                total_keywords = len(question_keywords.union(candidate_keywords))
                
                if total_keywords > 0:
                    score = overlap / total_keywords
                else:
                    score = 0.0
                
                result = RerankResult(
                    content=candidate,
                    score=score,
                    algorithm="keyword",
                    model="keyword_matching",
                    processing_time=0.0,
                    metadata={
                        "method": "keyword_overlap",
                        "question_keywords": list(question_keywords),
                        "candidate_keywords": list(candidate_keywords),
                        "overlap": overlap,
                        "total_keywords": total_keywords
                    }
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"关键词重排序失败: {str(e)}")
                result = RerankResult(
                    content=candidate,
                    score=0.5,
                    algorithm="keyword",
                    model="keyword_matching",
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def _semantic_rerank(self, question: str, candidates: List[str]) -> List[RerankResult]:
        """语义重排序算法"""
        results = []
        
        # 使用嵌入模型计算语义相似度
        try:
            # 获取问题嵌入
            question_embedding = self._get_embedding(question)
            
            # 并发计算候选文档嵌入
            futures = []
            for candidate in candidates:
                future = self._executor.submit(self._get_embedding, candidate)
                futures.append((future, candidate))
            
            # 计算相似度
            for future, candidate in futures:
                try:
                    candidate_embedding = future.result()
                    similarity = self._cosine_similarity(question_embedding, candidate_embedding)
                    
                    result = RerankResult(
                        content=candidate,
                        score=similarity,
                        algorithm="semantic",
                        model=self.config.models.embedding.name,
                        processing_time=0.0,
                        metadata={"method": "semantic_similarity"}
                    )
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"语义重排序失败: {str(e)}")
                    result = RerankResult(
                        content=candidate,
                        score=0.5,
                        algorithm="semantic",
                        model=self.config.models.embedding.name,
                        processing_time=0.0,
                        metadata={"error": str(e)}
                    )
                    results.append(result)
                    
        except Exception as e:
            self.logger.error(f"语义重排序初始化失败: {str(e)}")
            # 返回默认结果
            for candidate in candidates:
                result = RerankResult(
                    content=candidate,
                    score=0.5,
                    algorithm="semantic",
                    model=self.config.models.embedding.name,
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def _hybrid_rerank(self, question: str, candidates: List[str]) -> List[RerankResult]:
        """混合重排序算法"""
        # 获取不同算法的结果
        llm_results = self._llm_rerank(question, candidates)
        keyword_results = self._keyword_rerank(question, candidates)
        semantic_results = self._semantic_rerank(question, candidates)
        
        # 创建结果映射
        llm_scores = {r.content: r.score for r in llm_results}
        keyword_scores = {r.content: r.score for r in keyword_results}
        semantic_scores = {r.content: r.score for r in semantic_results}
        
        # 计算混合分数
        results = []
        for candidate in candidates:
            llm_score = llm_scores.get(candidate, 0.5)
            keyword_score = keyword_scores.get(candidate, 0.5)
            semantic_score = semantic_scores.get(candidate, 0.5)
            
            # 加权平均（可配置权重）
            hybrid_score = (
                llm_score * 0.5 +      # LLM权重50%
                keyword_score * 0.3 +  # 关键词权重30%
                semantic_score * 0.2   # 语义权重20%
            )
            
            result = RerankResult(
                content=candidate,
                score=hybrid_score,
                algorithm="hybrid",
                model="multiple",
                processing_time=0.0,
                metadata={
                    "method": "hybrid_scoring",
                    "llm_score": llm_score,
                    "keyword_score": keyword_score,
                    "semantic_score": semantic_score,
                    "weights": {"llm": 0.5, "keyword": 0.3, "semantic": 0.2}
                }
            )
            results.append(result)
        
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _calculate_llm_relevance(self, question: str, candidate: str) -> float:
        """计算LLM相关性分数"""
        # 检查缓存
        cache_key = f"{hash(question)}_{hash(candidate)}"
        if cache_key in self._relevance_cache:
            return self._relevance_cache[cache_key]
        
        try:
            # 构建提示词
            prompt = f"""请评估以下文档与问题的相关性，返回0-1之间的分数。

问题: {question}

文档: {candidate[:1000]}  # 限制长度

请只返回数字分数，不要其他内容。相关性分数(0-1):"""

            response = self.client.chat.completions.create(
                model=self.config.models.llm.name,
                messages=[
                    {"role": "system", "content": "你是一个文档相关性评估器。请评估给定文档与问题的相关性，返回0-1之间的分数。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # 提取数字
            score_match = re.search(r'0\.\d+|1\.0|1|0', score_text)
            if score_match:
                score = float(score_match.group())
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5
            
            # 缓存结果
            self._relevance_cache[cache_key] = score
            return score
            
        except Exception as e:
            self.logger.error(f"LLM相关性计算失败: {str(e)}")
            return 0.5
    
    def _extract_keywords(self, text: str) -> set:
        """提取关键词"""
        # 简单的关键词提取（可以扩展为更复杂的算法）
        # 去除停用词和标点符号
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 分词（简单按空格和标点分割）
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词和短词
        keywords = {word for word in words if word not in stop_words and len(word) > 1}
        
        return keywords
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model=self.config.models.embedding.name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"获取嵌入失败: {str(e)}")
            # 返回零向量
            return [0.0] * self.config.models.embedding.dim
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add_rerank_algorithm(self, name: str, algorithm_func):
        """添加自定义重排序算法"""
        self.rerank_algorithms[name] = algorithm_func
        self.logger.info(f"添加重排序算法: {name}")
    
    def list_algorithms(self) -> List[str]:
        """列出所有可用的重排序算法"""
        return list(self.rerank_algorithms.keys())
    
    def clear_cache(self):
        """清空相关性缓存"""
        self._relevance_cache.clear()
        self.logger.info("重排序缓存已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_reranks = self._stats["total_reranks"]
        success_rate = 0.0
        avg_time = 0.0
        
        if total_reranks > 0:
            success_rate = self._stats["successful_reranks"] / total_reranks
            avg_time = self._stats["total_processing_time"] / self._stats["successful_reranks"]
        
        return {
            "total_reranks": total_reranks,
            "successful_reranks": self._stats["successful_reranks"],
            "failed_reranks": self._stats["failed_reranks"],
            "success_rate": success_rate,
            "average_processing_time": avg_time,
            "average_batch_size": self._stats["average_batch_size"],
            "cache_size": len(self._relevance_cache),
            "available_algorithms": self.list_algorithms()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._stats = {
            "total_reranks": 0,
            "successful_reranks": 0,
            "failed_reranks": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0
        }
        self.logger.info("重排序服务统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            self.logger.info("重排序服务资源清理完成")
        except Exception as e:
            self.logger.error(f"重排序服务资源清理失败: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 