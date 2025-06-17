from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from openai import OpenAI
from .config import RAGConfig

logger = logging.getLogger(__name__)

class RerankerService:
    """重排序服务"""
    def __init__(self, config: RAGConfig, client: OpenAI):
        self.config = config
        self.client = client
    def rerank(self, question: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """重排序候选文档"""
        if not candidates:
            return []
        
        with ThreadPoolExecutor(max_workers=self.config.performance.max_workers) as executor:
            # 为每个候选文档生成重排序分数
            future_to_candidate = {
                executor.submit(self._calculate_relevance, question, candidate): candidate
                for candidate in candidates
            }
            
            results = []
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    score = future.result()
                    results.append((candidate, score))
                except Exception as e:
                    logger.error(f"重排序失败: {str(e)}")
                    results.append((candidate, 0.0))
        
        # 按分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_relevance(self, question: str, candidate: str) -> float:
        """计算相关性分数"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.models.reranker_model,
                messages=[
                    {"role": "system", "content": "你是一个文档相关性评估器。请评估给定文档与问题的相关性，返回0-1之间的分数。"},
                    {"role": "user", "content": f"问题: {question}\n文档: {candidate}\n相关性分数(0-1):"}
                ],
                temperature=0.1,
                max_tokens=10
            )
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.5
        except Exception as e:
            logger.error(f"相关性计算失败: {str(e)}")
            return 0.5 