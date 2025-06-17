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
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        if not documents:
            return []
        doc_scores = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_doc = {
                executor.submit(self._compute_score, query, doc): doc
                for doc in documents
            }
            for future in as_completed(future_to_doc):
                try:
                    score = future.result()
                    doc_scores.append((future_to_doc[future], score))
                except Exception as e:
                    logger.error(f"重排序失败: {str(e)}")
                    doc_scores.append((future_to_doc[future], 0.0))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores
    def _compute_score(self, query: str, doc: str) -> float:
        # 这里假设用LLM API做重排序，实际可根据模型API调整
        try:
            response = self.client.chat.completions.create(
                model=self.config.reranker_model,
                messages=[
                    {"role": "system", "content": "请判断文档与查询的相关性，返回相关性分数（0-1）"},
                    {"role": "user", "content": f"查询: {query}\n文档: {doc}\n请给出相关性分数（0-1）"}
                ],
                temperature=0.0,
                max_tokens=8
            )
            score_str = response.choices[0].message.content.strip()
            try:
                score = float(score_str)
            except Exception:
                score = 0.0
            return score
        except Exception as e:
            logger.error(f"重排序分数计算失败: {str(e)}")
            return 0.0 