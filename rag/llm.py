import logging
from openai import OpenAI
from .config import RAGConfig

logger = logging.getLogger(__name__)

class LLMService:
    """LLM 服务"""
    def __init__(self, config: RAGConfig, client: OpenAI):
        self.config = config
        self.client = client
    def generate_answer(self, question: str, context: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": f"问题: {question}\n上下文: {context}"}
                ],
                temperature=0.2,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"答案生成失败: {str(e)}")
            return "抱歉，生成答案失败。" 