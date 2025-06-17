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
        """生成答案"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.models.llm_model,
                messages=[
                    {"role": "system", "content": self.config.prompts.system_prompt},
                    {"role": "user", "content": self.config.prompts.query_prompt.format(context=context, question=question)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM生成答案失败: {str(e)}")
            return f"抱歉，生成答案时出现错误: {str(e)}" 