#!/usr/bin/env python3
"""
LLM服务模块

提供强大的大语言模型服务，包括：
- 流式输出支持
- 智能提示词工程
- 响应质量评估
- 错误处理和重试
- 性能优化
"""

import time
import json
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.rag.config import RAGConfig
from src.rag.exceptions import RAGException, handle_exception

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """LLM响应数据类"""
    content: str
    model: str
    usage: Dict[str, Any]
    processing_time: float
    quality_score: float
    metadata: Dict[str, Any]

@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    template: str
    variables: List[str]
    description: str

class LLMService:
    """增强的LLM服务"""
    
    def __init__(self, config: RAGConfig, client: OpenAI):
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_response_length": 0.0
        }
        
        # 线程池用于并发处理
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # 初始化提示词模板
        self._init_prompt_templates()
    
    def _init_prompt_templates(self):
        """初始化提示词模板"""
        self.prompt_templates = {
            "rag": PromptTemplate(
                name="rag",
                template="""你是一个专业的AI助手，基于以下上下文信息回答问题。

上下文信息：
{context}

用户问题：{question}

请基于上下文信息提供准确、详细的答案。如果上下文中没有相关信息，请明确说明。
答案：""",
                variables=["context", "question"],
                description="RAG问答模板"
            ),
            "summarize": PromptTemplate(
                name="summarize",
                template="""请对以下文本进行总结：

{text}

总结要求：
- 保持关键信息
- 简洁明了
- 不超过200字

总结：""",
                variables=["text"],
                description="文本总结模板"
            ),
            "analyze": PromptTemplate(
                name="analyze",
                template="""请分析以下内容：

{content}

分析要求：
- 识别主要观点
- 评估可信度
- 提供建议

分析结果：""",
                variables=["content"],
                description="内容分析模板"
            )
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def generate_answer(self, question: str, context: str, template_name: str = "rag") -> LLMResponse:
        """生成答案（同步版本）"""
        start_time = time.time()
        
        try:
            self._stats["total_requests"] += 1
            
            # 构建提示词
            prompt = self._build_prompt(template_name, {"context": context, "question": question})
            
            # 调用LLM
            response = self.client.chat.completions.create(
                model=self.config.models.llm.name,
                messages=[
                    {"role": "system", "content": self.config.models.llm.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.models.llm.temperature,
                top_p=self.config.models.llm.top_p,
                max_tokens=self.config.models.llm.max_tokens,
                stop=self.config.models.llm.stop
            )
            
            content = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            # 计算质量分数
            quality_score = self._calculate_quality_score(content, question, context)
            
            # 更新统计信息
            self._stats["successful_requests"] += 1
            self._stats["total_processing_time"] += processing_time
            self._stats["average_response_length"] = (
                (self._stats["average_response_length"] * (self._stats["successful_requests"] - 1) + len(content)) /
                self._stats["successful_requests"]
            )
            
            return LLMResponse(
                content=content,
                model=self.config.models.llm.name,
                usage=response.usage.model_dump() if response.usage else {},
                processing_time=processing_time,
                quality_score=quality_score,
                metadata={
                    "template": template_name,
                    "question_length": len(question),
                    "context_length": len(context),
                    "response_length": len(content)
                }
            )
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            self.logger.error(f"LLM生成答案失败: {str(e)}")
            raise
    
    def generate_answer_stream(self, question: str, context: str, template_name: str = "rag") -> Generator[str, None, None]:
        """生成答案（流式版本）"""
        try:
            # 构建提示词
            prompt = self._build_prompt(template_name, {"context": context, "question": question})
            
            # 调用LLM（流式）
            response = self.client.chat.completions.create(
                model=self.config.models.llm.name,
                messages=[
                    {"role": "system", "content": self.config.models.llm.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.models.llm.temperature,
                top_p=self.config.models.llm.top_p,
                max_tokens=self.config.models.llm.max_tokens,
                stop=self.config.models.llm.stop,
                stream=True
            )
            
            # 流式返回内容
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"LLM流式生成失败: {str(e)}")
            yield f"抱歉，生成答案时出现错误: {str(e)}"
    
    async def generate_answer_async(self, question: str, context: str, template_name: str = "rag") -> LLMResponse:
        """生成答案（异步版本）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.generate_answer,
            question,
            context,
            template_name
        )
    
    def batch_generate(self, questions: List[Tuple[str, str]], template_name: str = "rag") -> List[LLMResponse]:
        """批量生成答案"""
        responses = []
        
        try:
            self.logger.info(f"开始批量生成 {len(questions)} 个答案")
            
            # 并发处理
            futures = []
            for question, context in questions:
                future = self._executor.submit(
                    self.generate_answer,
                    question,
                    context,
                    template_name
                )
                futures.append(future)
            
            # 收集结果
            for future in futures:
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    self.logger.error(f"批量生成中的单个任务失败: {str(e)}")
                    # 创建错误响应
                    error_response = LLMResponse(
                        content=f"生成失败: {str(e)}",
                        model=self.config.models.llm.name,
                        usage={},
                        processing_time=0.0,
                        quality_score=0.0,
                        metadata={"error": str(e)}
                    )
                    responses.append(error_response)
            
            self.logger.info(f"批量生成完成，成功 {len([r for r in responses if r.quality_score > 0])} 个")
            return responses
            
        except Exception as e:
            self.logger.error(f"批量生成失败: {str(e)}")
            raise
    
    def _build_prompt(self, template_name: str, variables: Dict[str, str]) -> str:
        """构建提示词"""
        template = self.prompt_templates.get(template_name)
        if not template:
            self.logger.warning(f"未找到模板 {template_name}，使用默认RAG模板")
            template = self.prompt_templates["rag"]
        
        try:
            return template.template.format(**variables)
        except KeyError as e:
            self.logger.error(f"提示词模板变量缺失: {e}")
            # 使用默认值
            default_vars = {var: variables.get(var, f"[{var}]") for var in template.variables}
            return template.template.format(**default_vars)
    
    def _calculate_quality_score(self, response: str, question: str, context: str) -> float:
        """计算响应质量分数"""
        score = 0.0
        
        # 1. 长度检查（0-20分）
        if len(response) > 50:
            score += 20
        elif len(response) > 20:
            score += 10
        
        # 2. 相关性检查（0-30分）
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        if question_words:
            relevance = len(question_words.intersection(response_words)) / len(question_words)
            score += relevance * 30
        
        # 3. 上下文利用检查（0-25分）
        if context and len(context) > 0:
            context_utilization = min(len(response) / len(context) * 10, 25)
            score += context_utilization
        
        # 4. 结构完整性检查（0-25分）
        if response.endswith(('.', '。', '!', '！', '?', '？')):
            score += 10
        if len(response.split('.')) > 1:
            score += 15
        
        return min(score, 100.0) / 100.0
    
    def add_prompt_template(self, template: PromptTemplate):
        """添加自定义提示词模板"""
        self.prompt_templates[template.name] = template
        self.logger.info(f"添加提示词模板: {template.name}")
    
    def get_prompt_template(self, template_name: str) -> Optional[PromptTemplate]:
        """获取提示词模板"""
        return self.prompt_templates.get(template_name)
    
    def list_prompt_templates(self) -> List[str]:
        """列出所有提示词模板"""
        return list(self.prompt_templates.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self._stats["total_requests"]
        success_rate = 0.0
        avg_time = 0.0
        
        if total_requests > 0:
            success_rate = self._stats["successful_requests"] / total_requests
            avg_time = self._stats["total_processing_time"] / self._stats["successful_requests"]
        
        return {
            "total_requests": total_requests,
            "successful_requests": self._stats["successful_requests"],
            "failed_requests": self._stats["failed_requests"],
            "success_rate": success_rate,
            "average_processing_time": avg_time,
            "average_response_length": self._stats["average_response_length"],
            "available_templates": self.list_prompt_templates()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_response_length": 0.0
        }
        self.logger.info("LLM服务统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            self.logger.info("LLM服务资源清理完成")
        except Exception as e:
            self.logger.error(f"LLM服务资源清理失败: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 