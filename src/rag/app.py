#!/usr/bin/env python3
"""
RAG应用程序核心模块

提供统一的业务逻辑接口，包括：
- 应用程序初始化和配置管理
- RAG管道创建和管理
- 问答处理和交互模式
- 错误处理和日志记录
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from openai import OpenAI

from .config import RAGConfig
from .pipeline import RAGPipeline
from .utils import setup_logging, check_dependencies
from .embedding import EmbeddingService
from .exceptions import RAGException, handle_exception


class RAGApp:
    """RAG应用程序核心类"""
    
    def __init__(self, config_file: Optional[str] = None, env: Optional[str] = None):
        """初始化应用程序
        
        Args:
            config_file: 配置文件路径
            env: 环境名称（dev/prod/test等）
        """
        self.config_file = config_file
        self.env = env
        self.config: Optional[RAGConfig] = None
        self.pipeline: Optional[RAGPipeline] = None
        self.logger = setup_logging()
        
        # 确保输出目录存在
        self._ensure_output_directories()
    
    def _ensure_output_directories(self) -> None:
        """确保输出目录存在"""
        directories = ["answers", "logs"]
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"确保目录存在: {dir_name}")
    
    def initialize(self) -> None:
        """初始化应用程序（配置 + 管道）"""
        try:
            # 检查依赖
            check_dependencies()
            
            # 加载配置
            self.load_config()
            
            # 初始化管道
            self.initialize_pipeline()
            
            self.logger.info("RAG应用程序初始化完成")
            
        except Exception as e:
            self.logger.error(f"应用程序初始化失败: {str(e)}")
            raise
    
    def load_config(self) -> RAGConfig:
        """加载配置"""
        try:
            self.config = RAGConfig(config_file=self.config_file, env=self.env)
            self.logger.info(f"配置加载成功，当前环境: {self.config.env}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {str(e)}")
            raise
    
    def initialize_pipeline(self) -> RAGPipeline:
        """初始化RAG管道"""
        if not self.config:
            raise RuntimeError("配置未加载，无法初始化管道")
        
        try:
            self.logger.info("初始化RAG管道...")
            
            # 创建OpenAI客户端
            openai_client = OpenAI(
                api_key=self.config.api.openai_api_key,
                base_url=self.config.api.openai_base_url,
                timeout=self.config.api.timeout
            )
            
            # 初始化管道
            self.pipeline = RAGPipeline(self.config)
            
            self.logger.info("RAG管道初始化成功")
            return self.pipeline
            
        except Exception as e:
            self.logger.error(f"管道初始化失败: {str(e)}")
            raise
    
    def setup_collection(self, force_recreate: bool = False) -> bool:
        """设置Milvus集合"""
        if not self.pipeline:
            raise RuntimeError("管道未初始化")
        
        try:
            self.logger.info(f"设置Milvus集合，force_recreate={force_recreate}")
            return self.pipeline.setup_collection(force_recreate=force_recreate)
            
        except Exception as e:
            self.logger.error(f"设置Milvus集合失败: {str(e)}")
            raise
    
    def ask_question(self, question: str, output_file: Optional[str] = None) -> str:
        """处理单个问题
        
        Args:
            question: 用户问题
            output_file: 输出文件名（可选）
            
        Returns:
            生成的答案
        """
        if not self.pipeline:
            raise RuntimeError("管道未初始化")
        
        try:
            self.logger.info(f"处理问题: {question}")
            
            # 生成答案
            answer = self.pipeline.run(question)
            
            # 保存答案到文件（如果指定）
            if output_file:
                output_path = Path("answers") / output_file
                self._save_answer(question, answer, output_path)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"处理问题失败: {str(e)}")
            raise
    
    def run_interactive_mode(self) -> None:
        """运行交互模式"""
        if not self.pipeline:
            raise RuntimeError("管道未初始化")
        
        print("RAG 系统已启动，输入问题开始对话（输入 'quit' 退出）")
        
        while True:
            try:
                question = input("\n请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("退出程序...")
                    break
                
                if not question:
                    continue
                
                # 生成时间戳文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"answer_{timestamp}.txt"
                
                # 处理问题
                answer = self.ask_question(question, output_file)
                
                # 显示结果摘要
                self._display_result_summary(question, answer, output_file)
                
            except KeyboardInterrupt:
                print("\n\n用户中断，退出程序...")
                break
            except Exception as e:
                self.logger.error(f"处理问题时出错: {str(e)}")
                print(f"错误: {str(e)}")
    
    def _save_answer(self, question: str, answer: str, output_path: Path) -> None:
        """保存答案到文件"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"问题: {question}\n")
                f.write(f"答案: {answer}\n")
                f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"答案已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存答案失败: {str(e)}")
            raise
    
    def _display_result_summary(self, question: str, answer: str, output_file: str) -> None:
        """显示结果摘要"""
        output_path = Path("answers") / output_file
        
        print(f"\n问题: {question}")
        print(f"答案已保存到: {output_path}")
        print(f"答案长度: {len(answer)} 字符")
        
        # 显示答案摘要
        summary_length = 200
        if len(answer) > summary_length:
            summary = answer[:summary_length] + "..."
        else:
            summary = answer
        print(f"答案摘要: {summary}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "app_initialized": self.config is not None and self.pipeline is not None,
            "config_loaded": self.config is not None,
            "pipeline_initialized": self.pipeline is not None,
            "current_env": self.env or os.getenv("RAG_ENV", "dev"),
            "config_file": self.config_file
        }
        
        if self.config:
            status.update({
                "milvus_uri": self.config.database.milvus_uri,
                "collection_name": self.config.database.collection_name,
                "embedding_model": self.config.models.embedding.name,
                "llm_model": self.config.models.llm.name
            })
        
        return status
    
    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            if self.config and self.config.has_changed():
                self.logger.info("检测到配置文件变更，重新加载...")
                self.load_config()
                
                # 重新初始化管道
                self.initialize_pipeline()
                
                self.logger.info("配置重新加载完成")
                return True
            else:
                self.logger.info("配置文件未变更")
                return False
                
        except Exception as e:
            self.logger.error(f"配置重新加载失败: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.pipeline:
                # 如果pipeline有cleanup方法，调用它
                if hasattr(self.pipeline, 'cleanup'):
                    self.pipeline.cleanup()
            
            self.logger.info("应用程序资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {str(e)}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


# 便捷函数
def create_app(config_file: Optional[str] = None, env: Optional[str] = None) -> RAGApp:
    """创建RAG应用程序实例"""
    return RAGApp(config_file=config_file, env=env)


def quick_ask(question: str, config_file: Optional[str] = None, env: Optional[str] = None) -> str:
    """快速问答（一次性使用）"""
    with create_app(config_file=config_file, env=env) as app:
        return app.ask_question(question) 