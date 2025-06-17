#!/usr/bin/env python3
"""
RAG (检索增强生成) 系统主程序
使用 OpenAI 兼容 API 和 Milvus 向量数据库构建企业知识库
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from src.rag.config import RAGConfig
from src.rag.pipeline import RAGPipeline
from src.rag.utils import setup_logging, check_dependencies
from openai import OpenAI
from src.rag.embedding import EmbeddingService


class RAGApplication:
    """RAG应用程序主类"""
    
    def __init__(self) -> None:
        self.logger = setup_logging()
        self.config: Optional[RAGConfig] = None
        self.pipeline: Optional[RAGPipeline] = None
    
    def ensure_output_directories(self) -> None:
        """确保输出目录存在"""
        directories = ["answers", "logs"]
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"确保目录存在: {dir_name}")
    
    def load_config(self, config_path: Optional[str] = None) -> RAGConfig:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            
        Returns:
            加载的配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        try:
            config = RAGConfig()
            
            if config_path:
                self.logger.info(f"加载配置文件: {config_path}")
                config_data = self._load_config_file(config_path)
                self._apply_config_data(config, config_data)
            else:
                self.logger.info("使用默认配置")
            
            self.config = config
            return config
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {str(e)}")
            raise
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件内容"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON配置文件格式错误: {str(e)}")
        except Exception as e:
            raise ValueError(f"配置文件读取失败: {str(e)}")
    
    def _apply_config_data(self, config: RAGConfig, config_data: Dict[str, Any]) -> None:
        """应用配置数据到配置对象"""
        for key, value in config_data.items():
            if hasattr(config, key):
                group = getattr(config, key)
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if hasattr(group, subkey):
                            setattr(group, subkey, subval)
                        else:
                            self.logger.warning(f"未知配置项: {key}.{subkey}")
                else:
                    setattr(config, key, value)
            else:
                self.logger.warning(f"未知配置组: {key}")
    
    def initialize_pipeline(self) -> RAGPipeline:
        """初始化RAG管道
        
        Returns:
            初始化的管道对象
            
        Raises:
            RuntimeError: 管道初始化失败
        """
        if not self.config:
            raise RuntimeError("配置未加载，无法初始化管道")
        
        try:
            self.logger.info("初始化RAG管道...")
            self.config.print_config()
            
            openai_client = OpenAI(
                api_key=self.config.api.openai_api_key,
                base_url=self.config.api.openai_base_url
            )
            embedding_service = EmbeddingService(self.config, openai_client)
            
            pipeline = RAGPipeline(self.config)
            self.pipeline = pipeline
            
            self.logger.info("RAG管道初始化成功")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"管道初始化失败: {str(e)}")
            raise RuntimeError(f"RAG管道初始化失败: {str(e)}")
    
    def process_single_question(self, question: str, output_file: str) -> str:
        """处理单个问题
        
        Args:
            question: 用户问题
            output_file: 输出文件名
            
        Returns:
            生成的答案
        """
        if not self.pipeline:
            raise RuntimeError("管道未初始化")
        
        self.logger.info(f"处理问题: {question}")
        
        # 生成答案
        answer = self.pipeline.run(question)
        
        # 保存答案到文件
        output_path = Path("answers") / output_file
        self._save_answer(question, answer, output_path)
        
        # 显示结果摘要
        self._display_result_summary(question, answer, output_path)
        
        return answer
    
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
                self.process_single_question(question, output_file)
                
            except KeyboardInterrupt:
                print("\n\n用户中断，退出程序...")
                break
            except Exception as e:
                self.logger.error(f"处理问题时出错: {str(e)}")
                print(f"错误: {str(e)}")
    
    def _save_answer(self, question: str, answer: str, output_path: Path) -> None:
        """保存答案到文件"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"问题: {question}\n")
                f.write(f"答案: {answer}\n")
                f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"答案已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存答案失败: {str(e)}")
            raise
    
    def _display_result_summary(self, question: str, answer: str, output_path: Path) -> None:
        """显示结果摘要"""
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


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="RAG 系统 - 基于 Milvus 的检索增强生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m rag.main --question "什么是向量数据库？"
  python -m rag.main --config config.json --force-recreate
  python -m rag.main  # 交互模式
        """
    )
    
    parser.add_argument(
        "--question", "-q",
        help="要回答的问题"
    )
    parser.add_argument(
        "--force-recreate", "-f",
        action="store_true",
        help="强制重建 Milvus 集合"
    )
    parser.add_argument(
        "--output-file", "-o",
        default="answer.txt",
        help="答案输出文件路径（默认: answer.txt）"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径（支持JSON和YAML格式）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别（默认: INFO）"
    )
    
    return parser


def main() -> None:
    """主函数"""
    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 创建应用程序实例
    app = RAGApplication()
    
    try:
        # 检查依赖
        check_dependencies()
        
        # 确保输出目录存在
        app.ensure_output_directories()
        
        # 加载配置
        config = app.load_config(args.config)
        
        # 初始化管道
        pipeline = app.initialize_pipeline()
        
        # 如果需要强制重建集合
        if args.force_recreate:
            app.logger.info("强制重建 Milvus 集合...")
            pipeline.setup_collection(force_recreate=True)
        
        # 根据是否有问题参数决定运行模式
        if args.question:
            # 单问题模式
            app.process_single_question(args.question, args.output_file)
        else:
            # 交互模式
            app.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
