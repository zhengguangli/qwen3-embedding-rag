#!/usr/bin/env python3
"""
RAG系统主入口（简化版）

提供简单的命令行接口，业务逻辑统一由app.py处理
"""

import argparse
import sys
from typing import Optional

from .app import RAGApp


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="RAG 系统 - 基于 Milvus 的检索增强生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.rag.main --question "什么是向量数据库？"
  python -m src.rag.main --config config.json --force-recreate
  python -m src.rag.main  # 交互模式
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
        "--env",
        type=str,
        help="指定环境（如dev/prod/test），优先于RAG_ENV环境变量"
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
    
    try:
        # 创建应用程序实例
        app = RAGApp(config_file=args.config, env=args.env)
        
        # 初始化应用程序
        app.initialize()
        
        # 如果需要强制重建集合
        if args.force_recreate:
            app.setup_collection(force_recreate=True)
        
        # 根据是否有问题参数决定运行模式
        if args.question:
            # 单问题模式
            answer = app.ask_question(args.question, args.output_file)
            app._display_result_summary(args.question, answer, args.output_file)
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
