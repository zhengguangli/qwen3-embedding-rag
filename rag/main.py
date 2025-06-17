#!/usr/bin/env python3
"""
RAG (检索增强生成) 系统主程序
使用 OpenAI 兼容 API 和 Milvus 向量数据库构建企业知识库
"""

import argparse
import json
import os
from datetime import datetime
import sys
from .config import RAGConfig
from .pipeline import RAGPipeline
from .utils import setup_logging, check_dependencies

# 设置日志
logger = setup_logging()

def ensure_output_dirs():
    """确保输出目录存在"""
    dirs = ["answers", "logs"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG 系统 - 基于 Milvus 的检索增强生成")
    parser.add_argument("--question", "-q", help="要回答的问题")
    parser.add_argument("--force-recreate", "-f", action="store_true", help="强制重建 Milvus 集合")
    parser.add_argument("--output-file", "-o", default="answer.txt", help="答案输出文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    args = parser.parse_args()

    try:
        # 确保输出目录存在
        ensure_output_dirs()
        
        # 检查依赖
        check_dependencies()
        
        # 加载配置
        config = RAGConfig()
        if args.config:
            try:
                with open(args.config, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    # 支持分组配置
                    for key, value in config_data.items():
                        if hasattr(config, key):
                            group = getattr(config, key)
                            if isinstance(value, dict):
                                for subkey, subval in value.items():
                                    if hasattr(group, subkey):
                                        setattr(group, subkey, subval)
                            else:
                                setattr(config, key, value)
            except Exception as e:
                logger.error(f"配置文件加载失败: {str(e)}")
                return
        
        # 打印配置
        config.print_config()
        
        # 初始化 RAG 管道
        pipeline = RAGPipeline(config)
        
        if args.force_recreate:
            logger.info("强制重建 Milvus 集合...")
            pipeline.setup_collection(force_recreate=args.force_recreate)
        
        if args.question:
            logger.info(f"处理问题: {args.question}")
            
            # 获取答案
            answer = pipeline.run(args.question)
            
            # 保存答案到文件
            output_path = os.path.join("answers", args.output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"问题: {args.question}\n")
                f.write(f"答案: {answer}\n")
                f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 在终端显示简短摘要
            print(f"\n问题: {args.question}")
            print(f"答案已保存到: {output_path}")
            print(f"答案长度: {len(answer)} 字符")
            print(f"答案摘要: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
        else:
            # 交互模式
            print("RAG 系统已启动，输入问题开始对话（输入 'quit' 退出）")
            while True:
                try:
                    question = input("\n请输入问题: ").strip()
                    if question.lower() in ['quit', 'exit', '退出']:
                        break
                    if not question:
                        continue
                    
                    logger.info(f"处理问题: {question}")
                    answer = pipeline.run(question)
                    
                    # 保存答案到文件
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"answer_{timestamp}.txt"
                    output_path = os.path.join("answers", output_file)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(f"问题: {question}\n")
                        f.write(f"答案: {answer}\n")
                        f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    print(f"\n答案已保存到: {output_path}")
                    print(f"答案摘要: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                    
                except KeyboardInterrupt:
                    print("\n\n退出程序...")
                    break
                except Exception as e:
                    logger.error(f"处理问题时出错: {str(e)}")
                    print(f"错误: {str(e)}")
    
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        print(f"程序错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
