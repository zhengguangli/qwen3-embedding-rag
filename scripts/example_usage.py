#!/usr/bin/env python3
"""
RAG 系统使用示例
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.utils import setup_logging

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("RAG 系统基本使用示例")
    print("=" * 60)
    
    # 设置日志
    logger = setup_logging("INFO")
    
    # 创建配置
    config = RAGConfig()
    
    # 打印配置
    config.print_config()
    
    # 创建RAG管道
    pipeline = RAGPipeline(config)
    
    # 设置集合（如果不存在则创建）
    pipeline.setup_collection(force_recreate=False)
    
    # 测试问题
    questions = [
        "Milvus 的数据是如何存储的？",
        "什么是向量数据库？",
        "Milvus 支持哪些索引类型？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 40)
        
        try:
            answer = pipeline.run(question)
            print(f"答案: {answer}")
        except Exception as e:
            print(f"错误: {str(e)}")
        
        print("-" * 40)


def example_with_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("自定义配置示例")
    print("=" * 60)
    
    # 创建自定义配置
    config = RAGConfig(
        embedding_model="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
        reranker_model="qwen3:4b",
        llm_model="qwen3:4b",
        embedding_dim=1024,
        max_workers=2,  # 减少并发数
        cache_size=500,  # 减少缓存大小
        max_retries=2,   # 减少重试次数
        search_limit=5,  # 减少搜索数量
        rerank_top_k=2   # 减少重排序数量
    )
    
    # 打印配置
    config.print_config()
    
    # 创建RAG管道
    pipeline = RAGPipeline(config)
    
    # 设置集合
    pipeline.setup_collection(force_recreate=False)
    
    # 测试单个问题
    question = "Milvus 的架构是怎样的？"
    print(f"\n问题: {question}")
    print("-" * 40)
    
    try:
        answer = pipeline.run(question)
        print(f"答案: {answer}")
    except Exception as e:
        print(f"错误: {str(e)}")


def main():
    """主函数"""
    print("RAG 系统使用示例")
    print("请选择要运行的示例:")
    print("1. 基本使用示例")
    print("2. 自定义配置示例")
    
    try:
        choice = input("\n请输入选择 (1-2): ").strip()
        
        if choice == "1":
            example_basic_usage()
        elif choice == "2":
            example_with_custom_config()
        else:
            print("无效选择，运行基本示例...")
            example_basic_usage()
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")


if __name__ == "__main__":
    main() 