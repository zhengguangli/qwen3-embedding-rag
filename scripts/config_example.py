#!/usr/bin/env python3
"""
RAG系统配置使用示例
展示新配置系统的各种功能和使用方法
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import RAGConfig

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("基本使用示例")
    print("=" * 60)
    
    # 创建默认配置
    config = RAGConfig()
    
    # 打印配置信息
    config.print_config()
    
    # 获取配置值
    print(f"\n获取配置值:")
    print(f"API基础URL: {config.get('api.openai_base_url')}")
    print(f"Milvus URI: {config.database.milvus_uri}")
    print(f"嵌入模型: {config.models.embedding_model}")
    print(f"分块大小: {config.data.chunk_size}")

def example_file_loading():
    """文件加载示例"""
    print("\n" + "=" * 60)
    print("文件加载示例")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 从JSON文件加载配置
    if config.load_from_file("config.json"):
        print("✅ 从config.json加载配置成功")
        print(f"配置文件: {config._config_file}")
    else:
        print("❌ 从config.json加载配置失败")
    
    # 从YAML文件加载配置
    if config.load_from_file("config.template.yaml"):
        print("✅ 从config.template.yaml加载配置成功")
        print(f"配置文件: {config._config_file}")
    else:
        print("❌ 从config.template.yaml加载配置失败")

def example_environment_variables():
    """环境变量示例"""
    print("\n" + "=" * 60)
    print("环境变量示例")
    print("=" * 60)
    
    # 设置环境变量
    os.environ["OPENAI_API_KEY"] = "test_api_key_123"
    os.environ["MILVUS_URI"] = "http://test-milvus:19530"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    # 创建配置（会自动加载环境变量）
    config = RAGConfig()
    
    print("环境变量覆盖的配置:")
    print(f"API密钥: {config.api.openai_api_key}")
    print(f"Milvus URI: {config.database.milvus_uri}")
    print(f"日志级别: {config.logging.log_level}")
    
    # 清理环境变量
    del os.environ["OPENAI_API_KEY"]
    del os.environ["MILVUS_URI"]
    del os.environ["LOG_LEVEL"]

def example_dynamic_configuration():
    """动态配置示例"""
    print("\n" + "=" * 60)
    print("动态配置示例")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 动态修改配置
    print("修改前的配置:")
    print(f"API超时: {config.api.timeout}秒")
    print(f"LLM温度: {config.models.llm_temperature}")
    print(f"搜索限制: {config.search.search_limit}")
    
    # 修改配置
    config.set("api.timeout", 60)
    config.set("models.llm_temperature", 0.5)
    config.set("search.search_limit", 20)
    
    print("\n修改后的配置:")
    print(f"API超时: {config.api.timeout}秒")
    print(f"LLM温度: {config.models.llm_temperature}")
    print(f"搜索限制: {config.search.search_limit}")

def example_config_validation():
    """配置验证示例"""
    print("\n" + "=" * 60)
    print("配置验证示例")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 验证有效配置
    print("验证默认配置:")
    if config.validate():
        print("✅ 默认配置有效")
    else:
        print("❌ 默认配置无效")
    
    # 尝试设置无效配置
    print("\n尝试设置无效配置:")
    
    # 设置无效的分块重叠（大于分块大小）
    if config.set("data.chunk_overlap", 1500):  # 大于chunk_size(1000)
        print("❌ 应该失败：分块重叠大于分块大小")
    else:
        print("✅ 正确拒绝无效配置：分块重叠大于分块大小")
    
    # 设置无效的重排序top-k（大于搜索限制）
    if config.set("search.rerank_top_k", 15):  # 大于search_limit(10)
        print("❌ 应该失败：重排序top-k大于搜索限制")
    else:
        print("✅ 正确拒绝无效配置：重排序top-k大于搜索限制")
    
    # 设置无效的温度值
    if config.set("models.llm_temperature", 3.0):  # 超出范围(0.0-2.0)
        print("❌ 应该失败：温度值超出范围")
    else:
        print("✅ 正确拒绝无效配置：温度值超出范围")

def example_config_save_load():
    """配置保存和加载示例"""
    print("\n" + "=" * 60)
    print("配置保存和加载示例")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 修改一些配置
    config.set("api.timeout", 45)
    config.set("models.llm_temperature", 0.8)
    config.set("search.search_limit", 15)
    
    # 保存配置到文件
    test_config_file = "test_config.json"
    if config.save_to_file(test_config_file):
        print(f"✅ 配置已保存到: {test_config_file}")
    else:
        print(f"❌ 配置保存失败: {test_config_file}")
    
    # 创建新配置实例并加载
    new_config = RAGConfig()
    if new_config.load_from_file(test_config_file):
        print("✅ 配置加载成功")
        print(f"API超时: {new_config.api.timeout}秒")
        print(f"LLM温度: {new_config.models.llm_temperature}")
        print(f"搜索限制: {new_config.search.search_limit}")
    else:
        print("❌ 配置加载失败")
    
    # 清理测试文件
    try:
        os.remove(test_config_file)
        print(f"✅ 测试文件已清理: {test_config_file}")
    except:
        pass

def example_config_comparison():
    """配置比较示例"""
    print("\n" + "=" * 60)
    print("配置比较示例")
    print("=" * 60)
    
    # 创建两个不同的配置
    config1 = RAGConfig()
    config2 = RAGConfig()
    
    # 修改config2的一些设置
    config2.set("api.timeout", 60)
    config2.set("models.llm_temperature", 0.5)
    config2.set("search.search_limit", 20)
    
    # 转换为字典进行比较
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    print("配置差异:")
    if dict1["api"]["timeout"] != dict2["api"]["timeout"]:
        print(f"API超时: {dict1['api']['timeout']} vs {dict2['api']['timeout']}")
    
    if dict1["models"]["llm_temperature"] != dict2["models"]["llm_temperature"]:
        print(f"LLM温度: {dict1['models']['llm_temperature']} vs {dict2['models']['llm_temperature']}")
    
    if dict1["search"]["search_limit"] != dict2["search"]["search_limit"]:
        print(f"搜索限制: {dict1['search']['search_limit']} vs {dict2['search']['search_limit']}")

def main():
    """主函数"""
    print("RAG系统配置使用示例")
    print("展示新配置系统的各种功能")
    
    try:
        # 运行各种示例
        example_basic_usage()
        example_file_loading()
        example_environment_variables()
        example_dynamic_configuration()
        example_config_validation()
        example_config_save_load()
        example_config_comparison()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 