#!/usr/bin/env python3
"""
测试优化后的配置系统
验证新增功能和改进
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import RAGConfig, MetricType, ConsistencyLevel, IndexType, LogLevel

def test_enum_validation():
    """测试枚举类型验证"""
    print("=" * 60)
    print("测试枚举类型验证")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 测试有效的枚举值
    print("✅ 测试有效枚举值:")
    print(f"  距离度量类型: {config.database.metric_type}")
    print(f"  一致性级别: {config.database.consistency_level}")
    print(f"  索引类型: {config.database.index_type}")
    print(f"  日志级别: {config.logging.log_level}")
    
    # 测试枚举值设置
    config.set("database.metric_type", MetricType.COSINE)
    config.set("database.consistency_level", ConsistencyLevel.EVENTUALLY)
    config.set("database.index_type", IndexType.HNSW)
    config.set("logging.log_level", LogLevel.DEBUG)
    
    print(f"  设置后的距离度量类型: {config.database.metric_type}")
    print(f"  设置后的一致性级别: {config.database.consistency_level}")
    print(f"  设置后的索引类型: {config.database.index_type}")
    print(f"  设置后的日志级别: {config.logging.log_level}")

def test_enhanced_validation():
    """测试增强的验证功能"""
    print("\n" + "=" * 60)
    print("测试增强的验证功能")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 测试URL验证
    print("✅ 测试URL验证:")
    try:
        config.set("api.openai_base_url", "invalid-url")
        print("❌ 应该失败：无效URL")
    except Exception as e:
        print(f"✅ 正确拒绝无效URL: {e}")
    
    # 测试集合名称验证
    print("\n✅ 测试集合名称验证:")
    try:
        config.set("database.collection_name", "invalid collection name!")
        print("❌ 应该失败：无效集合名称")
    except Exception as e:
        print(f"✅ 正确拒绝无效集合名称: {e}")
    
    # 测试编码验证
    print("\n✅ 测试编码验证:")
    try:
        config.set("data.encoding", "invalid-encoding")
        print("❌ 应该失败：无效编码")
    except Exception as e:
        print(f"✅ 正确拒绝无效编码: {e}")
    
    # 测试文件格式验证
    print("\n✅ 测试文件格式验证:")
    try:
        config.set("data.supported_formats", ["invalid", "format"])
        print("❌ 应该失败：无效文件格式")
    except Exception as e:
        print(f"✅ 正确拒绝无效文件格式: {e}")

def test_config_cache():
    """测试配置缓存功能"""
    print("\n" + "=" * 60)
    print("测试配置缓存功能")
    print("=" * 60)
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config = {
            "api": {"timeout": 60},
            "models": {"llm_temperature": 0.5}
        }
        json.dump(temp_config, f)
        temp_file = f.name
    
    try:
        config = RAGConfig(temp_file)
        
        # 检查缓存
        print(f"✅ 配置文件: {config._config_file}")
        print(f"✅ 缓存存在: {config._cache is not None}")
        if config._cache:
            print(f"✅ 缓存哈希: {config._cache.hash[:16]}...")
            print(f"✅ 缓存时间戳: {config._cache.timestamp}")
        
        # 测试重复加载（应该使用缓存）
        print("\n✅ 测试重复加载（应该使用缓存）:")
        start_time = os.times().elapsed
        config.load_from_file(temp_file)
        end_time = os.times().elapsed
        print(f"  加载时间: {end_time - start_time:.6f}秒")
        
    finally:
        os.unlink(temp_file)

def test_temporary_config():
    """测试临时配置功能"""
    print("\n" + "=" * 60)
    print("测试临时配置功能")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 保存原始值
    original_timeout = config.api.timeout
    original_temperature = config.models.llm_temperature
    
    print(f"✅ 原始配置:")
    print(f"  API超时: {original_timeout}")
    print(f"  LLM温度: {original_temperature}")
    
    # 使用临时配置
    with config.temporary_config(
        api__timeout=120,
        models__llm_temperature=0.9
    ):
        print(f"\n✅ 临时配置:")
        print(f"  API超时: {config.api.timeout}")
        print(f"  LLM温度: {config.models.llm_temperature}")
    
    # 检查是否恢复
    print(f"\n✅ 恢复后配置:")
    print(f"  API超时: {config.api.timeout}")
    print(f"  LLM温度: {config.models.llm_temperature}")
    
    # 验证恢复
    assert config.api.timeout == original_timeout, "API超时未恢复"
    assert config.models.llm_temperature == original_temperature, "LLM温度未恢复"
    print("✅ 配置恢复验证通过")

def test_config_diff():
    """测试配置差异比较"""
    print("\n" + "=" * 60)
    print("测试配置差异比较")
    print("=" * 60)
    
    config1 = RAGConfig()
    config2 = RAGConfig()
    
    # 修改config2的一些配置
    config2.set("api.timeout", 60)
    config2.set("models.llm_temperature", 0.5)
    config2.set("search.search_limit", 20)
    
    # 获取差异
    diff = config1.get_diff(config2)
    
    print("✅ 配置差异:")
    for key, (val1, val2) in diff.items():
        print(f"  {key}: {val1} -> {val2}")
    
    assert len(diff) == 3, f"期望3个差异，实际{len(diff)}个"
    print("✅ 差异比较验证通过")

def test_config_migration():
    """测试配置迁移功能"""
    print("\n" + "=" * 60)
    print("测试配置迁移功能")
    print("=" * 60)
    
    # 旧格式配置
    old_config = {
        "embedding_model": "old-embedding-model",
        "reranker_model": "old-reranker-model",
        "llm_model": "old-llm-model",
        "milvus_uri": "http://old-milvus:19530",
        "collection_name": "old_collection",
        "embedding_dim": 512,
        "data_path_glob": "old/path/*.md",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "search_limit": 5,
        "rerank_top_k": 2,
        "batch_size": 16,
        "max_workers": 2,
        "cache_size": 500,
        "system_prompt": "旧的系统提示词"
    }
    
    config = RAGConfig()
    
    # 执行迁移
    success = config.migrate_from_old_format(old_config)
    
    if success:
        print("✅ 配置迁移成功:")
        print(f"  嵌入模型: {config.models.embedding_model}")
        print(f"  重排序模型: {config.models.reranker_model}")
        print(f"  LLM模型: {config.models.llm_model}")
        print(f"  Milvus URI: {config.database.milvus_uri}")
        print(f"  集合名称: {config.database.collection_name}")
        print(f"  嵌入维度: {config.database.embedding_dim}")
        print(f"  数据路径: {config.data.data_path_glob}")
        print(f"  分块大小: {config.data.chunk_size}")
        print(f"  分块重叠: {config.data.chunk_overlap}")
        print(f"  搜索限制: {config.search.search_limit}")
        print(f"  重排序top-k: {config.search.rerank_top_k}")
        print(f"  批处理大小: {config.performance.batch_size}")
        print(f"  最大工作线程: {config.performance.max_workers}")
        print(f"  缓存大小: {config.performance.cache_size}")
        print(f"  系统提示词: {config.prompts.system_prompt}")
    else:
        print("❌ 配置迁移失败")

def test_detailed_print():
    """测试详细打印功能"""
    print("\n" + "=" * 60)
    print("测试详细打印功能")
    print("=" * 60)
    
    config = RAGConfig()
    
    print("✅ 简要配置信息:")
    config.print_config(detailed=False)
    
    print("\n✅ 详细配置信息:")
    config.print_config(detailed=True)

def test_enhanced_save():
    """测试增强的保存功能"""
    print("\n" + "=" * 60)
    print("测试增强的保存功能")
    print("=" * 60)
    
    config = RAGConfig()
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        # 保存带注释的配置
        success = config.save_to_file(temp_file, include_comments=True)
        if success:
            print("✅ 配置保存成功（带注释）")
            
            # 读取并检查注释
            with open(temp_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            
            print("✅ 检查保存的配置:")
            print(f"  版本: {saved_config.get('_version', 'N/A')}")
            print(f"  描述: {saved_config.get('_description', 'N/A')}")
            print(f"  API注释: {saved_config.get('api', {}).get('_comment', 'N/A')}")
            print(f"  数据库注释: {saved_config.get('database', {}).get('_comment', 'N/A')}")
        else:
            print("❌ 配置保存失败")
        
        # 保存不带注释的配置
        temp_file_no_comments = temp_file.replace('.json', '_no_comments.json')
        success = config.save_to_file(temp_file_no_comments, include_comments=False)
        if success:
            print("✅ 配置保存成功（不带注释）")
            
            # 读取并检查无注释
            with open(temp_file_no_comments, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            
            print("✅ 检查无注释配置:")
            print(f"  无版本字段: {'_version' not in saved_config}")
            print(f"  无描述字段: {'_description' not in saved_config}")
            print(f"  API无注释: '_comment' not in saved_config['api']")
        else:
            print("❌ 配置保存失败")
            
    finally:
        # 清理临时文件
        for file_path in [temp_file, temp_file.replace('.json', '_no_comments.json')]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def main():
    """主函数"""
    print("测试优化后的配置系统")
    print("验证新增功能和改进")
    
    try:
        test_enum_validation()
        test_enhanced_validation()
        test_config_cache()
        test_temporary_config()
        test_config_diff()
        test_config_migration()
        test_detailed_print()
        test_enhanced_save()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 