#!/usr/bin/env python3
"""
配置模块测试
"""

import pytest
import tempfile
import json
import os
from pathlib import Path

from rag.config import RAGConfig, APIConfig, DatabaseConfig


class TestRAGConfig:
    """RAG配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RAGConfig()
        
        assert config.api.openai_base_url == "http://10.172.10.103:11434/v1"
        assert config.database.milvus_uri == "http://10.172.10.100:19530"
        assert config.database.collection_name == "qwen3_embedding_rag"
        assert config.models.embedding.name == "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
        assert config.models.llm.name == "qwen3:4b"
    
    def test_config_validation(self):
        """测试配置验证"""
        config = RAGConfig()
        assert config.validate() is True
    
    def test_save_and_load_config(self):
        """测试配置保存和加载"""
        config = RAGConfig()
        
        # 修改一些配置
        config.set("api.openai_base_url", "http://test:8080")
        config.set("database.collection_name", "test_collection")
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_to_file(f.name)
            temp_file = f.name
        
        try:
            # 加载配置
            loaded_config = RAGConfig(temp_file)
            
            assert loaded_config.api.openai_base_url == "http://test:8080"
            assert loaded_config.database.collection_name == "test_collection"
            
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_environment_variables(self):
        """测试环境变量加载"""
        # 设置环境变量
        os.environ['OPENAI_BASE_URL'] = 'http://env-test:8080'
        os.environ['MILVUS_URI'] = 'http://env-milvus:19530'
        
        try:
            config = RAGConfig()
            
            assert config.api.openai_base_url == "http://env-test:8080"
            assert config.database.milvus_uri == "http://env-milvus:19530"
            
        finally:
            # 清理环境变量
            os.environ.pop('OPENAI_BASE_URL', None)
            os.environ.pop('MILVUS_URI', None)
    
    def test_invalid_config(self):
        """测试无效配置"""
        with pytest.raises(ValueError):
            # 创建无效的配置数据
            invalid_config_data = {
                "api": {
                    "openai_base_url": "invalid-url"  # 无效URL
                }
            }
            
            # 这里需要修改RAGConfig以支持直接传入配置数据
            # 暂时跳过这个测试
            pass


class TestAPIConfig:
    """API配置测试类"""
    
    def test_valid_url(self):
        """测试有效URL"""
        config = APIConfig(openai_base_url="https://api.example.com")
        assert config.openai_base_url == "https://api.example.com"
    
    def test_invalid_url(self):
        """测试无效URL"""
        with pytest.raises(ValueError):
            APIConfig(openai_base_url="invalid-url")
    
    def test_url_normalization(self):
        """测试URL规范化"""
        config = APIConfig(openai_base_url="https://api.example.com/")
        assert config.openai_base_url == "https://api.example.com"


class TestDatabaseConfig:
    """数据库配置测试类"""
    
    def test_valid_collection_name(self):
        """测试有效集合名称"""
        config = DatabaseConfig(collection_name="test_collection")
        assert config.collection_name == "test_collection"
    
    def test_invalid_collection_name(self):
        """测试无效集合名称"""
        with pytest.raises(ValueError):
            DatabaseConfig(collection_name="invalid collection name!")
    
    def test_collection_name_length(self):
        """测试集合名称长度限制"""
        long_name = "a" * 65  # 超过64字符
        with pytest.raises(ValueError):
            DatabaseConfig(collection_name=long_name) 