#!/usr/bin/env python3
"""
RAG系统配置管理工具
提供配置验证、转换、生成和管理功能
"""

import argparse
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.config import RAGConfig, RAGConfigModel

def load_config(config_file: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.yaml':
            return yaml.safe_load(f)
        else:
            return json.load(f)

def save_config(config_data: Dict[str, Any], output_file: str) -> None:
    """保存配置文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_path.suffix.lower() == '.yaml':
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        else:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

def validate_config(config_data: Dict[str, Any]) -> bool:
    """验证配置"""
    try:
        # 移除注释字段
        clean_data = remove_comments(config_data)
        RAGConfigModel.model_validate(clean_data)
        print("✅ 配置验证通过")
        return True
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False

def remove_comments(data: Any) -> Any:
    """移除配置中的注释字段"""
    if isinstance(data, dict):
        return {k: remove_comments(v) for k, v in data.items() 
               if not k.startswith('_')}
    elif isinstance(data, list):
        return [remove_comments(item) for item in data]
    else:
        return data

def convert_config(input_file: str, output_file: str) -> None:
    """转换配置文件格式"""
    try:
        config_data = load_config(input_file)
        clean_data = remove_comments(config_data)
        
        if validate_config(clean_data):
            save_config(clean_data, output_file)
            print(f"✅ 配置已转换并保存到: {output_file}")
        else:
            print("❌ 配置验证失败，转换中止")
    except Exception as e:
        print(f"❌ 转换失败: {e}")

def generate_template(output_file: str, format_type: str = "json") -> None:
    """生成配置模板"""
    try:
        # 创建默认配置
        config = RAGConfig()
        config_data = config.to_dict()
        
        # 添加注释
        config_data["_comment"] = "RAG系统配置文件 - 基于Qwen3 Embedding和Milvus向量数据库"
        config_data["_version"] = "1.0.0"
        config_data["_description"] = "检索增强生成系统配置"
        
        # 为每个部分添加注释
        for section in config_data:
            if isinstance(config_data[section], dict):
                config_data[section]["_comment"] = f"{section}相关配置"
        
        # 确定输出格式
        if format_type.lower() == "yaml":
            if not output_file.endswith(('.yaml', '.yml')):
                output_file += '.yaml'
        else:
            if not output_file.endswith('.json'):
                output_file += '.json'
        
        save_config(config_data, output_file)
        print(f"✅ 配置模板已生成: {output_file}")
        
    except Exception as e:
        print(f"❌ 生成模板失败: {e}")

def show_config_info(config_file: str) -> None:
    """显示配置信息"""
    try:
        config_data = load_config(config_file)
        clean_data = remove_comments(config_data)
        
        print(f"📋 配置文件: {config_file}")
        print("=" * 50)
        
        # 显示基本信息
        version = clean_data.get("_version", "未知")
        description = clean_data.get("_description", "无描述")
        print(f"版本: {version}")
        print(f"描述: {description}")
        print()
        
        # 显示关键配置
        key_configs = [
            ("API配置", "api"),
            ("数据库配置", "database"),
            ("模型配置", "models"),
            ("数据处理", "data"),
            ("搜索配置", "search"),
            ("性能配置", "performance")
        ]
        
        for title, section in key_configs:
            if section in clean_data:
                print(f"🔧 {title}:")
                section_data = clean_data[section]
                for key, value in section_data.items():
                    if not key.startswith('_'):
                        print(f"  {key}: {value}")
                print()
        
        # 验证配置
        if validate_config(clean_data):
            print("✅ 配置状态: 有效")
        else:
            print("❌ 配置状态: 无效")
            
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")

def create_env_template(output_file: str) -> None:
    """创建环境变量模板"""
    env_vars = {
        "OPENAI_API_KEY": "OpenAI API密钥",
        "OPENAI_BASE_URL": "OpenAI兼容API基础URL",
        "MILVUS_URI": "Milvus服务URI",
        "MILVUS_COLLECTION": "Milvus集合名称",
        "LOG_LEVEL": "日志级别"
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RAG系统环境变量配置\n")
            f.write("# 复制此文件为.env并填入实际值\n\n")
            
            for var, desc in env_vars.items():
                f.write(f"# {desc}\n")
                f.write(f"{var}=\n\n")
        
        print(f"✅ 环境变量模板已生成: {output_file}")
        
    except Exception as e:
        print(f"❌ 生成环境变量模板失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统配置管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 验证配置命令
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('config_file', help='配置文件路径')
    
    # 转换配置命令
    convert_parser = subparsers.add_parser('convert', help='转换配置文件格式')
    convert_parser.add_argument('input_file', help='输入配置文件')
    convert_parser.add_argument('output_file', help='输出配置文件')
    
    # 生成模板命令
    template_parser = subparsers.add_parser('template', help='生成配置模板')
    template_parser.add_argument('output_file', help='输出文件路径')
    template_parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='输出格式')
    
    # 显示配置信息命令
    info_parser = subparsers.add_parser('info', help='显示配置信息')
    info_parser.add_argument('config_file', help='配置文件路径')
    
    # 生成环境变量模板命令
    env_parser = subparsers.add_parser('env', help='生成环境变量模板')
    env_parser.add_argument('output_file', default='.env.template', help='输出文件路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'validate':
            config_data = load_config(args.config_file)
            clean_data = remove_comments(config_data)
            validate_config(clean_data)
            
        elif args.command == 'convert':
            convert_config(args.input_file, args.output_file)
            
        elif args.command == 'template':
            generate_template(args.output_file, args.format)
            
        elif args.command == 'info':
            show_config_info(args.config_file)
            
        elif args.command == 'env':
            create_env_template(args.output_file)
            
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 