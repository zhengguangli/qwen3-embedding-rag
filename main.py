#!/usr/bin/env python3
"""
RAG系统主入口

提供向后兼容的命令行接口，同时支持新的CLI功能
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

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.rag.cli import main
except ImportError as e:
    print(f"错误: 无法导入RAG CLI模块: {e}")
    print("请确保所有依赖已正确安装")
    sys.exit(1)

if __name__ == "__main__":
    main()
