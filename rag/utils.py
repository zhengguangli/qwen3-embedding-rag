import os
import logging
import subprocess
import sys

def setup_logging(level: str = "INFO") -> logging.Logger:
    """设置日志配置"""
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置根日志级别
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'rag_system.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 降低第三方库的日志级别
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    return logging.getLogger('rag')

def check_dependencies():
    """检查依赖"""
    logger = logging.getLogger(__name__)
    try:
        import pkg_resources
        required = [
            ("pymilvus", None),
            ("openai", None),
            ("tqdm", None),
        ]
        
        for pkg, ver in required:
            try:
                pkg_resources.get_distribution(pkg)
            except Exception:
                logger.error(f"缺少依赖包: {pkg}，请先安装！")
                sys.exit(1)
        
        if not (sys.version_info.major == 3 and sys.version_info.minor >= 8):
            logger.warning(f"建议使用 Python 3.8+，当前为 {sys.version}")
            
    except Exception as e:
        logger.error(f"依赖检查失败: {str(e)}")
        sys.exit(1) 