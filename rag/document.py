from pathlib import Path
from glob import glob
from typing import List
import logging
from .config import RAGConfig

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器"""
    def __init__(self, config: RAGConfig):
        self.config = config
    def load_documents(self) -> List[str]:
        documents = []
        try:
            logger.info(f"加载文档: {self.config.data_path_glob}")
            for file_path in glob(self.config.data_path_glob, recursive=True):
                if Path(file_path).is_file():
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            documents.append(content)
            logger.info(f"加载了 {len(documents)} 个文档")
            return documents
        except Exception as e:
            logger.error(f"文档加载失败: {str(e)}")
            raise
    def split_documents(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            chunks.extend(self._split_text(doc))
        logger.info(f"分块后共 {len(chunks)} 段")
        return chunks
    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            if end < len(text):
                last_period = text.rfind(".", start, end)
                if last_period > start:
                    end = last_period + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(start + 1, end - self.config.chunk_overlap)
        return chunks 