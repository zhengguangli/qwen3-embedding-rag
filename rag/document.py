#!/usr/bin/env python3
"""
文档处理模块

提供智能的文档加载、预处理和分块功能，包括：
- 多格式文件支持
- 智能分块策略
- 元数据管理
- 性能优化
"""

import re
import hashlib
from pathlib import Path
from glob import glob
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import mimetypes

from .config import RAGConfig

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """文档块数据类"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    chunk_index: int
    start_pos: int
    end_pos: int

@dataclass
class DocumentMetadata:
    """文档元数据"""
    file_path: str
    file_size: int
    file_type: str
    encoding: str
    last_modified: float
    content_hash: str
    word_count: int
    language: Optional[str] = None

class DocumentProcessor:
    """智能文档处理器"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 支持的文件类型
        self.supported_extensions = set(self.config.data.file_extensions)
        
        # 分块策略映射
        self.chunk_strategies = {
            "fixed": self._split_fixed,
            "semantic": self._split_semantic,
            "sentence": self._split_sentence
        }
        
        # 线程池用于并发处理
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def load_documents(self) -> List[Tuple[str, DocumentMetadata]]:
        """加载文档并返回内容和元数据"""
        documents = []
        
        try:
            self.logger.info(f"开始加载文档: {self.config.data.data_path_glob}")
            
            # 获取所有匹配的文件
            file_paths = glob(self.config.data.data_path_glob, recursive=True)
            self.logger.info(f"找到 {len(file_paths)} 个文件")
            
            # 并发加载文档
            futures = []
            for file_path in file_paths:
                if Path(file_path).is_file() and self._is_supported_file(file_path):
                    future = self._executor.submit(self._load_single_document, file_path)
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        content, metadata = result
                        documents.append((content, metadata))
                except Exception as e:
                    self.logger.error(f"加载文档失败: {str(e)}")
            
            self.logger.info(f"成功加载 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            self.logger.error(f"文档加载失败: {str(e)}")
            raise
    
    def _load_single_document(self, file_path: str) -> Optional[Tuple[str, DocumentMetadata]]:
        """加载单个文档"""
        try:
            path = Path(file_path)
            
            # 检查文件大小
            file_size = path.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB限制
                self.logger.warning(f"文件过大，跳过: {file_path}")
                return None
            
            # 检测文件类型和编码
            file_type = self._detect_file_type(file_path)
            encoding = self._detect_encoding(file_path)
            
            # 读取内容
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read().strip()
            
            if not content:
                return None
            
            # 计算内容哈希
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # 创建元数据
            metadata = DocumentMetadata(
                file_path=str(file_path),
                file_size=file_size,
                file_type=file_type,
                encoding=encoding,
                last_modified=path.stat().st_mtime,
                content_hash=content_hash,
                word_count=len(content.split()),
                language=self._detect_language(content)
            )
            
            return content, metadata
            
        except Exception as e:
            self.logger.error(f"加载文档失败 {file_path}: {str(e)}")
            return None
    
    def split_documents(self, documents: List[Tuple[str, DocumentMetadata]]) -> List[DocumentChunk]:
        """智能分割文档"""
        all_chunks = []
        
        try:
            self.logger.info(f"开始分割 {len(documents)} 个文档")
            
            # 获取分块策略
            strategy = self.chunk_strategies.get(
                self.config.data.chunk_strategy, 
                self._split_fixed
            )
            
            # 并发处理文档分块
            futures = []
            for i, (content, metadata) in enumerate(documents):
                future = self._executor.submit(
                    self._split_single_document, 
                    content, metadata, i, strategy
                )
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    self.logger.error(f"文档分块失败: {str(e)}")
            
            self.logger.info(f"分块完成，共生成 {len(all_chunks)} 个块")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"文档分块失败: {str(e)}")
            raise
    
    def _split_single_document(
        self, 
        content: str, 
        metadata: DocumentMetadata, 
        doc_index: int,
        strategy
    ) -> List[DocumentChunk]:
        """分割单个文档"""
        chunks = []
        
        # 预处理内容
        processed_content = self._preprocess_content(content)
        
        # 使用指定策略分块
        text_chunks = strategy(processed_content)
        
        # 创建文档块对象
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
            
            # 计算位置信息
            start_pos = processed_content.find(chunk_text)
            end_pos = start_pos + len(chunk_text)
            
            # 生成块ID
            chunk_id = f"{metadata.content_hash}_{doc_index}_{i}"
            
            # 创建块元数据
            chunk_metadata = {
                "source_file": metadata.file_path,
                "file_type": metadata.file_type,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "word_count": len(chunk_text.split()),
                "language": metadata.language,
                "original_metadata": metadata.__dict__
            }
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                source_file=metadata.file_path,
                chunk_index=i,
                start_pos=start_pos,
                end_pos=end_pos
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _split_fixed(self, text: str) -> List[str]:
        """固定长度分块"""
        chunks = []
        chunk_size = self.config.data.chunk_size
        overlap = self.config.data.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句号处分割
            if end < len(text):
                last_period = text.rfind(".", start, end)
                if last_period > start and (end - last_period) < 100:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def _split_semantic(self, text: str) -> List[str]:
        """语义分块（基于段落和章节）"""
        # 分割为段落
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果当前块加上新段落不超过限制，则合并
            if len(current_chunk) + len(paragraph) <= self.config.data.chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落就超过限制，需要进一步分割
                if len(paragraph) > self.config.data.chunk_size:
                    sub_chunks = self._split_fixed(paragraph)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = paragraph
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_sentence(self, text: str) -> List[str]:
        """句子级分块"""
        # 分割句子（支持多种标点符号）
        sentences = re.split(r'[.!?。！？]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 如果当前块加上新句子不超过限制，则合并
            if len(current_chunk) + len(sentence) <= self.config.data.chunk_size:
                current_chunk += ". " + sentence if current_chunk else sentence
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk + ".")
                
                # 如果单个句子就超过限制，需要进一步分割
                if len(sentence) > self.config.data.chunk_size:
                    sub_chunks = self._split_fixed(sentence)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = sentence
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk + ".")
        
        return chunks
    
    def _preprocess_content(self, content: str) -> str:
        """预处理内容"""
        # 去除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        
        # 去除特殊字符（保留中文、英文、数字和基本标点）
        content = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\[\]{}"\'-]', '', content)
        
        # 截断到最大长度
        max_length = self.config.data.chunk_size * 10  # 允许的最大文档长度
        if len(content) > max_length:
            content = content[:max_length]
            self.logger.warning(f"文档被截断到 {max_length} 字符")
        
        return content.strip()
    
    def _is_supported_file(self, file_path: str) -> bool:
        """检查是否为支持的文件类型"""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def _detect_file_type(self, file_path: str) -> str:
        """检测文件类型"""
        path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or path.suffix.lower()
    
    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        import chardet
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _detect_language(self, text: str) -> Optional[str]:
        """检测文本语言"""
        try:
            import langdetect
            return langdetect.detect(text)
        except Exception:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            "supported_extensions": list(self.supported_extensions),
            "chunk_strategy": self.config.data.chunk_strategy,
            "chunk_size": self.config.data.chunk_size,
            "chunk_overlap": self.config.data.chunk_overlap
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            self.logger.info("文档处理器资源清理完成")
        except Exception as e:
            self.logger.error(f"文档处理器资源清理失败: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 