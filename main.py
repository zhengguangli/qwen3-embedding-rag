import os
import sys
import json
import logging
import time
from glob import glob
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
from functools import lru_cache
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import requests
import ollama

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查关键依赖和环境版本，给出友好提示"""
    import pkg_resources
    required = [
        ("torch", "2.2.2"),
        ("numpy", None),
        ("pymilvus", None),
        ("openai", None),
        ("sentence-transformers", None),
        ("transformers", None),
        ("tqdm", None),
    ]
    for pkg, ver in required:
        try:
            dist = pkg_resources.get_distribution(pkg)
            if ver and dist.version != ver:
                logger.warning(f"依赖 {pkg} 版本为 {dist.version}，建议为 {ver}")
        except Exception:
            logger.error(f"缺少依赖包: {pkg}，请先安装！")
            sys.exit(1)
    if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
        logger.warning(f"建议使用 Python 3.12，当前为 {sys.version}")


class RAGConfig:
    """RAG 管道配置类"""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "http://10.172.10.103:11434/v1")
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://10.172.10.100:19530")
    DATA_PATH_GLOB: str = "milvus_docs/en/faq/*.md"
    COLLECTION_NAME: str = "my_rag_collection"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    RERANKER_MODEL: str = "Qwen/Qwen3-Reranker-0.6B"
    LLM_MODEL: str = "qwen3:4b"
    EMBEDDING_DIM: int = 1024
    METRIC_TYPE: str = "IP"
    CONSISTENCY_LEVEL: str = "Strong"
    SEARCH_LIMIT: int = 10
    RERANK_TOP_K: int = 3
    MAX_RERANKER_LENGTH: int = 8192
    RERANKER_PROMPT_INSTRUCTION: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CACHE_SIZE: int = 1000
    BATCH_SIZE: int = 32
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def print_config(self):
        logger.info(f"设备: {self.DEVICE}")
        logger.info(f"嵌入模型: {self.EMBEDDING_MODEL}")
        logger.info(f"重排序模型: {self.RERANKER_MODEL}")
        logger.info(f"LLM模型: {self.LLM_MODEL}")
        logger.info(f"Milvus URI: {self.MILVUS_URI}")
        logger.info(f"数据路径: {self.DATA_PATH_GLOB}")
        logger.info(f"分块大小: {self.CHUNK_SIZE}, 重叠: {self.CHUNK_OVERLAP}")
        logger.info(f"批处理: {self.BATCH_SIZE}, 缓存: {self.CACHE_SIZE}")


class RAGPipeline:
    """RAG 管道主类，负责数据加载、检索、重排、生成答案。"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self._setup_clients()
        self._load_models()
        self.prefix_tokens = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix_tokens = ("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self._init_cache()

    def _init_cache(self):
        self.encode = lru_cache(maxsize=self.config.CACHE_SIZE)(self.encode)
        self._format_reranker_input = lru_cache(maxsize=self.config.CACHE_SIZE)(self._format_reranker_input)

    def _setup_clients(self):
        try:
            os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY
            os.environ["OPENAI_BASE_URL"] = self.config.OPENAI_BASE_URL
            self.openai_client = OpenAI()
            self.milvus_client = MilvusClient(uri=self.config.MILVUS_URI)
            logger.info("API 客户端初始化成功")
        except Exception as e:
            logger.error(f"API 客户端初始化失败: {str(e)}")
            raise

    def _load_models(self):
        try:
            logger.info("加载模型...")
            t0 = time.time()
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.embedding_model.to(self.config.DEVICE)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.config.RERANKER_MODEL, padding_side="left", trust_remote_code=True
            )
            self.reranker_model = (
                AutoModelForCausalLM.from_pretrained(
                    self.config.RERANKER_MODEL, trust_remote_code=True
                )
                .eval()
                .to(self.config.DEVICE)
            )
            self.token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
            logger.info(f"模型加载完成，耗时 {time.time()-t0:.2f}s")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def _load_and_split_docs(self) -> List[str]:
        """加载文档并智能分块"""
        text_chunks = []
        try:
            logger.info(f"加载文档: {self.config.DATA_PATH_GLOB}")
            for file_path in glob(self.config.DATA_PATH_GLOB, recursive=True):
                with open(file_path, "r", encoding="utf-8") as f:
                    file_text = f.read()
                start = 0
                while start < len(file_text):
                    end = start + self.config.CHUNK_SIZE
                    chunk = file_text[start:end]
                    if end < len(file_text):
                        last_period = chunk.rfind(".")
                        if last_period != -1:
                            end = start + last_period + 1
                            chunk = file_text[start:end]
                    if chunk.strip():
                        text_chunks.append(chunk)
                    start = end - self.config.CHUNK_OVERLAP
            logger.info(f"分块后共 {len(text_chunks)} 段")
            return text_chunks
        except Exception as e:
            logger.error(f"文档加载失败: {str(e)}")
            raise

    def setup_collection(self, force_recreate: bool = False) -> None:
        """初始化 Milvus 集合，必要时重建并导入数据"""
        try:
            collection_exists = self.milvus_client.has_collection(self.config.COLLECTION_NAME)
            if collection_exists and force_recreate:
                logger.info(f"删除已存在集合: {self.config.COLLECTION_NAME}")
                self.milvus_client.drop_collection(self.config.COLLECTION_NAME)
                collection_exists = False
            if not collection_exists:
                logger.info("集合不存在，创建并导入数据...")
                self.milvus_client.create_collection(
                    collection_name=self.config.COLLECTION_NAME,
                    dimension=self.config.EMBEDDING_DIM,
                    metric_type=self.config.METRIC_TYPE,
                    consistency_level=self.config.CONSISTENCY_LEVEL,
                )
                docs = self._load_and_split_docs()
                logger.info(f"共加载 {len(docs)} 段文档，开始编码...")
                t0 = time.time()
                embeddings = []
                for i in tqdm(range(0, len(docs), self.config.BATCH_SIZE), desc="编码文档"):
                    batch = docs[i:i+self.config.BATCH_SIZE]
                    batch_emb = self.embedding_model.encode(batch, show_progress_bar=False)
                    embeddings.extend(batch_emb)
                logger.info(f"编码完成，耗时 {time.time()-t0:.2f}s")
                data = [
                    {"id": i, "vector": emb.tolist(), "text": doc}
                    for i, (doc, emb) in enumerate(zip(docs, embeddings))
                ]
                self.milvus_client.insert(
                    collection_name=self.config.COLLECTION_NAME, data=data
                )
                logger.info("数据导入完成")
            else:
                logger.info("集合已存在，跳过数据导入")
        except Exception as e:
            logger.error(f"集合初始化失败: {str(e)}")
            raise

    def encode(self, text: str, is_query: bool = False) -> List[float]:
        prompt_name = "query" if is_query else None
        embedding = self.embedding_model.encode([text], prompt_name=prompt_name)
        return embedding[0].tolist()

    def retrieve(self, question: str) -> List[str]:
        try:
            search_res = self.milvus_client.search(
                collection_name=self.config.COLLECTION_NAME,
                data=[self.encode(question, is_query=True)],
                limit=self.config.SEARCH_LIMIT,
                output_fields=["text"],
            )
            return [res["entity"]["text"] for res in search_res[0]]
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        pairs = [self._format_reranker_input(query, doc) for doc in documents]
        inputs = self._process_reranker_inputs(pairs)
        scores = self._compute_reranker_logits(inputs)
        doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return doc_scores

    def _format_reranker_input(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.config.RERANKER_PROMPT_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _process_reranker_inputs(self, pairs: List[str]):
        full_texts = [f"{self.prefix_tokens}{p}{self.suffix_tokens}" for p in pairs]
        inputs = self.reranker_tokenizer(
            full_texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.config.MAX_RERANKER_LENGTH,
        ).to(self.reranker_model.device)
        return inputs

    @torch.no_grad()
    def _compute_reranker_logits(self, inputs) -> List[float]:
        logits = self.reranker_model(**inputs).logits[:, -1, :]
        true_scores = logits[:, self.token_true_id]
        false_scores = logits[:, self.token_false_id]
        scores = torch.stack([false_scores, true_scores], dim=1)
        log_softmax_scores = torch.nn.functional.log_softmax(scores, dim=1)
        return log_softmax_scores[:, 1].exp().tolist()

    def generate_answer(self, question: str, context: str) -> str:
        """使用 Ollama Python 客户端本地推理生成答案"""
        try:
            response = ollama.chat(
                model=self.config.LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': '你是AI助手'},
                    {'role': 'user', 'content': f'请基于以下上下文信息回答问题：\n\n上下文：\n{context}\n\n问题：{question}\n\n请提供准确、详细的回答。'}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"生成答案失败: {str(e)}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

    def run(self, question: str) -> str:
        try:
            logger.info(f"开始处理问题: '{question}'")
            t0 = time.time()
            candidate_docs = self.retrieve(question)
            logger.info(f"检索到 {len(candidate_docs)} 个候选文档")
            reranked_docs_with_scores = self.rerank(question, candidate_docs)
            top_docs_with_scores = reranked_docs_with_scores[: self.config.RERANK_TOP_K]
            logger.info(f"重排后选取前 {len(top_docs_with_scores)} 个文档")
            context = "\n".join([doc for doc, score in top_docs_with_scores])
            answer = self.generate_answer(question, context)
            logger.info(f"处理完成，总耗时 {time.time()-t0:.2f}s")
            return answer
        except Exception as e:
            logger.error(f"RAG 管道执行失败: {str(e)}")
            raise


def main():
    check_dependencies()
    parser = argparse.ArgumentParser(description="Qwen3-Embedding-RAG 管道")
    parser.add_argument("--question", type=str, help="要检索和回答的问题", required=False)
    parser.add_argument("--force-recreate", action="store_true", help="强制重建 Milvus 集合")
    args = parser.parse_args()
    config = RAGConfig()
    config.print_config()
    pipeline = RAGPipeline(config)
    pipeline.setup_collection(force_recreate=args.force_recreate)
    default_question = "Milvus 的数据是如何存储的？"
    question = args.question if args.question else default_question
    answer = pipeline.run(question)
    print("\n" + "=" * 30 + " 最终答案 " + "=" * 30)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}")
        sys.exit(1)
