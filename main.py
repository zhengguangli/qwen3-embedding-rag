from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


import os

os.environ["OPENAI_API_KEY"] = "sk-2fd05fe02b9940d19e5a79de4cdf5fa4"
os.environ["OPENAI_BASE_URL"] = "http://10.172.10.103:11434/v1"


from glob import glob

text_lines = []
for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()
    text_lines += file_text.split("# ")


# Initialize OpenAI client for LLM generation
openai_client = OpenAI()
# Load Qwen3-Embedding-0.6B model for text embeddings
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# Load Qwen3-Reranker-0.6B model for reranking
reranker_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Reranker-0.6B", padding_side="left"
)
reranker_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
# Reranker configuration
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
max_reranker_length = 8192
prefix_tokens = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix_tokens = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def encode(text, is_query=False):
    """
    Generate text embeddings using Qwen3-Embedding-0.6B model.
    Args:
        text: Input text to embed
        is_query: Whether this is a query (True) or document (False)
    Returns:
        List of embedding values
    """
    if is_query:
        # For queries, use the "query" prompt for better retrieval performance
        embeddings = embedding_model.encode([text], prompt_name="query")
    else:
        # For documents, use default encoding
        embeddings = embedding_model.encode([text])
    return embeddings[0].tolist()


def format_instruction(instruction, query, doc):
    """Format instruction for reranker input"""
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


task = "Given a web search query, retrieve relevant passages that answer the query"

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def process_inputs(pairs):
    """Process inputs for reranker"""
    # Convert prefix and suffix to token IDs
    prefix_ids = reranker_tokenizer.encode(prefix_tokens, add_special_tokens=False)
    suffix_ids = reranker_tokenizer.encode(suffix_tokens, add_special_tokens=False)

    inputs = reranker_tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_reranker_length - len(prefix_ids) - len(suffix_ids),
    )

    # 正确处理 token IDs 的连接
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_ids + ele + suffix_ids

    inputs = reranker_tokenizer.pad(
        inputs, padding=True, return_tensors="pt", max_length=max_reranker_length
    )
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    """Compute relevance scores using reranker"""
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def rerank_documents(query, documents, task_instruction=None):
    """
    Rerank documents based on query relevance using Qwen3-Reranker
    Args:
        query: Search query
        documents: List of documents to rerank
        task_instruction: Task instruction for reranking
    Returns:
        List of (document, score) tuples sorted by relevance score
    """
    if task_instruction is None:
        task_instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    # Format inputs for reranker
    pairs = [format_instruction(task_instruction, query, doc) for doc in documents]
    # Process inputs and compute scores
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    # Combine documents with scores and sort by score (descending)
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores


test_embedding = encode("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])


from pymilvus import MilvusClient

# milvus_client = MilvusClient(uri="./milvus_demo.db")

milvus_client = MilvusClient(uri="http://10.172.10.100:19530")
collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)


milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

from tqdm import tqdm

data = []
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": encode(line), "text": line})
milvus_client.insert(collection_name=collection_name, data=data)


question = "How is data stored in milvus?"

# Step 1: Initial retrieval with larger candidate set
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        encode(question, is_query=True)
    ],  # Use the `encode` function with query prompt to convert the question to an embedding vector
    limit=10,  # Return top 10 candidates for reranking
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)
# Step 2: Extract candidate documents for reranking
candidate_docs = [res["entity"]["text"] for res in search_res[0]]
# Step 3: Rerank documents using Qwen3-Reranker
print("Reranking documents...")
reranked_docs = rerank_documents(question, candidate_docs)
# Step 4: Select top 3 reranked documents
top_reranked_docs = reranked_docs[:3]
print(f"Selected top {len(top_reranked_docs)} documents after reranking")

import json

# Display reranked results with reranker scores
reranked_lines_with_scores = [(doc, score) for doc, score in top_reranked_docs]
print("Reranked results:")
print(json.dumps(reranked_lines_with_scores, indent=4))
# Also show original embedding-based results for comparison
print("\n" + "=" * 80)
print("Original embedding-based results (top 3):")
original_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0][:3]
]
print(json.dumps(original_lines_with_distances, indent=4))


context = "\n".join(
    [line_with_distance[0] for line_with_distance in reranked_lines_with_scores]
)


SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""


response = openai_client.chat.completions.create(
    model="deepseek-r1",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)
