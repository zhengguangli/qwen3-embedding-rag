# 最新｜用Qwen3 Embedding+Milvus，搭建最强企业知识库

## 参考链接
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
https://mp.weixin.qq.com/s/abz-7JcAjcwulgU1MUW5_A


## 安装环境

```bash
uv pip install --upgrade pymilvus openai requests tqdm sentence-transformers transformers "numpy<2" "httpx[socks]"

wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip \
    && unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
```

## Or 创建新的代码仓库
```
echo "# qwen3-embedding-rag" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:zhengguangli/qwen3-embedding-rag.git
git push -u origin main
```


## Or 推送代码到远程仓库
```bash
git remote add origin git@github.com:zhengguangli/qwen3-embedding-rag.git
git branch -M main
git push -u origin main
```