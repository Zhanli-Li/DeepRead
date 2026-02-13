export OPENROUTER_API_KEY=""
export OPENROUTER_MODEL="deepseek/deepseek-v3.2"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export EMBED_API_KEY=""
export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export EMBED_BASE_URL="https://api.siliconflow.cn/v1"
export RERANK_API_KEY=""
export RERANK_MODEL="Qwen/Qwen3-Reranker-8B"
export RERANK_BASE_URL="https://api.siliconflow.cn/v1"
export SILICONFLOW_API_KEY=""
python DeepRead/Code/DeepRead.py \
   --doc 'DeepRead/Demo/金山办公2023年报/11724-金山办公：金山办公2023年年度报告_corpus.json' \
   --question "金山办公2023年营收增长多少？" \
   --neighbor-window 0,0 \
   --disable-regex \
   --disable-bm25 \
   --enable-semantic \
   --log test.jsonl
