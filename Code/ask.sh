export OPENROUTER_API_KEY=""
export OPENAI_MODEL="deepseek/deepseek-v3.2"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export EMBED_API_KEY=""
export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export EMBED_BASE_URL="https://api.siliconflow.cn/v1"
export SILICONFLOW_API_KEY=""
python DeepRead/Code/DeepRead.py \
   --doc 'DeepRead/Code/斗破苍穹_corpus.json' \
   --question "萧族被灭后，萧炎找到了当时的先祖吗，有几位" \
   --neighbor-window 0,0 \
   --disable-regex \
   --disable-bm25 \
   --enable-semantic \
   --log test.jsonl
