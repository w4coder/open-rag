#!/usr/bin/env bash
# Full Open RAG Bench text-only run: every text question (1914), all 600 hard negatives,
# 50 held-out papers + papers absent from the index for the refusal test.
# Needs a Qdrant server (RAG_QDRANT_URL) and real providers in .env.
set -u
cd "$(dirname "$0")/.."
export RAG_QDRANT_URL="${RAG_QDRANT_URL:-http://127.0.0.1:6333}"
ARGS="--queries 1914 --negatives 600 --holdout-papers 50 --absent 100 --workers 6"
python=".venv/Scripts/python"; [ -x "$python" ] || python=".venv/bin/python"
"$python" -u -m evals.open_ragbench $ARGS --reranker llm --tag full_llm > evals/results/orb_full_llm.log 2>&1
echo "DONE full_llm exit=$?"
"$python" -u -m evals.open_ragbench $ARGS --reranker none --tag full_none > evals/results/orb_full_none.log 2>&1
echo "DONE full_none exit=$?"
