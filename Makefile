.PHONY: install test eval eval-orb up down ingest ask lint
install:        ## create venv + install deps
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"
test:           ## unit + integration tests (offline)
	pytest -q
eval:           ## retrieval ablation + e2e + quality gate (offline)
	python -m evals.run_eval && python -m evals.check_thresholds
eval-langsmith: ## push dataset + experiment to LangSmith (needs LANGSMITH_API_KEY)
	python -m evals.run_eval --langsmith
eval-orb:       ## Open RAG Bench text-only (needs OPENAI_API_KEY, real providers)
	python -m evals.open_ragbench
up:             ## full stack: api + worker + qdrant + redis
	docker compose -f deploy/docker-compose.yml up --build
down:
	docker compose -f deploy/docker-compose.yml down
ingest:
	python -m open_rag.cli ingest data/corpus
ask:
	python -m open_rag.cli ask "What does error NB-4021 mean?"
lint:
	ruff check src
