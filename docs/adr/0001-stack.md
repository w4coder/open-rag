# ADR 0001 — Stack : LangChain + LangGraph + LangSmith + Qdrant + FastAPI

**Statut** : accepté

- **LangChain** : abstractions (prompts, sorties structurées, embeddings, text splitters). On n'utilise *pas* les chaînes « tout-en-un » : composants seulement, pour garder le contrôle.
- **LangGraph** : pipeline en graphe d'état → étapes traçables/testables, branches explicites, base de l'agentique.
- **LangSmith** : tracing, datasets, expériences, monitoring ; indispensable pour itérer sur la qualité.
- **Qdrant** : filtrage payload performant, sharding/replication natifs, mode embarqué pour dev/tests, alias de collections. Alternatives : pgvector (simple si vous avez déjà Postgres, limites à grande échelle), OpenSearch (hybride natif), Weaviate/Milvus.
- **FastAPI** : async, OpenAPI, écosystème.
- **Provider « fake »** : exécution offline déterministe (CI, démos) ; coût : ne mesure pas la qualité sémantique réelle (cf. docs/evaluation.md).

Conséquence : le coût d'un changement de vector DB est limité à `indexing/store.py` et `retrieval/sparse.py`.
