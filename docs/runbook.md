# Runbook

| Symptôme | Diagnostic | Action |
|---|---|---|
| Réponses « je ne sais pas » en hausse | LangSmith : filtrer `status=no_evidence`; vérifier `n_candidates`; ingestion récente en échec ? | `GET /v1/jobs/{id}`, DLQ `rag:ingest:dlq`; baisser `min_relevance` seulement après eval |
| Latence p95 haute | traces par nœud (retrieve vs generate) | scale API/Qdrant ; réduire `top_k_*` ; activer cache ; modèle plus rapide |
| 429 du provider LLM | métriques `rag_requests_total` | backoff, fallback modèle, quotas par tenant |
| Qualité dégradée après changement | `make eval` + LangSmith comparer expériences | rollback : `swap_alias` vers l'ancienne collection |
| File d'ingestion qui grossit | longueur de `rag:ingest` | augmenter `maxReplicaCount` KEDA ; vérifier erreurs workers |
| Fuite suspectée inter-tenant | test `test_tenant_isolation`, audit filtres | bloquer l'API, corriger, rejouer les tests |

**Ré-indexation** : `python -m open_rag.cli reindex` → évaluer → alias déjà basculé (ajouter une étape de gate avant le swap en prod).
**Suppression d'un document** : `VectorIndex.delete_source(tenant, source)` (droit à l'effacement RGPD).
