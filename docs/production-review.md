# Revue « production à grande échelle » — open-rag & agentic-rag (v0.2)

Date : 2026-10-18. Première revue (v0.1) : 4/10 et 2,5/10. Ce document fait l'état après le durcissement.

## Verdict

| | v0.1 | **v0.2** | Ce qui sépare encore de 10/10 |
|---|---|---|---|
| **open-rag** | 4/10 | **8,5/10** | des preuves d'exploitation (charge, vrais modèles, SLO), l'async, une auth d'entreprise |
| **agentic-rag** | 2,5/10 | **8/10** | une éval du `LLMBrain` sur de vraies données, Postgres testé, OIDC, un classifieur d'injection |

**Pourquoi pas 10/10.** Une note de 10 ne s'obtient pas avec du code seul. Il faut aussi des preuves d'exploitation : tests de charge sur la vraie config, évals avec les vrais modèles sur un corpus métier, SLO tenus, exercices d'incident. Ces preuves ne peuvent pas être produites dans ce dépôt (pas de clé API, pas de cluster). Tout ce qui pouvait être corrigé ou durci par le code l'a été, et chaque correctif est couvert par un test.

**Vérifications effectuées :**
- open-rag : **30 tests**, verts en Qdrant embarqué **et contre un vrai serveur Qdrant 1.19.2**.
- agentic-rag : **41 tests**, stables sur 6 exécutions.
- Tests de mutation sur les correctifs critiques (versioning, rattrapage du reindex, double écriture, suppression des chunks périmés, reset de thread, authorizer SQL) : chaque mutation fait échouer un test.
- Quality gates vertes ; ruff propre ; venvs synchronisés sur les lockfiles.

---

## Les 3 bugs (corrigés, testés)

| Bug | Correctif | Test |
|---|---|---|
| Cache / BM25 périmés quand un document change sans changer de nombre de chunks | Version d'index = jeton aléatoire par tenant, tourné à **chaque** écriture, + epoch au swap d'alias (`store.index_version`). BM25 déplacé **dans Qdrant** (vecteurs sparse + IDF serveur), donc il ne peut plus être périmé. | `test_cache_invalidated_…`, `test_lexical_index_is_never_stale` |
| Réutiliser un `thread_id` héritait de l'état du run précédent | `RUN_RESET` + reducer `extend_or_reset` ; `Conflict` si le thread attend une approbation | `test_reusing_a_thread_starts_a_clean_run` |
| Contournement de l'allow-list SQL (`FROM sales, sqlite_master`) | sqlglot (toutes les tables et fonctions, SQL normalisé exécuté) + **authorizer SQLite** (la base refuse elle-même) + timeout | 7 cas de contournement + `test_database_enforces_policy_even_if_the_validator_is_bypassed` |

## open-rag — ce qui a changé

| Problème v0.1 | v0.2 |
|---|---|
| Upsert delete-puis-insert (le document disparaît, perdu si échec) | Embedding d'abord, upsert, **puis** suppression des points obsolètes ; document vidé = supprimé |
| File `BRPOP` (job perdu au crash), backoff bloquant | Redis Streams + consumer group, `XACK`, `XAUTOCLAIM`, tentatives comptées par livraison, retries différés (ZSET), DLQ, arrêt propre sur SIGTERM |
| Reindex : `extra` perdu, 10k points max, écritures perdues | Copie paginée de tous les tenants, `extra` conservé, **double écriture + rattrapage**, gate de validation, abandon propre, `rollback` |
| Pool de 2 threads partagé, 2 embeddings et ~6 allers-retours par requête | Hybride RRF **côté Qdrant en 1 requête**, embedding calculé une fois et réutilisé, alias résolu par Qdrant |
| Cache local O(n), propre à chaque réplica | Cache sémantique dans Qdrant (ANN, partagé, TTL, tolérant aux pannes) |
| Rate limit propre à chaque réplica, pas de budget | `Limits` via Redis (cluster-wide, fail-open) + budget de tokens journalier par tenant |
| Job visible par tous les tenants, upload illimité et non atomique | 404 cross-tenant, 413 au-delà de la taille max, écriture `.part` + `os.replace` |
| Historique non validé | Rôles contraints + filtre d'injection sur l'historique |
| Pas de fallback LLM | `structured_llm` : timeout, retries, modèle de secours (y compris sur erreur de parsing) ; 503 typé avec `Retry-After` |
| Streaming factice qui contournait cache et métriques | Chemin d'exécution unique `_execute` pour query et stream |
| 2 métriques | Latence par nœud, tokens par tenant et modèle, cache hit/miss, ingestion ; logs JSON + `X-Request-ID` propagé jusqu'à LangSmith |
| Pas de lockfile, k8s minimal | `requirements*.lock` (uv), Dockerfile multi-stage ; k8s : securityContext, rootfs en lecture seule, spread par zone, startupProbe, preStop, KEDA redis-streams, NetworkPolicy, PVC RWX |
| Gate CI déjà rouge (0,833 < 0,9) | Deux gates : non-régression hors ligne et qualité absolue avec vrais modèles (sur `main`) ; `pip-audit` |

## agentic-rag — ce qui a changé

| Problème v0.1 | v0.2 |
|---|---|
| Aucune auth, `user_id` venant du body | `Principal` issu de la clé API ; threads privés (registre + clé de checkpoint préfixée par le tenant) ; rôle `approver`, *four eyes* ; journal d'audit |
| Checkpointer en mémoire | `memory` / `sqlite` (testé : reprise après redémarrage) / `postgres` (pool) |
| Mémoire empoisonnable | Filtre d'injection, plafond de taille, PII masquées, nombre d'éléments borné, cloisonnement par tenant, injectée comme **données** |
| Embeddings forcés en factice, corpus ré-ingéré à chaque démarrage | Index partagé avec open-rag (`RAG_QDRANT_URL`), embeddings issus de l'environnement, tenant passé à chaque recherche |
| Grades séquentiels, aucun budget de tokens | `grade_many` en batch ; `TokenBudget` par run + budget journalier + rate limit ; `step_budget_exceeded` |
| Une panne LLM donnait un 500 | Défauts sûrs par décision (vérification qui échoue fermé), fallback de modèle ; testé avec des stubs |
| Web stub uniquement | `TavilyWebTool` (timeout, dégradation sans évidence), testé avec `MockTransport` |
| Dépendance `open-rag` résolvable sur PyPI (*dependency confusion*) | `[tool.uv.sources]` vers le chemin local + lockfile |
| Pas de déploiement ni de CI | Dockerfile, compose (2 réplicas + Postgres), CI avec gate et `pip-audit` |

---

## Ce qui reste (pour viser 9–10)

**À prouver en exploitation** (non faisable ici) :
1. Tests de charge (`benchmarks/locustfile.py`) sur la vraie config : p95/p99, saturation par pod, coût par requête. En déduire `RAG_API_THREADS`, le HPA et les SLO.
2. Évals avec les vrais modèles sur **200 à 500 questions métier annotées**, avec un LLM-juge calibré, et une éval par décision du `LLMBrain`. Les datasets actuels (14 et 10 questions) valident le comportement, pas la qualité.
3. Tests d'intégration sur un vrai Redis et un vrai Postgres en CI (services GitHub Actions). Aujourd'hui : fakeredis, et SQLite pour le contrat LangGraph.
4. Règles d'alerte Prometheus (taux de 5xx, p95, DLQ > 0, lag du stream, budget de tokens), dashboards, runbook d'incident et de restauration (snapshots Qdrant, PITR Postgres).

**À construire :**
5. Auth d'entreprise (OIDC/JWT, clés hachées avec rotation) à la place de clés en clair dans l'environnement ; secrets via un vault.
6. Pipeline async de bout en bout (plus de concurrence par pod) et vrai streaming de tokens.
7. Classifieur d'injection (Prompt Guard / Llama Guard) en plus des regex ; filtrage de sortie.
8. Uploads sur stockage objet (S3/GCS) plutôt qu'un volume RWX ; parsing riche (Docling) pour les PDF complexes.
9. Conformité : endpoints d'effacement par utilisateur ou tenant (RGPD), rétention des traces et des journaux d'audit.
10. Pour un entrepôt SQL réel : rôle en lecture seule, GRANT par table, `statement_timeout`, RLS par tenant (le SQLite de démo reproduit le principe).
