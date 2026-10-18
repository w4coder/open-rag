# Fiche de rappel : ce qu'il faut savoir pour concevoir un RAG moderne

Chaque ligne renvoie au code qui l'illustre.

## 1. Ingestion
| À retenir | Pourquoi | Code |
|---|---|---|
| Le parsing plafonne la qualité globale | tables/titres perdus = chunks inutilisables | `ingestion/loaders.py` |
| Chunker selon la **structure** (titres) puis la taille | un chunk = une idée, avec son chemin de section | `ingestion/chunking.py` |
| **Parent-child** : chercher petit, lire grand | embeddings précis + contexte suffisant pour le LLM | `Chunk.parent_text` |
| Ids déterministes + upsert « replace » | ingestion idempotente, retries sûrs | `make_chunk_id`, `VectorIndex.upsert` |
| **Contextual retrieval** | le chunk « le chiffre a augmenté de 3 % » doit savoir de quoi il parle | `ingestion/pipeline.py::enrich_chunks` |
| Métadonnées (tenant, source, ACL, version) | filtrage, citations, suppression, audit | `Chunk.payload` |

## 2. Indexation
- **Alias de collection + blue/green** : changer de modèle d'embedding sans coupure → `indexing/store.py`, `indexing/reindex.py`.
- **Multi-tenant** par filtre de payload indexé (jamais de fuite inter-tenant) → `_tenant_filter`, test `test_tenant_isolation`.
- Le modèle d'embedding est un **choix versionné** : changer = tout ré-indexer.

## 3. Retrieval
- Dense seul rate les termes exacts (codes d'erreur, noms) ; BM25 seul rate la sémantique → **hybride** : `retrieval/hybrid.py`.
- **RRF** fusionne par rang, sans normaliser les scores.
- **Rerank cross-encoder** sur 20-40 candidats : meilleur gain qualité/effort → `retrieval/rerank.py`.
- **Query rewriting** pour les questions de suivi (« et pour l'Enterprise ? ») → `graph.py::rewrite`.
- Seuil de pertinence : mieux vaut « je ne sais pas » qu'une hallucination (`min_relevance`).

## 4. Génération
- Contexte marqué **non fiable** (prompt injection indirecte) → `generation/prompts.py`.
- **Sortie structurée** (`Answer`) : citations vérifiables par code → `models.py`.
- Refus explicite si contexte insuffisant (`insufficient_context`).
- **Vérification d'ancrage** : toute citation doit avoir été récupérée → `guardrails.check_grounding`.

## 5. Évaluation (le cœur)
- Séparer **retrieval** (hit@k, MRR) et **génération** (correction, fidélité, refus) pour localiser les pannes → `evals/metrics.py`.
- Labels = **snippets de texte**, pas des ids de chunks : le dataset survit aux changements de chunking.
- Une expérience = une variable qui change, même dataset → `evals/run_eval.py` (+ LangSmith).
- **Gate CI** : un seuil minimal bloque les régressions → `evals/check_thresholds.py`.
- En production : traces LangSmith taguées (tenant, version d'index), feedback utilisateur → dataset.

## 6. Production
| Sujet | Mise en œuvre |
|---|---|
| API stateless, auth → tenant, rate limit | `api.py` |
| Streaming SSE (latence perçue) | `/v1/query/stream` |
| Ingestion asynchrone, retries, DLQ | `queue.py` |
| Cache sémantique **scopé** tenant + version d'index | `cache.py` |
| Probes live/ready, métriques Prometheus | `/healthz`, `/readyz`, `/metrics` |
| Autoscaling API (CPU) vs workers (longueur de file) | `deploy/k8s/rag.yaml` |
| Observabilité LLM | LangSmith (`service.py` : tags + metadata, PII masquée) |

## 7. Pièges classiques
1. Évaluer « à l'œil » → régression silencieuse.
2. Cache sans clé de version d'index → réponses périmées.
3. Chunks trop gros (bruit) ou trop petits (contexte perdu).
4. Oublier le filtre tenant sur **une** des deux branches (dense/sparse).
5. Ne pas borner timeouts/retries LLM → cascade de pannes.
6. Embeddings changés sans ré-indexer → recherche incohérente.
