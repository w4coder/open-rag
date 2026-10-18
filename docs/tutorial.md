# Tutoriel — lire `open-rag` fichier par fichier

À lire **à côté du code**. Chaque section = un fichier : son rôle, ses fonctions/classes, comment elles s'utilisent.

## 0. Vue d'ensemble

```
ÉCRITURE  fichier ─► loaders ─► chunking ─► (enrichissement LLM) ─► store.upsert ─► Qdrant (dense + BM25)
          API /v1/documents ─► Redis Stream ─► worker(s) ─► ingest_path            └─► version tenant ↻

LECTURE   API /v1/query ─► auth + rate limit ─► RagService._execute
          ─► embed(question) 1× ─► cache sémantique (Qdrant, clé = tenant + version d'index)
          ─► graph: guard ─► rewrite ─► retrieve (hybride RRF côté Qdrant) ─► rerank ─► generate (+grounding)
```

**Ordre de lecture** : `config` → `models` → `providers` → `ingestion/*` → `retrieval/sparse` → `indexing/store` → `retrieval/hybrid` + `rerank` → `generation/*` → `graph` → `observability` + `limits` → `cache` → `service` → `queue` → `api` → `indexing/reindex` → `evals`.

---

## 1. Fondations

### `config.py` — configuration typée
`Settings` regroupe toutes les variables `RAG_*` (lues depuis `.env`). Groupes : providers (modèle, **fallback**, timeout, retries), vector store, chunking, retrieval, cache, infra (Redis, clés API, threads, upload, rate limit, budget tokens, logs JSON).
- `api_key_map()` : `"key:tenant,…"` → `{key: tenant}`.
- `get_settings()` : singleton. En test : `get_settings.cache_clear()` après `monkeypatch.setenv`.

### `models.py`
| Modèle | Rôle |
|---|---|
| `Answer` | Sortie structurée du LLM (`answer`, `cited_ids`, `insufficient_context`) : les citations sont vérifiables par code. |
| `Message` | Tour d'historique : `role` limité à `user`/`assistant` (un client ne peut pas injecter `system`), 4000 caractères max. |
| `QueryRequest` | Question (≤ 8000 caractères) + historique (≤ 50 messages). |
| `QueryResponse`, `Citation`, `IngestResponse` | Sorties de l'API. |

### `providers.py` — fabriques de modèles
| Fonction | Rôle |
|---|---|
| `tokenize`, `token_hash` | Tokeniseur partagé (BM25, embedding factice, générateur extractif) ; hash stable entre process. |
| `HashingEmbeddings` | Embedding **factice** déterministe (tests/CI). Pas sémantique. |
| `get_embeddings` | OpenAI (timeout + retries) ou factice. |
| `get_chat_model` / `get_fallback_model` | Modèle principal / de secours (`RAG_LLM_FALLBACK_MODEL`), `None` en mode `fake`. |
| `structured_llm(settings, schema)` | `with_structured_output` **puis** `with_fallbacks` : une erreur de parsing déclenche aussi le fallback. |
| `text_llm(settings)` | Idem pour une sortie texte. |

---

## 2. Ingestion (`ingestion/`)

### `loaders.py`
`RawDocument` (source, texte, tenant, `content_hash`). `load_html` garde les titres (`<h2>` → `## `) pour le chunking structurel. `load_pdf` utilise pypdf. `SUPPORTED_EXTENSIONS` sert à l'API (415). `load_file` / `iter_corpus`.

### `chunking.py`
- `Chunk` : `text` (enfant, **indexé**), `parent_text` (contexte large, **envoyé au LLM**), `section` (`H1 > H2`), `extra` (contexte LLM).
- `embed_text` : ce qui est réellement indexé (contexte + section + texte).
- `make_chunk_id` : UUID **déterministe** → ré-ingestion idempotente.
- `chunk_document` : titres markdown → parents ~2400 caractères → enfants ~800 caractères (*small-to-big*).

### `pipeline.py`
| Fonction | Rôle |
|---|---|
| `enrich_chunks` | *Contextual Retrieval* optionnel : 1 appel LLM par chunk, en batch. |
| `ingest_document` | chunk → enrich → `index.upsert`. Un document **vidé** (0 chunk) est supprimé de l'index. |
| `ingest_path` | Fichier ou dossier ; point d'entrée commun CLI / worker / tests. |

---

## 3. Retrieval lexical : `retrieval/sparse.py`
BM25 = IDF × fréquence de terme saturée. **La TF est calculée ici, l'IDF par Qdrant** (`Modifier.IDF`, statistiques live).
- `sparse_doc_vector(text)` : vecteur sparse BM25 (k1 = 1.2, b = 0.75) d'un chunk.
- `sparse_query_vector(text)` : poids 1 par terme de la requête.

Conséquence : l'index lexical vit **dans** Qdrant, il est mis à jour en même temps que le vecteur dense et partagé par tous les réplicas. Il ne peut pas devenir périmé et ne se reconstruit jamais.

---

## 4. Index vectoriel : `indexing/store.py`

| Élément | Rôle |
|---|---|
| `Hit` | Résultat unifié. |
| `make_client` | Qdrant serveur (`url` + `api_key`) ou embarqué (`path`). |
| **Méta & versions** | |
| `_meta_get/_meta_put` | Petite collection `docs_meta` (sans vecteur) pour l'état de contrôle. |
| `bump_version(tenants)` | Nouveau jeton aléatoire par tenant, **à chaque écriture**. |
| `index_version(tenant)` | `epoch:jeton_tenant` (l'epoch change à chaque swap d'alias). Clé du cache : toute modification de contenu l'invalide. |
| `set_dual_write` / `_dual_write_targets` | Pendant un reindex, les écritures vont aussi dans la nouvelle collection. |
| **Collections / alias** | |
| `_create_collection` | Vecteur nommé `dense` (cosinus) + sparse `bm25` (IDF) + index payload `tenant`, `source`. |
| `_ensure_initial_collection` | Crée `docs_v1` + alias ; détecte un **schéma legacy** (`legacy=True` → les recherches lèvent `LegacySchemaError`). |
| `collection` | Collection physique derrière l'alias (plan de contrôle uniquement, pas sur le chemin chaud). |
| `new_version`, `swap_alias`, `drop_collection` | Blue/green ; `swap_alias` fait tourner l'epoch ; on ne peut pas supprimer la collection live. |
| **Écriture** | |
| `_points` | Embedde dense + sparse **avant** toute écriture. |
| `upsert(chunks, collection, batch, bump, replace)` | Par document : upsert des nouveaux points (ids déterministes), **puis** suppression des anciens points (`must_not HasId`). Aucune fenêtre où le document disparaît ; un échec d'embedding laisse l'index intact. `replace=False` = copie en masse. |
| `delete_source`, `delete_tenant` | Suppressions par filtre (+ bump de version). |
| **Lecture** (toujours via l'alias) | |
| `hybrid_search(query, tenant, query_vector)` | **Un seul aller-retour** : prefetch dense + prefetch BM25, chacun filtré par tenant, fusion `RrfQuery(k)` côté serveur. Réutilise le vecteur déjà calculé. |
| `dense_search`, `sparse_search` | Ablations pour les évals. |
| `count`, `scroll_payloads`, `scan`, `healthy` | Comptage, lecture paginée (tous les tenants ou un seul), probe. |

## 5. `retrieval/hybrid.py` et `rerank.py`
- `HybridRetriever.retrieve(query, tenant, query_vector)` délègue à `hybrid_search` : plus de thread pool côté client. `retrieve_dense_only` / `retrieve_sparse_only` servent aux baselines.
- `rrf_fuse` : implémentation de référence de RRF, pour fusionner des résultats de systèmes **différents**.
- `rerank.py` : `NoReranker` (défaut), `CrossEncoderReranker` (bge, extra `[rerank]`), `LLMReranker` (notes 0–10, avec fallback). `get_reranker` choisit selon `RAG_RERANKER`.

---

## 6. Génération (`generation/`)
- `prompts.py` : `REWRITE`, `ANSWER` (contexte = donnée non fiable, citations obligatoires), `CONTEXTUAL`, `RERANK`.
- `chains.py` : `build_rewriter`, `build_answerer`, `build_contextualizer`, `build_llm_scorer`, tous construits sur `structured_llm` / `text_llm` (avec fallback). Fallback hors ligne : `extractive_answer` (meilleure phrase pondérée IDF + la suivante ; refus sous 0.34). Constantes publiques : `STOPWORDS`, `REFUSAL`.
- `guardrails.py` : `check_input` (vide, longueur, injection) ; **`check_history`** (garde les N derniers tours et applique le même filtre d'injection, car un faux tour `assistant` est un vecteur connu) ; `redact_pii` ; `check_grounding` (citations ∈ chunks récupérés).

## 7. `graph.py` — pipeline LangGraph
`RagState` contient maintenant `query_vector`. Chaque nœud est enveloppé par `timed_node` (latence par nœud dans Prometheus).

| Nœud | Fait |
|---|---|
| `guard` | `check_input` + `check_history` → `status=blocked` si injection |
| `rewrite` | Réécrit seulement s'il y a un historique (et invalide alors `query_vector`) |
| `retrieve` | `retriever.retrieve(search_query, tenant, query_vector)` |
| `rerank` | Top-k ; seuil `min_relevance` appliqué **seulement si un reranker est actif** (les scores RRF n'ont pas d'échelle absolue) |
| `generate` | Réponse structurée → grounding → citations |
| `no_evidence` | Refus |

---

## 8. Transverse
### `observability.py`
- Métriques : `REQUESTS{status}`, `LATENCY`, `NODE_LATENCY{node}`, `CACHE{hit|miss}`, `TOKENS{tenant,model,kind}` (coût), `INGEST{status}`.
- `request_id` (contextvar) ; `configure_logging(json)` : logs JSON avec request id.
- `timed_node`, `record_usage(tenant, usage)`.

### `limits.py`
`Limits(rate_per_min, daily_tokens, redis_client)` :
- `allow_request(tenant)` : fenêtre fixe de 60 s.
- `has_token_budget` / `spend_tokens` : budget journalier.

Avec Redis, la limite est commune à tous les réplicas ; sans Redis, repli en mémoire borné. **Fail-open** si Redis tombe : une panne du limiteur ne doit pas devenir une panne de l'API.

### `cache.py` — cache sémantique dans Qdrant
`SemanticCache(client, name, dim, threshold, ttl_s)` utilise la collection `docs_semcache` :
- `get(vector, tenant, version)` : recherche ANN filtrée par tenant + version + TTL, seuil 0.97.
- `put(...)` : ne stocke que des réponses *grounded* ; purge périodique.
- Si la dimension d'embedding change, la collection est recréée.
- Toute erreur compte comme un miss : le cache ne casse jamais une requête.

## 9. `service.py` — composition et exécution
| Élément | Rôle |
|---|---|
| `RagService.__init__` | settings → embeddings → index → retriever → graphe → cache → limites. |
| `_execute(req, tenant)` | Chemin **unique** (query et stream) : budget tenant → version → embedding 1× → cache → `graph.stream` (yield un `step` par nœud) → comptage tokens (`UsageMetadataCallbackHandler`) → métriques → mise en cache. Yield `("step", nœud)*` puis `("result", QueryResponse)`. |
| `query`, `stream` | Consomment `_execute`. |
| `UpstreamError`, `BudgetExceeded` | Erreurs typées → 503 / 429 dans l'API. |

## 10. `queue.py` — ingestion fiable (Redis Streams)
| Élément | Rôle |
|---|---|
| `InlineQueue` | Dev : thread pool ; `status(job, tenant)` filtré par tenant. |
| `RedisQueue.enqueue` | Hash `rag:job:{id}` (statut, tenant, path, attempts) + `XADD rag:ingest`. |
| `_next_messages` | 1) `XAUTOCLAIM` des jobs abandonnés (worker mort, inactif plus longtemps que le *visibility timeout*), 2) `XREADGROUP`. |
| `_process` | Chaque livraison compte comme une tentative (un document *poison* finit en DLQ). Succès → `done` + `XACK`. Échec → retry planifié dans un ZSET (backoff 2^n, sans bloquer le worker), ou DLQ après 3 tentatives. |
| `_promote_due_retries` | Remet en file les retries échus (`ZREM` atomique : un seul worker gagne). |
| `work_once` / `work_forever(stop)` | Boucle ; arrêt propre sur `SIGTERM`. |
| `worker_main` | `python -m open_rag.queue` : logs, métriques sur `:9100`, signaux. |

## 11. `api.py`
- `lifespan` : logs, taille du thread pool (`RAG_API_THREADS`), service, file d'ingestion.
- Middleware `correlate` : `X-Request-ID` (entrant ou généré) propagé aux logs et aux traces.
- Handlers : `UpstreamError`/`LegacySchemaError` → 503 (+ `Retry-After`), `BudgetExceeded` → 429.
- `tenant_from_key` : clé → tenant (401), puis `limits.allow_request` (429).
- `POST /v1/query` ; `POST /v1/query/stream` (SSE : `step`*, `answer`, `citations`, `done`, ou `error`).
- `POST /v1/documents` : nom de base uniquement, extension autorisée, **taille max** (413), écriture dans un `.part` puis `os.replace` atomique (un worker ne lit jamais un fichier partiel).
- `GET /v1/jobs/{id}` : 404 si inconnu **ou** appartenant à un autre tenant.
- `/healthz` (process) · `/readyz` (Qdrant + Redis) · `/metrics`.

## 12. `indexing/reindex.py` — blue/green avec écritures live
`reindex(service, validate=None)` :
1. Crée `docs_v{N+1}` et **active la double écriture**.
2. Copie tous les payloads (pagination, tous les tenants, champs `extra` conservés, ré-embedding avec le modèle courant).
3. **Rattrapage** : les tenants dont la version a changé pendant la copie sont recopiés, jusqu'à une passe stable.
4. Validation : `validate(new)`, par défaut des comptes égaux par tenant.
5. Swap atomique de l'alias (+ epoch), puis désactivation de la double écriture.

En cas d'échec, tout est annulé (double écriture coupée, nouvelle collection supprimée). `rollback(service, collection)` re-pointe l'alias. CLI : `reindex`, `rollback <collection>`.

---

## 13. Déploiement
- `requirements.lock` / `requirements-dev.lock` : versions exactes (générées par `uv pip compile --universal`).
- `deploy/Dockerfile` : multi-stage, installe depuis le lock, user 10001, arrêt gracieux.
- `deploy/docker-compose.yml` : api + worker (`stop_grace_period`) + qdrant 1.19 + redis.
- `deploy/k8s/rag.yaml` : PVC RWX pour les uploads ; API avec probes startup/readiness/liveness, `securityContext` (non-root, rootfs en lecture seule, capabilities supprimées), répartition par zone, `preStop`, HPA, PDB ; worker avec grâce de 600 s et KEDA `redis-streams` sur le lag ; NetworkPolicy.
- `.github/workflows/ci.yml` : lock → ruff → pytest → éval + gate → `pip-audit` → build Docker ; sur `main`, éval avec les vrais modèles + gate absolue.

## 14. Évaluation et tests
| Fichier | Rôle |
|---|---|
| `evals/run_eval.py` | Ablation dense / sparse / hybride + bout-en-bout ; enregistre la config (provider…) dans `results/latest.json`. |
| `evals/check_thresholds.py` | Deux gates : `fake` = non-régression hors ligne, `openai` = barre de qualité absolue. |
| `tests/conftest.py` | `RAG_TEST_QDRANT_URL=http://127.0.0.1:6333` → **même suite contre un vrai serveur Qdrant** (alias unique par test). |
| `tests/test_pipeline.py`, `test_api.py` | Comportements de base. |
| `tests/test_production.py` | Invalidation cache/BM25, upsert sans perte, document réduit ou vidé, reindex (tenants, extra, rattrapage, double écriture, abandon), file (succès, crash, retry/DLQ, isolation), limites (mémoire et Redis partagé), fallback LLM, API (413/415/404, historique, SSE, request id, 503, 429). |

## 15. Exercices
1. `graph.stream(..., stream_mode="updates")` sur `"What does error NB-4021 mean?"` : notez l'état après chaque nœud.
2. Modifiez un document sans changer son nombre de chunks et observez `index_version` changer.
3. Lancez `reindex` pendant qu'un script ingère en boucle, puis vérifiez qu'aucune écriture n'a été perdue.
