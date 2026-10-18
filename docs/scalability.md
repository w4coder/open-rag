# Scalabilité

## Où sont les goulots ?
| Charge | Goulot | Réponse |
|---|---|---|
| Requêtes | latence/coût LLM, débit provider | réplicas API stateless + HPA, cache sémantique, rate limiting par tenant, modèle plus petit pour rewrite/rerank, timeouts + retries bornés, fallback de modèle |
| Recherche | RAM/CPU Qdrant | cluster Qdrant (shards + replicas), quantization scalar/binary, HNSW paramétré, payload index, collections ou sharding par gros tenant |
| Ingestion | embeddings + LLM de contexte | workers découplés via file, autoscaling KEDA sur la longueur de file, batching des embeddings, DLQ |
| Ré-indexation | double stockage temporaire | blue/green (`docs_vN` + alias), évaluation avant swap |

## Principes appliqués dans le code
1. **Stateless** : l'état est dans Qdrant/Redis (`service.py`) → n'importe quel réplica répond.
2. **Découpler lecture et écriture** : `api.py` n'ingère jamais, il met en file (`queue.py`).
3. **Idempotence** partout (ids déterministes) → at-least-once sûr.
4. **Probes distinctes** : `/healthz` (process) vs `/readyz` (dépendances).
5. **Version d'index** dans la clé de cache → pas de réponses périmées.

## Ordres de grandeur (à valider par vos tests de charge)
- 1M chunks × 1536 dims float32 ≈ 6 Go de vecteurs (+ HNSW ~1,5×) ; scalar quantization int8 ÷4.
- Au-delà de ~10-50M vecteurs : sharding, quantization, éventuellement disk-based index.
- BM25 = vecteurs sparse dans Qdrant (ADR 0003) : il scale avec le cluster Qdrant, sans index dans le process.

## Ce que ce dépôt ne fait pas (volontairement) et comment l'ajouter
- **Cache distribué** : Redis + RedisVL (la classe `SemanticCache` a l'interface cible).
- **Rate limiting global** : au niveau gateway (Envoy/Kong) ou Redis token bucket.
- **Stockage objet** pour les uploads (S3/GCS) au lieu d'un volume.
- **Kafka/Streams** si besoin de replay ou fan-out multi-consommateurs.
- **Tests de charge** : Locust/k6 sur `/v1/query` avec le dataset d'évaluation comme trafic ; mesurer p50/p95/p99, coût par requête, saturation Qdrant.

## Déploiement
`deploy/docker-compose.yml` (dev) · `deploy/k8s/rag.yaml` : rolling update sans coupure, HPA CPU, PDB, workers KEDA (scale-to-zero).
