# ADR 0004 — Ingestion asynchrone par Redis Streams

**Décision** : l'API dépose un job (`XADD rag:ingest`) ; des workers d'un consumer group le traitent puis l'acquittent (`XACK`). Un job resté *pending* plus longtemps que le visibility timeout (worker mort) est repris par `XAUTOCLAIM`. Chaque livraison compte comme une tentative ; au bout de 3 → stream DLQ. Les retries sont différés via un ZSET (backoff 2^n) sans bloquer le worker. Sans Redis, `InlineQueue` garde la même interface.

**Pourquoi** : ingestion en rafales et longue, scaling indépendant (KEDA `redis-streams` sur le lag), isolation des pannes.

**Garanties** : at-least-once, y compris en cas de crash du worker ; l'idempotence de l'ingestion (ids déterministes + remplacement par source sans fenêtre de perte) rend les rejeux sûrs. Le statut d'un job n'est visible que par son tenant.

**Historique** : v0.1 utilisait `LPUSH`/`BRPOP`, qui perdait le job si le worker mourait en cours de traitement ; le backoff bloquait aussi le worker.

**Alternatives** : SQS (visibility timeout natif), Kafka (rejeu, fan-out) derrière le même `JobQueue`.
