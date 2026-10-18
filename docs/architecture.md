# Architecture

```mermaid
sequenceDiagram
  participant C as Client
  participant API as FastAPI
  participant Cache as Cache sémantique
  participant G as Graph LangGraph
  participant Q as Qdrant
  participant L as LLM
  C->>API: POST /v1/query (X-API-Key)
  API->>API: API key → tenant, rate limit
  API->>Cache: lookup (tenant, version d'index)
  alt hit
    Cache-->>C: réponse en cache
  else miss
    API->>G: invoke
    G->>G: guard (longueur, injection, historique)
    G->>L: rewrite (si historique)
    G->>Q: 1 requête : prefetch dense + prefetch BM25 (filtre tenant) → RRF serveur
    G->>G: rerank + seuil
    alt aucune preuve
      G-->>API: refus (no_evidence)
    else
      G->>L: generate (sortie structurée)
      G->>G: vérification des citations
    end
    API-->>C: réponse + citations
  end
```

## Plan de données
- Collection physique `docs_vN`, alias `docs`. Payload : `chunk_id, text, parent_text, source, section, tenant, doc_hash`.
- Index payload sur `tenant` et `source` (obligatoires pour la perf filtrée).

## Pipeline d'ingestion
`upload → fichier (stockage objet) → queue → worker → load → chunk → (contexte LLM) → embed → upsert (replace par source)`.

## Pourquoi LangGraph ici
Pipeline linéaire avec 3 branches (blocked / no_evidence / ungrounded). Le graphe donne : spans nommés par nœud dans LangSmith, état typé, tests par nœud, et c'est la base réutilisée dans `agentic-rag` (où l'on ajoute des boucles).

## Décisions
[ADR 0001-0004](adr/) — stack, chunking, hybride, file d'ingestion.
