# ADR 0002 — Chunking : structure + parent-child

**Contexte** : la taille/forme des chunks est le paramètre le plus influent sur la qualité.

**Décision** : (1) découpage par titres markdown, (2) blocs parents ~2400 car., (3) enfants ~800 car. avec 100 de recouvrement ; enfants indexés, parents envoyés au LLM. Le chemin de section est préfixé au texte embeddé.

**Alternatives écartées** : taille fixe naïve (coupe les idées) ; *semantic chunking* par similarité d'embeddings (coûteux, gain variable → à tester comme expérience) ; *late chunking* (nécessite embeddings long-contexte).

**Conséquences** : le parent est dupliqué dans le payload de chaque enfant (coût stockage ×~2-3) ; avantage : pas de docstore séparé. À grande échelle, stocker les parents dans un KV (Redis/Postgres) et ne garder que `parent_id`.

**Validation** : `python -m evals.run_eval --chunk-size N` sur un vrai corpus.
