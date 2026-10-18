# ADR 0003 — Recherche hybride (dense + BM25) avec RRF, côté Qdrant

**Décision** : chaque point Qdrant porte un vecteur dense (`dense`) et un vecteur sparse BM25 (`bm25`, `Modifier.IDF`). Une seule requête `query_points` fait deux `prefetch` filtrés par tenant puis une fusion `RrfQuery(k=60)` côté serveur ; rerank optionnel ensuite.

**Pourquoi** : le dense échoue sur les identifiants exacts (`NB-4021`), le BM25 sur les paraphrases. RRF évite de normaliser des scores incomparables.

**Historique** : v0.1 utilisait un BM25 en mémoire du process (`rank_bm25`), reconstruit quand le *nombre* de points changeait. Problèmes : index périmé quand un document changeait sans changer de taille (bug reproduit), mémoire et reconstruction O(N) par réplica, 2 threads partagés pour toutes les requêtes. Remplacé en v0.2.

**Conséquences** : la TF est calculée côté client (`retrieval/sparse.py`), l'IDF par Qdrant à partir des statistiques live. Tout changement de tokenizer ou d'encodeur sparse impose un reindex blue/green (`cli reindex`), car l'index et la requête doivent rester alignés. Pour aller plus loin : SPLADE/BM42 sous le même nom de vecteur.
