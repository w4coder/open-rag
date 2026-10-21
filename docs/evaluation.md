# Évaluation

## Principe
On évalue **séparément** le retrieval et la génération : si la réponse est fausse, il faut savoir si le bon passage n'a pas été trouvé (retrieval) ou mal utilisé (génération).

| Niveau | Métriques | Où |
|---|---|---|
| Retrieval | hit@5, MRR (labels = snippets texte) | `evals/metrics.py` |
| Génération | correction, taux de citations, précision du refus | `evals/run_eval.py` |
| Production | traces LangSmith, feedback, évaluateurs online | `service.py` (tags/metadata) |

## Lancer
```bash
python -m evals.run_eval                                  # local, offline
python -m evals.check_thresholds                          # gate (exit 1 si régression)
RAG_LLM_PROVIDER=openai ... python -m evals.run_eval --langsmith   # expérience LangSmith
python -m evals.run_eval --chunk-size 400                 # une variable = une expérience
```
Dans LangSmith : onglet *Datasets & Experiments* → comparer les expériences côte à côte (préfixe `chunk800-none`, `chunk400-none`, ...).

## Résultats actuels (mode offline, corpus de 13 chunks)
| | hit@5 | MRR |
|---|---|---|
| dense seul | 1.00 | 1.00 |
| hybrid RRF | 1.00 | 1.00 |

| End-to-end | valeur |
|---|---|
| answer_correctness | 0.833 |
| citation_rate | 0.833 |
| refusal_accuracy | 1.00 |

### Lecture honnête
- **Retrieval saturé** : avec 13 chunks et `top_k=20`, tout est retourné ; ce corpus **ne peut pas** départager dense et hybride. Pour mesurer le gain réel, utiliser un corpus de milliers de chunks (ex. documentation open source) et un vrai modèle d'embedding.
- Les 2 échecs end-to-end (`q03` « rotated » vs « expires after 90 days » ; `q12` « protocol » vs « SAML 2.0 ») sont des **décalages de vocabulaire** : le retrieval trouve le bon passage mais le générateur **extractif lexical** du mode offline ne fait pas le lien. Un LLM (ou un vrai embedding côté retrieval) les résout. Bonne illustration de la séparation erreur de génération / erreur de retrieval.
- La correction est mesurée par inclusion lexicale (valable pour des faits courts). Pour des réponses libres : LLM-as-judge (faithfulness, relevance) **calibré sur un échantillon annoté humainement**.

## Construire un bon dataset
1. 30-100 questions réelles (logs, support) + questions synthétiques générées depuis les documents puis **relues**.
2. Inclure : factuelles, multi-passages, paraphrases, termes exacts (codes), **sans réponse**, ambiguës.
3. Versionner dans LangSmith ; ajouter chaque échec observé en production.

## Expériences à mener (checklist)
- [ ] chunk 400 / 800 / 1200 · [ ] avec/sans parent-child · [ ] avec/sans contextual retrieval
- [ ] dense vs hybride vs hybride + reranker · [ ] top_k rerank 3/5/8 · [ ] seuil `min_relevance`
- [ ] modèle d'embedding A vs B · [ ] modèle de génération petit vs grand (coût/qualité)

## Benchmark externe : Open RAG Bench (texte seul)
`python -m evals.open_ragbench` (ou `make eval-orb`) évalue le vrai pipeline sur
[vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) : papiers arXiv,
dont 600 distracteurs « durs », et questions extractives/abstractives avec qrels (doc, section).
- **Texte seul** pour l'instant : nos loaders ignorent images et tableaux. Le multimodal viendra ensuite.
- On ingère les **sections markdown pré-parsées** (pas les PDF) : on mesure le RAG, pas le parseur.
- Retrieval : `doc_hit@k`, `doc_mrr`, `section_hit@k`, `doc_in_candidates` (plafond du reranker).
- Bout en bout : LLM-juge guidé par la référence (1-5, correct si ≥ 4), taux de réponse,
  citations pointant dans le bon papier. Résultats détaillés par question dans
  `evals/results/open_ragbench_details.jsonl` : c'est la matière de l'analyse d'erreurs.
- L'index est mis en cache par (embedding, chunking) : changer reranker/top-k/LLM ne ré-embedde rien.
- Questions sans réponse : celles des papiers absents de l'index (`--absent`) et celles de N papiers
  retirés d'une copie de l'index (`--holdout-papers`, sans réembedding). Le bon comportement est un refus.

### Résultats (octobre 2026, `bash evals/run_full.sh`)
1 914 questions texte, 987 papiers, 262 questions sans réponse ; `gpt-4o-mini`, `text-embedding-3-small`,
juge `gpt-4.1-mini`.

| Configuration | Réponses correctes (IC 95 %) | Abstractives | Refus corrects | Latence p95 |
|---|---|---|---|---|
| Hybride RRF | 86,7 % (85,1–88,2) | 83,4 % | 96,2 % | 1,8 s |
| Hybride + reranker LLM | **90,2 % (88,8–91,4)** | **90,0 %** | 95,0 % | 5,2 s |

Test apparié : 135 questions corrigées, 69 dégradées (McNemar, p < 0,001). Échec principal restant :
refus excessifs alors que le bon passage était récupéré (environ 5 % des questions).

### Escalade des refus (octobre 2026)
Diagnostic : sur les refus à des questions répondables où la bonne section était transmise, la
réponse figurait dans le contexte dans la grande majorité des cas (faiblesse de `gpt-4o-mini`).
Avec `RAG_LLM_ESCALATION_MODEL=gpt-6-luna` (même prompt, même contexte, uniquement sur refus) :

| Configuration | Réponses correctes (IC 95 %) | Refus sur questions sans réponse | Réponses non appuyées | Escaladées | Latence p95 |
|---|---|---|---|---|---|
| Hybride + reranker LLM | 90,2 % (88,8–91,4) | 95,0 % | – | – | 5,2 s |
| + escalade | **94,2 % (93,1–95,2)** | 88,5 % | 0,8 % | 6,4 % | 5,5 s |

Test apparié : 105 corrigées, 29 dégradées (p < 10⁻¹⁰). Le taux de refus brut baisse, mais 28 des
30 réponses données aux questions « sans réponse » sont appuyées par les passages cités (un autre
papier traite le sujet). Le critère du projet (refus ≥ 95 %) est conservé tel quel pour l'instant ;
la mesure « réponses non appuyées » est ajoutée pour éclairer une éventuelle révision.
Reproduire : `RAG_LLM_ESCALATION_MODEL=gpt-6-luna python -m evals.open_ragbench --queries 1914 --negatives 600 --holdout-papers 50 --reranker llm --reuse-retrieval full_llm`.
