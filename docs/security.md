# Sécurité (OWASP LLM Top 10, appliqué)

| Risque | Mesure | Code |
|---|---|---|
| LLM01 Prompt injection directe | heuristiques d'entrée + refus | `guardrails.check_input` |
| LLM01 Injection **indirecte** (dans les documents) | contexte déclaré non fiable dans le prompt ; sorties structurées ; pas d'outils exécutant des actions | `prompts.ANSWER_PROMPT` |
| LLM02 Fuite de données | filtre `tenant` obligatoire sur dense **et** sparse ; PII masquée avant d'être envoyée aux traces | `store._tenant_filter`, `redact_pii` |
| LLM04 DoS / coûts | limite de taille, rate limit par tenant, timeouts LLM | `api.py`, `providers.get_chat_model` |
| LLM09 Hallucination | refus si pas de preuve ; citations vérifiées | `graph.py`, `check_grounding` |
| Upload malveillant | extension allow-list, nom de fichier normalisé (`Path.name`) | `api.upload` |

À ajouter selon le niveau de risque : classifieur d'injection (Prompt Guard / Llama Guard), ACL par document (champ `acl` dans le payload + filtre), scan antivirus des uploads, chiffrement au repos, rotation des clés, journal d'audit.

⚠ Les heuristiques regex sont contournables : elles réduisent le bruit, elles ne remplacent pas une défense en profondeur.
