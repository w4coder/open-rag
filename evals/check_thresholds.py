"""CI quality gate: exit 1 if the latest eval run is below the agreed floor.

Two floors, chosen by the provider recorded in results/latest.json:
* `fake`   - offline CI on every PR. The extractive stand-in cannot paraphrase (q03 asks about
             "rotation", the doc says "expires"), so this is a *regression* gate pinned to
             the known offline baseline, not a quality claim.
* `openai` - real models (main branch / nightly, with secrets): the absolute quality bar.
"""
import json
import sys
from pathlib import Path

THRESHOLDS = {
    "fake": {
        ("hybrid_rrf", "hit@5"): 0.85,
        ("hybrid_rrf", "mrr"): 0.6,
        ("end_to_end", "answer_correctness"): 0.8,
        ("end_to_end", "refusal_accuracy"): 0.9,
        ("end_to_end", "citation_rate"): 0.8,
    },
    "openai": {
        ("hybrid_rrf", "hit@5"): 0.9,
        ("hybrid_rrf", "mrr"): 0.7,
        ("end_to_end", "answer_correctness"): 0.9,
        ("end_to_end", "refusal_accuracy"): 0.95,
        ("end_to_end", "citation_rate"): 0.95,
    },
}

res = json.loads((Path(__file__).parent / "results" / "latest.json").read_text())
provider = res.get("config", {}).get("llm_provider", "fake")
floors = THRESHOLDS[provider]
failed = [(k, res[k[0]][k[1]], v) for k, v in floors.items() if res[k[0]][k[1]] < v]
for (grp, metric), got, floor in failed:
    print(f"FAIL [{provider}] {grp}.{metric}: {got} < {floor}")
print(f"quality gate [{provider}]: {'FAILED' if failed else 'passed'}")
sys.exit(1 if failed else 0)
