"""CLI: `python -m open_rag.cli ingest <path>` | `ask "<question>"` | `reindex` | `rollback <coll>`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_rag.models import QueryRequest
from open_rag.service import RagService


def main() -> None:
    p = argparse.ArgumentParser(prog="open-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("path", type=Path)
    ing.add_argument("--tenant", default="tenant-a")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--tenant", default="tenant-a")
    sub.add_parser("reindex", help="blue/green re-embedding into a new collection version")
    rb = sub.add_parser("rollback", help="point the alias back to a previous collection")
    rb.add_argument("collection")
    args = p.parse_args()

    svc = RagService()
    if args.cmd == "ingest":
        from open_rag.ingestion.pipeline import ingest_path

        print(f"{ingest_path(args.path, svc.index, svc.settings, args.tenant)} chunks indexed")
    elif args.cmd == "ask":
        print(json.dumps(svc.query(QueryRequest(question=args.question), args.tenant).model_dump(),
                         indent=2, ensure_ascii=False))
    elif args.cmd == "reindex":
        from open_rag.indexing.reindex import reindex

        print(reindex(svc))
    elif args.cmd == "rollback":
        from open_rag.indexing.reindex import rollback

        print(rollback(svc, args.collection))


if __name__ == "__main__":
    main()
