#!/usr/bin/env python
"""
Upload the five versions of the Northwind policy to a running backend, as one
version lineage (each version is the child of the previous one).

    python evaluation/ingest_corpus.py [--url http://localhost:8000]

Run it once, against the *isolated* backend described in evaluation/README.md.
It refuses to touch a backend that already holds other documents, because extra
documents would leak into the evaluation's retrieval results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000", help="backend base URL")
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    title, versions = manifest["title"], manifest["versions"]
    api = args.url.rstrip("/") + "/api/v1"

    with httpx.Client(timeout=300) as http:
        try:
            listing = http.get(f"{api}/documents", params={"limit": 100}).json()
        except httpx.ConnectError:
            print(f"Cannot reach the backend at {args.url}. Start it first (see evaluation/README.md).")
            return 1

        existing = listing["items"]
        if existing:
            ours = [d for d in existing if d["title"] == title]
            if len(ours) == len(existing) == len(versions):
                print(f"The {len(versions)} policy versions are already ingested. Nothing to do.")
                return 0
            print(
                f"The backend already contains {listing['total']} document(s), so ingesting here "
                "would mix them into the evaluation.\nUse the isolated database and index "
                "directory described in evaluation/README.md."
            )
            return 1

        parent_id = None
        for v in versions:
            path = HERE / "documents" / v["file"]
            form = {"title": title, "version_string": v["version"], "published_at": v["published_at"]}
            if parent_id:
                form["parent_doc_id"] = parent_id
            resp = http.post(
                f"{api}/ingest",
                data=form,
                files={"file": (v["file"], path.read_bytes(), "text/markdown")},
            )
            if resp.status_code != 201:
                print(f"Ingesting {v['file']} failed: HTTP {resp.status_code} {resp.text[:300]}")
                return 1
            body = resp.json()
            parent_id = body["doc_id"]
            print(f"  v{v['version']:<4} {v['published_at']}  {body['chunks_created']} chunks  ({body['lineage_message']})")

    print(f"\nIngested {len(versions)} versions of '{title}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
