"""
scripts/rebuild_indexes.py — Rebuild the FAISS and BM25 indexes from Postgres.

Postgres is the source of truth; the two search indexes are derived data. They
can drift out of sync with it when:

  * documents are deleted straight from the database (there is no delete
    endpoint, so this is the only way to remove one), leaving vectors behind
    that no longer map to any chunk;
  * an ingest failed midway in an older build, which wrote to the indexes
    before committing the transaction;
  * an index file is lost, corrupted, or copied between machines.

Drifted indexes fail silently at query time: orphaned entries are retrieved,
cannot be resolved back to a chunk row, and are dropped — so every result set
quietly comes back shorter than requested. `main.py` warns about this at
startup; this script fixes it.

The rebuild re-embeds every chunk, reassigns faiss_index_id values densely from
zero, and updates those ids in the database, so the result is a clean and
internally consistent set of indexes.

Usage (from the backend/ directory, with the virtualenv active):

    python scripts/rebuild_indexes.py            # rebuild
    python scripts/rebuild_indexes.py --check    # report drift, change nothing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/rebuild_indexes.py` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func  # noqa: E402

from database import SessionLocal  # noqa: E402
from models import Chunk  # noqa: E402
from services import bm25_store, embedder, faiss_store  # noqa: E402

log = logging.getLogger("rebuild_indexes")

BATCH_SIZE = 256


def check() -> int:
    """Report drift between the database and the indexes. Returns an exit code."""
    faiss_store.load()
    bm25_store.load()

    with SessionLocal() as session:
        chunk_count = session.query(func.count(Chunk.chunk_id)).scalar() or 0

    faiss_size = faiss_store.index_size()
    bm25_size = bm25_store.corpus_size()

    print(f"Postgres chunks : {chunk_count}")
    print(f"FAISS vectors   : {faiss_size}")
    print(f"BM25 documents  : {bm25_size}")

    if faiss_size == chunk_count and bm25_size == chunk_count:
        print("\nIndexes are consistent with the database.")
        return 0

    print(
        f"\nDRIFT DETECTED: FAISS is off by {faiss_size - chunk_count:+d}, "
        f"BM25 by {bm25_size - chunk_count:+d}."
    )
    print("Run this script without --check to rebuild.")
    return 1


def rebuild() -> int:
    """Re-embed and re-index every chunk in the database. Returns an exit code."""
    embedder.load()
    faiss_store.reset()
    bm25_store.reset()

    with SessionLocal() as session:
        total = session.query(func.count(Chunk.chunk_id)).scalar() or 0
        if total == 0:
            faiss_store.persist()
            bm25_store.persist()
            print("No chunks in the database — wrote empty indexes.")
            return 0

        print(f"Rebuilding indexes for {total} chunks …")
        processed = 0

        # Ordered by chunk_id so the run is deterministic and resumable-looking.
        query = session.query(Chunk).order_by(Chunk.chunk_id)

        for offset in range(0, total, BATCH_SIZE):
            batch: list[Chunk] = query.offset(offset).limit(BATCH_SIZE).all()
            if not batch:
                break

            # `content` is what was embedded at ingest time (it carries the
            # "[Section: …]" prefix); `content` is re-used here so vectors match
            # the originals. BM25 indexes the same text the ingest path used.
            vectors = embedder.embed([c.content for c in batch])
            faiss_ids = [faiss_store.next_id() for _ in batch]
            faiss_store.add(faiss_ids, vectors, persist=False)
            bm25_store.add(
                [c.chunk_id for c in batch],
                [c.content for c in batch],
                persist=False,
            )

            # Point each chunk at its new position in the rebuilt index.
            for chunk, faiss_id in zip(batch, faiss_ids):
                chunk.faiss_index_id = faiss_id

            processed += len(batch)
            print(f"  {processed}/{total} chunks", end="\r", flush=True)

        session.commit()

    faiss_store.persist()
    bm25_store.persist()

    print(f"\nDone. FAISS: {faiss_store.index_size()} vectors, "
          f"BM25: {bm25_store.corpus_size()} documents.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without modifying anything.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    return check() if args.check else rebuild()


if __name__ == "__main__":
    raise SystemExit(main())
