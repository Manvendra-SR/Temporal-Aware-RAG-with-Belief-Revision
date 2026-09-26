# Temporal RAG with Belief Revision — Architecture

Interactive version: `docs/architecture.html` (open in any browser; click through the 5 flow tabs).

## Components

| Node | Role | What it is |
|---|---|---|
| Browser | client | React 18 SPA (`frontend/src`) — Ingest, Library, Query, Conflicts pages |
| FastAPI Backend | orchestrator | `backend/main.py` + routers (`query`, `ingest`, `conflicts`, `analytics`, `health`), mounted at `/api/v1` |
| Embedder | preprocessor | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim), one in-process singleton used both at ingest and at query time |
| Groq LLM | model call | Hosted model `openai/gpt-oss-120b` via Groq API, used for (1) temporal intent classification and (2) final answer generation |
| NLI Cross-Encoder | model call | `cross-encoder/nli-deberta-v3-small`, loaded once at FastAPI startup, runs locally (no external API) |
| Hybrid Index | datastore | FAISS `IndexFlatIP` behind an `IndexIDMap` (dense) + `rank_bm25` `BM25Okapi` (keyword), both file-backed under `data/`, derived from Postgres |
| Postgres | datastore | Source of truth: `Document`, `Chunk`, `QueryLog`, `ConflictPair` tables |
| Version Resolver | lineage logic | `services/version_resolver.py` — validates a submitted version against its parent, links and supersedes it at ingest, and unlinks it again on delete |

## Flows

### 1. Temporal Query (`POST /api/v1/query`, default)

```
query_analyzer.analyze(query)            [Groq call #1: classify intent]
  → retriever.hybrid_retrieve(query, k)  [BM25 + FAISS + Reciprocal Rank Fusion, joined to Postgres]
    → temporal_reranker.rerank(...)      [drop chunks outside their valid_from/valid_to window]
      → conflict_detector.detect(...)    [batch NLI on top 20 candidates, classify, persist to Postgres]
        → belief_revision.revise(...)    [include/exclude chunks, confidence, conflict notices]
          → context.build_context(...)   [token-budgeted prompt]
            → llm.generate(...)          [Groq call #2: final answer]
              → QueryLog write → response]
```

Two Groq calls and one local NLI batch call per query. Time (validity window) decides which chunks are *eligible*; relevance score decides *rank*; NLI + belief revision decide what to do when two eligible chunks disagree.

### 2. Baseline Query (Compare mode)

Same `/query` endpoint with `no_temporal: true`. Skips intent classification, temporal filtering, conflict detection, and belief revision — plain hybrid RAG on relevance score alone. `QueryPage.tsx` fires this and the temporal flow in parallel (`Promise.all`) when "Compare against standard RAG" is checked, and the evaluation harness (`evaluation/run_eval.py`) runs both arms for every question to measure the improvement from temporal reasoning.

### 3. Ingest Document (`POST /api/v1/ingest`)

The only ingestion entry point in the running app (there's a separate standalone script, `evaluation/ingest_corpus.py`, that loads the evaluation corpus directly). Sequence:

```
validate metadata format
  → version_resolver.validate_lineage()      [read-only guards, no writes]
    → SHA-256 dedup check → parse → chunk → embed
      → write Document + Chunk rows
        → version_resolver.apply_lineage()   [supersede the parent, under its row lock]
          → COMMIT
            → add to FAISS/BM25 indexes
```

**Early validation → expensive processing → transactional lineage mutation.** The lineage guards (parent exists, parent is the latest version of its lineage, the new version is strictly greater, the new date is not earlier) need only the submitted metadata and the parent row — never the new document. Running them first means an out-of-order version is rejected with a 422 before the file is parsed, chunked and embedded, instead of after all of that work has been done and thrown away by a rollback.

The early check cannot be authoritative, though: between it and the commit, a concurrent ingest may have superseded the same parent, which would leave two "latest" versions in one lineage. `apply_lineage()` therefore re-loads the parent `FOR UPDATE` and runs the same `_check_guards()` again inside the transaction — one implementation of the rules, called from both phases, with the second call holding the lock that makes it atomic.

Indexing happens *after* the DB commit deliberately, so a crash mid-index is recoverable by rebuilding the indexes from Postgres (`backend/scripts/rebuild_indexes.py`).

Together the two phases are the single mechanism that maintains the version-lineage backbone: `Document.parent_doc_id` + `is_latest`, and `Chunk.valid_from` / `valid_to` / `is_superseded`. Both the temporal reranker's eligibility filter and the conflict detector's `version_supersession` vs `direct_contradiction` classification read from this same data.

### 4. Delete Document (`DELETE /api/v1/documents/{doc_id}`)

Deletion is lineage-aware. **A document can only be deleted if no other version names it as its parent**, so for a chain

```
v1 → v2 → v3 (latest)

delete v3 → allowed
delete v2 → 409, v3 depends on it
delete v1 → 409, v2 depends on it
```

That rule is stricter than it looks: `apply_lineage()` only accepts a parent that is still `is_latest`, so every superseded version already has exactly one child. **The one deletable version of a lineage is therefore its current version** — a lineage can only be dismantled newest-first. The alternative, relinking `v3.parent_doc_id` to `v1` when `v2` goes, would rewrite history the validity windows were derived from; refusing is both simpler and truer. (The `parent_doc_id` foreign key is `ON DELETE SET NULL`, so deleting a parent without this check would not even error — it would silently split one lineage into two.)

Sequence:

```
load the document FOR UPDATE      [404 if it does not exist]
  → version_resolver.unlink_lineage()
      no descendants?             [409 and no writes if there are]
      restore the parent as latest
  → read the chunks' chunk_ids and faiss_index_ids
  → DELETE the document row       [chunks cascade, then conflict_pairs]
    → COMMIT
      → faiss_store.remove(faiss_ids)   [exact, then persisted]
      → bm25_store.remove(chunk_ids)    [exact, then persisted]
```

What happens to each piece of state:

- **`is_latest`** — deleting the current version makes its parent current again, and reopens the parent's chunks (`valid_to = NULL`, `is_superseded = false`). This is the exact inverse of the supersession that ingesting the deleted version performed, and it is required, not cosmetic: without it the lineage would be left with no current version and every chunk expired, so it would vanish from "current" questions while still sitting in the database. A document with no parent (a whole lineage of one) simply disappears.
- **Chunks** — removed by the database through the `ON DELETE CASCADE` on `chunks.doc_id`; `conflict_pairs` cascade from the chunks in turn.
- **Parent/child links** — unchanged, because only a childless document is ever deleted. No `parent_doc_id` is ever rewritten.
- **BM25** — the chunks are removed exactly; its corpus is a plain list.
- **FAISS** — the vectors are removed, by the chunks' `faiss_index_id` values, with `remove_ids()`. The index is flat, so removal is exact and permanent: nothing is left behind to filter out later. The id counter is not rewound, so a deleted id is never handed to a future chunk.

The whole database operation is one transaction and the target row is held `FOR UPDATE` from the first read, so a concurrent ingest cannot add a child between the dependency check and the delete: whichever commits first wins and the other is rejected (409 for the delete, 422 for the ingest).

The two index files cannot join that transaction, so the ordering decides how an interrupted delete fails. Postgres commits first: the worst case is index entries that outlive their rows, which retrieval already ignores (a hit that resolves to no chunk row is skipped) and `rebuild_indexes.py` cleans up. The reverse order would leave a document that still exists but can no longer be found — silent, and much harder to notice.

### 5. Resolve Conflict (`POST /api/v1/conflicts/{id}/resolve`)

A person reviews a flagged contradiction on the Conflicts page and records a resolution (e.g. "prefer newer"). This is stored on the `ConflictPair` row and is checked first by `belief_revision._apply_stored_resolution()` on every future query touching that pair — a human resolution always overrides the automatic recency/lineage rules.

## Key architectural notes

- **Three storage backends**, all singletons loaded once at FastAPI startup and kept in-process: Postgres (source of truth), FAISS (dense vectors, file-backed), BM25 (keyword index, file-backed, pickled, rebuilt in full on every ingest since `rank_bm25` has no incremental update).
- **The dense index is flat, not approximate.** `IndexFlatIP` scans every vector, which at this corpus size costs single-digit milliseconds (~2.6 ms over 10k vectors, ~6.4 ms over 100k — against two Groq calls of several hundred ms each per query), and buys two things an HNSW graph could not: exact top-k rather than an approximation, and `remove_ids()`, so deleting a document actually deletes its vectors. Inner product is the right metric because `services/embedder.py` L2-normalises every vector, which makes the inner product the cosine similarity — the same quantity the reranker and the UI have always displayed.
- **One embedding model** shared between ingest and query time — chunks and queries land in the same vector space by construction.
- **One NLI model**, local, separate from the two Groq calls — conflict detection never leaves the machine.
- **Version lineage is written only by `services/version_resolver.py`** — `apply_lineage()` at ingest (guarded ahead of time by `validate_lineage()`) and `unlink_lineage()` on delete — and read everywhere else (retriever, temporal reranker, conflict detector). It is the single source of "what's current" and "what supersedes what."
- **A lineage is a linked list, built and dismantled from the newest end.** Ingest requires the parent to be `is_latest`, so a document has at most one child; delete requires no child at all. Those two rules together are what keep exactly one `is_latest` version per lineage without any repair logic.
- The evaluation harness (`evaluation/run_eval.py`) is a black-box HTTP client against the same `/api/v1/query` endpoint the frontend uses — it calls it twice per question (temporal vs baseline) and grades both with an LLM judge.
