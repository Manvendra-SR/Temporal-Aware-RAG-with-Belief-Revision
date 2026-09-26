# Temporal RAG with Belief Revision — Architecture

Interactive version: `docs/architecture.html` (open in any browser; click through the 4 flow tabs).

## Components

| Node | Role | What it is |
|---|---|---|
| Browser | client | React 18 SPA (`frontend/src`) — Ingest, Library, Query, Conflicts pages |
| FastAPI Backend | orchestrator | `backend/main.py` + routers (`query`, `ingest`, `conflicts`, `analytics`, `health`), mounted at `/api/v1` |
| Embedder | preprocessor | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim), one in-process singleton used both at ingest and at query time |
| Groq LLM | model call | Hosted model `openai/gpt-oss-120b` via Groq API, used for (1) temporal intent classification and (2) final answer generation |
| NLI Cross-Encoder | model call | `cross-encoder/nli-deberta-v3-small`, loaded once at FastAPI startup, runs locally (no external API) |
| Hybrid Index | datastore | FAISS `IndexHNSWFlat` (dense) + `rank_bm25` `BM25Okapi` (keyword), both file-backed under `data/`, derived from Postgres |
| Postgres | datastore | Source of truth: `Document`, `Chunk`, `QueryLog`, `ConflictPair` tables |
| Version Resolver | ingest-time job | `services/version_resolver.py` — validates a submitted version against its parent, then links and supersedes it |

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

### 4. Resolve Conflict (`POST /api/v1/conflicts/{id}/resolve`)

A person reviews a flagged contradiction on the Conflicts page and records a resolution (e.g. "prefer newer"). This is stored on the `ConflictPair` row and is checked first by `belief_revision._apply_stored_resolution()` on every future query touching that pair — a human resolution always overrides the automatic recency/lineage rules.

## Key architectural notes

- **Three storage backends**, all singletons loaded once at FastAPI startup and kept in-process: Postgres (source of truth), FAISS (dense vectors, file-backed), BM25 (keyword index, file-backed, pickled, rebuilt in full on every ingest since `rank_bm25` has no incremental update).
- **One embedding model** shared between ingest and query time — chunks and queries land in the same vector space by construction.
- **One NLI model**, local, separate from the two Groq calls — conflict detection never leaves the machine.
- **Version lineage is written only at ingest time** (`version_resolver.apply_lineage()`, guarded ahead of time by `validate_lineage()`) and read everywhere else (retriever, temporal reranker, conflict detector) — it is the single source of "what's current" and "what supersedes what."
- The evaluation harness (`evaluation/run_eval.py`) is a black-box HTTP client against the same `/api/v1/query` endpoint the frontend uses — it calls it twice per question (temporal vs baseline) and grades both with an LLM judge.
