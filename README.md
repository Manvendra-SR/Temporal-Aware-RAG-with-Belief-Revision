# Temporal-Aware RAG with Belief Revision

A retrieval-augmented generation system that treats documents as **versioned over
time**. It tracks which version of a document supersedes which, prefers current
information when answering, detects when two retrieved passages contradict each
other, and decides which claim should inform the answer.

The problem it addresses: a plain RAG system retrieves whatever is semantically
similar, so an outdated passage from an old release competes on equal footing
with the current one. Ask "is this API still recommended?" and you may well get
the answer from three versions ago.

> **Status:** the system is implemented and runs end to end. It has **not yet
> been quantitatively evaluated** against a baseline — no accuracy claims are
> made anywhere in this repository. See [Evaluation](#evaluation-not-yet-run).

---

## What it does

**Ingestion** — Upload a PDF, Markdown or text file with a version string and a
publication date. The file is parsed, split into structure-aware chunks
(headings, code blocks and tables are respected; long prose is split on
sentence-embedding similarity), embedded, and indexed into both a FAISS vector
index and a BM25 keyword index.

**Version lineage** — Uploading a new version links it to its parent. All
ancestors are marked superseded and their chunks are stamped with a validity
end-date. The version guard rejects an upload whose version is not strictly
greater than its parent's.

**Retrieval** — Hybrid search: BM25 and dense vector search are fused with
Reciprocal Rank Fusion.

**Temporal reranking** — Candidates are re-scored by combining fused relevance,
an exponential freshness decay, how well the chunk's version matches any version
the question named, and whether it is the current version. Superseded chunks are
filtered out unless the question asks about the past.

**Conflict detection** — Top candidates from *different* documents are paired by
a cheap lexical gate, then scored by an NLI cross-encoder. Contradictions are
classified and cached in the database so they are not recomputed.

**Belief revision** — For each conflict, a decision tree decides what reaches the
model: the superseded claim is excluded but quoted in a notice so the answer can
explain what changed, and the answer is assigned a confidence level. A version
the user explicitly asked for is never excluded.

---

## Architecture

```
frontend/  React + Vite + TypeScript
  Ingest    upload a document or a new version of one
  Library   documents, chunk inspection, version-lineage timeline
  Query     ask a question; see the sources, scores and conflicts behind it
  Conflicts review contradictions and record how they were resolved

backend/   FastAPI + SQLAlchemy + PostgreSQL
  routers/   ingest, query, conflicts, analytics, health
  services/  parser → chunker → embedder → {faiss,bm25}_store
             retriever → temporal_reranker → conflict_detector
                       → belief_revision → context → llm
             version_resolver, query_analyzer, nli, sentence_splitter
  scripts/   rebuild_indexes.py, list_llm_models.py
  tests/     unit tests for the deterministic logic

data/      faiss.index, bm25.pkl  (derived from Postgres, not in version control)
```

Postgres is the source of truth. The two search indexes are derived from it and
can be rebuilt at any time — see [Rebuilding the indexes](#rebuilding-the-search-indexes).

---

## Quick start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL running locally
- A [Groq](https://console.groq.com) API key (for answer generation; retrieval
  works without one)

### 1. Database

```bash
psql postgres -c "CREATE DATABASE temporal_rag;"
```

Tables are created automatically at startup. There are no migrations — during
development, change a model and recreate the table.

### 2. Configuration

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env` and set at minimum `DB_PASSWORD` and `GROQ_API_KEY`.

### 3. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

On first run this downloads the embedding model (~90 MB) and the NLI
cross-encoder (~80 MB), then caches them.

`requirements.txt` pins CUDA builds of PyTorch. On a machine without an NVIDIA
GPU, install the CPU build instead (`pip install torch --index-url
https://download.pytorch.org/whl/cpu`) before installing the rest. The
application itself detects the device automatically and runs fine on CPU.

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The sidebar shows a green "API connected" dot when
the backend is reachable.

---

## Trying it out

The version-awareness only becomes visible with two versions of one document:

1. **Ingest** a document as a *New document* with version `1.0` and a
   publication date a year or two in the past.
2. **Ingest** an edited copy as a *New version*, selecting the first as its
   parent, with version `2.0` and a recent date — change a factual claim so the
   two versions genuinely disagree.
3. **Query** something the two versions answer differently. The current version
   is used, and the superseded passage is filtered out.
4. Ask the same thing with historical wording ("was it previously …?"). Now both
   versions are retrieved, the contradiction is detected, and the answer explains
   what changed and when.
5. Tick **Compare against standard RAG** to run the same question with the
   temporal pipeline disabled, side by side.

---

## Score semantics

The UI is deliberate about which numbers are shown as bars and which as raw
values, because they are not on comparable scales:

| Score | Range | Shown as |
|---|---|---|
| `relevance_score` | 0–1, normalised per query | bar — "Relevance" |
| `temporal_score` | 0–1 | bar — "Recency" |
| `composite_score` | 0–1 | bar — "Final score" |
| `semantic_score` | 0–1 cosine | raw number |
| `bm25_score` | unbounded | raw number |
| `rrf_score` | ≤ 2/61 ≈ 0.033 | raw number |

BM25 and RRF are never rendered as percentages. Full contract in
[`services/retriever.py`](backend/services/retriever.py).

---

## Temporal decay

A chunk's freshness weight is `2 ** (-age_days / half_life)`, so a chunk exactly
one half-life old scores 0.5.

The half-life is a **single global value**, `TEMPORAL_HALF_LIFE_DAYS` (default
180), defined once in [`backend/config.py`](backend/config.py). An earlier design
sketched per-domain half-lives backed by a `domain_config` table; that was never
implemented and there is no `domain` concept in the data model, so per-domain
decay is not supported.

The right value is an open question — 180 days is a starting point, not a
calibrated figure, and choosing it properly is a job for the evaluation.

---

## Version ordering

One policy, defined in
[`services/version_resolver.py`](backend/services/version_resolver.py) and used
everywhere versions are compared or sorted:

- Versions reduce to their integer components: `v1.13.1` → `(1, 13, 1)`
- Trailing zeros are insignificant: `2.2` == `2.2.0` == `2.2.0.0`
- Components compare numerically: `2.10` > `2.9`
- Missing or unparseable versions sort below every real version and are never
  "greater than" anything

Date-style versions (`2024-03-01`) parse as `(2024, 3, 1)`, so they always sort
above semver-style ones. A single lineage should therefore use one scheme
throughout.

---

## Testing

```bash
cd backend
python -m pytest
```

The suite covers the deterministic logic — version ordering, query analysis,
temporal scoring, belief revision, conflict-detection helpers and score
semantics. It needs no database, no models and no network, and runs in under a
second.

Not covered: the NLI model's judgements, the LLM's output, and the HTTP layer.
Those need an integration test against a running stack.

---

## Rebuilding the search indexes

FAISS and BM25 are derived from Postgres and can drift out of sync with it —
deleting documents directly from the database leaves orphaned vectors that are
retrieved, fail to resolve to a row, and are silently dropped, so results come
back short. The backend warns about this at startup.

```bash
cd backend
python scripts/rebuild_indexes.py --check   # report drift
python scripts/rebuild_indexes.py           # re-embed and rebuild both indexes
```

---

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` / `DB_NAME` | postgres / — / localhost / 5432 / temporal_rag | PostgreSQL connection |
| `GROQ_API_KEY` | — | Required for answer generation |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model id |
| `EMBEDDING_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `TEMPORAL_HALF_LIFE_DAYS` | `180` | Freshness half-life |
| `LOG_LEVEL` | `INFO` | Python log level |
| `DB_ECHO` | `false` | Log every SQL statement |

Groq retires hosted models periodically. If answer generation fails with a
model error, run `python scripts/list_llm_models.py` to see what your key can
reach and update `LLM_MODEL`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, DB connectivity, index sizes |
| POST | `/api/v1/ingest` | Upload a document (multipart) |
| GET | `/api/v1/documents` | Paginated document list |
| GET | `/api/v1/documents/roots` | Current versions, for the parent selector |
| GET | `/api/v1/documents/{id}` | Document with its chunks |
| GET | `/api/v1/documents/{id}/lineage` | Version chain, oldest first |
| POST | `/api/v1/query` | Ask a question |
| GET | `/api/v1/conflicts` | Detected contradictions (filterable) |
| POST | `/api/v1/conflicts/{id}/resolve` | Record a resolution |
| GET | `/api/v1/analytics/overview` | Corpus counts |
| GET | `/api/v1/analytics/timeline` | Version lineage graph |

Interactive docs at <http://localhost:8000/docs> while the backend is running.

---

## Evaluation (not yet run)

The system has not been measured against a baseline. Nothing in this repository
reports accuracy, precision, or improvement figures, and none should be quoted
until an evaluation has actually been run.

The pieces needed for one are in place: `POST /api/v1/query` accepts
`no_temporal: true` to disable temporal reranking, conflict detection and belief
revision, giving a like-for-like standard-RAG baseline over the same retrieval
stack; every query is written to the `query_log` table; and the response reports
which sources informed the answer, which were excluded and why.

---

## Known limitations

- **No document deletion.** Removing a document means deleting its rows and
  rebuilding the indexes.
- **The half-life is uncalibrated.** 180 days is a plausible default, not a
  measured one.
- **Conflict detection is bounded.** Only the top 20 reranked candidates are
  scanned, and only across different documents.
- **Chunk contents are not deduplicated** across versions, so unchanged sections
  are embedded and stored again for every version.
- **No authentication.** It binds to localhost and assumes a single trusted user.
