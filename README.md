# Temporal RAG with Belief Revision

A RAG system for documents that **change over time**. It knows which version of
a document replaced which, works out *when* a question is asking about, and only
uses the versions that were valid then. If two sources contradict each other, it
decides what the answer should trust.

**The problem:** a normal RAG system retrieves whatever text looks most similar
to the question, so an old version ("Rahul is the CEO") can beat the current one
("Priya is the CEO"). This project fixes that.

Built with FastAPI, PostgreSQL, FAISS + BM25, a Hugging Face NLI model, Groq
(LLM) and a React frontend.

---

## How it works

1. **Ingest.** Upload a document with a version and a date. When a new version
   is added, the old version's chunks get an end date. Every chunk therefore has
   a validity window: `valid_from` to `valid_to`.
2. **Understand the question.** One LLM call reads the question and returns its
   time intent as JSON (table below). The LLM only interprets the question; it
   never chooses which chunks to use.
3. **Retrieve.** Hybrid search: BM25 keyword search + FAISS vector search,
   merged with Reciprocal Rank Fusion. It fetches 3x the needed chunks because
   the next step throws many away.
4. **Filter and rank.** Plain code keeps only the chunks valid for that intent.
   Recency scoring is used only for "current" questions, so a question about
   2023 isn't pulled toward the newest version.
5. **Detect conflicts.** An NLI model checks whether retrieved chunks from
   different documents contradict each other. Two versions of the same document
   are labelled a *version change*; unrelated documents are a *contradiction*.
6. **Belief revision.** Simple rules decide what to do with each conflict. A
   source is only dropped for questions about the present, and only when the
   sources are unrelated and one is clearly newer. For past questions, both are
   kept and the LLM is told each one's validity dates. The answer gets a
   confidence level.
7. **Answer.** The LLM writes a cited answer from sources labelled with their
   validity windows.

| Intent | Example question | Chunks kept |
|---|---|---|
| current | "Who is the CEO?" | valid today |
| point_in_time | "Who was CEO on 2023-06-15?" | valid on that date |
| range | "Who was CEO during 2023?" | valid at any time in the range |
| version | "According to v2.0, who was CEO?" | that version only |
| historical | "Who used to be CEO?" | all versions |
| atemporal | "What does the policy cover?" | valid today |

If the LLM call fails or returns invalid output, the question is treated as
"current", and the UI says so.

### Example

Three versions of a handbook: v1.0 (Rahul is CEO), v2.0 (Priya becomes CEO in
March 2023), v3.0 (Priya continues, from 2025).

| Question | Read as | Evidence used |
|---|---|---|
| Who is the CEO now? | current | v3.0 only |
| Who was the CEO in 2023? | range 2023-01-01 to 2023-12-31 | v1.0 and v2.0 |
| Who was the CEO according to v2? | version 2 | v2.0 only |

For the 2023 question the answer is "Rahul until March, then Priya". A normal
RAG system would see all three versions with no dates and could not tell.

More detail: [ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md).

---

## Project structure

```
backend/    FastAPI app
  routers/    ingest, query, conflicts, analytics, health
  services/   query_analyzer, retriever, temporal_reranker, conflict_detector,
              belief_revision, version_resolver, context, llm, ...
  tests/      unit and end-to-end pipeline tests (no DB or network needed)
frontend/   React + Vite: ingest, library, query and conflict-review pages
evaluation/ benchmark: 5 policy versions, 30 questions, run script
```

---

## Setup

You need Python 3.10+, Node.js 18+, PostgreSQL, and a
[Groq](https://console.groq.com) API key.

```bash
# 1. database
psql postgres -c "CREATE DATABASE temporal_rag;"

# 2. config: copy .env.example to .env, then set DB_PASSWORD and GROQ_API_KEY
cp .env.example .env

# 3. backend  (http://localhost:8000)
cd backend
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# 4. frontend  (http://localhost:5173), in a second terminal
cd frontend
npm install
npm run dev
```

Tables are created automatically. The first start downloads the embedding and
NLI models (~170 MB). `requirements.txt` pins CUDA builds of PyTorch; without an
NVIDIA GPU, install the CPU build first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

**Try it:** upload a document as version `1.0` with an old date, then upload an
edited copy as a *new version* (pick the first as its parent) with a newer date
and a changed fact. Ask about it now, as of a past date, and "according to
version 1.0". Tick "Compare against standard RAG" to see the difference.

If the search indexes ever get out of sync with the database, rebuild them with
`python scripts/rebuild_indexes.py` (run from `backend/`).

---

## Tests

```bash
cd backend
python -m pytest
```

Covers version lineage, the analyzer's output validation, filtering for every
intent, conflict handling, prompt building, and the full query endpoint with
the search indexes, NLI model and LLMs faked.

---

## Evaluation

[`evaluation/`](evaluation/README.md) compares the full system against the same
system with temporal reasoning turned off (`no_temporal=true`). It uses five
versions of one policy document and 30 questions (current, point-in-time,
historical, changed-value and never-changing). An LLM judge grades each answer
against a reference answer.

**Initial run:**

| | Correct | Accuracy |
|---|---|---|
| Normal RAG (temporal off) | 9 / 30 | 30% |
| Temporal RAG | 27 / 30 | 90% |
| **Improvement** | | **+60 percentage points** |

The gain comes from current and point-in-time questions, where normal RAG sees
several near-identical versions with no dates and cannot tell which applies. On
the questions whose answers never change, both systems scored 5/5. This is one
run on one synthetic document with 30 questions, and answers vary a little
between runs, so treat it as indicative, not exact.

It runs against its own database, so it never touches your documents. Setup
steps are in [`evaluation/README.md`](evaluation/README.md).

---

## Limitations

- Time comes from the version date you enter at upload, not from dates written
  inside the text.
- One LLM call interprets each question. A question mixing two time frames
  ("is it higher than in 2023?") may be misread.
- Conflicts are only checked between different documents, among the top 20
  results.
- No document deletion, no authentication (single local user).
- The recency half-life (180 days, `TEMPORAL_HALF_LIFE_DAYS`) and the 90-day
  rule in belief revision are fixed heuristics, not tuned values.

---

## Main API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/ingest` | Upload a document version |
| POST | `/api/v1/query` | Ask a question (`no_temporal: true` = plain RAG) |
| GET | `/api/v1/documents` | List documents |
| GET | `/api/v1/documents/{id}/lineage` | Version chain of a document |
| GET | `/api/v1/conflicts` | Detected contradictions |
| POST | `/api/v1/conflicts/{id}/resolve` | Record a manual resolution |

Interactive docs at <http://localhost:8000/docs>.
