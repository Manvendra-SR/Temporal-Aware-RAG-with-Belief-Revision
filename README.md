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
   a validity window: `valid_from` to `valid_to`. A version that doesn't extend
   its parent (unknown parent, already-superseded parent, a version or date that
   doesn't move forward) is rejected up front, before the file is parsed,
   chunked and embedded.
2. **Understand the question.** One LLM call reads the question and returns its
   time intent as JSON (table below). The LLM only interprets the question; it
   never chooses which chunks to use.
3. **Retrieve.** Hybrid search: BM25 keyword search + FAISS vector search,
   merged with Reciprocal Rank Fusion. It fetches 3x the needed chunks because
   the next step throws many away.
4. **Filter and rank.** Plain code keeps only the chunks valid for that intent,
   then ranks the survivors by relevance alone. Time decides which passages may
   answer; it never nudges the order, so a question about 2023 isn't pulled
   toward the newest version.
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

A question whose answer doesn't depend on time ("what does the policy cover?")
is a "current" question — the answer should come from the version in force
today, which is what "current" already retrieves.

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

---

## Project structure

```
backend/    FastAPI app
  routers/    ingest, query, conflicts, analytics, health
  services/   query_analyzer, retriever, temporal_reranker, conflict_detector,
              belief_revision, version_resolver, context, llm, ...
  tests/      unit and end-to-end pipeline tests (no DB or network needed)
frontend/   React + Vite: ingest, library, query and conflict-review pages
evaluation/ benchmark: 5 policy versions, 12 questions, run script
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

Covers version lineage (including that an invalid version is rejected before
any parsing, chunking or embedding happens), the analyzer's output validation,
filtering for every intent, conflict handling, prompt building, and the full
query endpoint with the search indexes, NLI model and LLMs faked.

---

## Evaluation

[`evaluation/`](evaluation/README.md) compares the full system against the same
system with temporal reasoning turned off (`no_temporal=true`). It uses five
versions of one policy document and **12 questions — two or three for each of
the five intents** (current, point-in-time, range, version, historical). An LLM
judge grades each answer against a reference answer. The set is small on purpose:
every question costs three LLM calls per arm on a free Groq API key.

**One run, 2026-09-25, against the current implementation:**

| | Correct | Incorrect | Accuracy |
|---|---|---|---|
| Normal RAG (temporal off) | 1 / 12 | 11 | 8.3% |
| Temporal RAG | 11 / 12 | 1 | 91.7% |

Per intent, temporal arm: current 3/3, point_in_time 2/2, range 2/2, version
3/3, historical 1/2. Each question was answered from exactly the versions its
reference answer comes from — v5.0 for "current", v2.0 for "on 2022-06-15",
v1.0+v2.0 for "during 2022", the named version for "according to version 4.0",
all five for "historical".

The single miss was an analyzer fallback, not a retrieval error: the intent call
returned `point_in_time` with no date, failed validation, and the question was
answered as "current". Every wrong answer on the normal side is the same failure
— it sees several near-identical versions with no dates and either picks one or
refuses.

This is **one run on one synthetic document with 12 questions**; a single
question is worth 8.3 points. Treat it as indicative, not exact, and note that
the baseline's low score partly reflects a question set in which every question
discriminates between versions.

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
- The 90-day publication gap that lets belief revision prefer one unrelated
  source over another is a fixed heuristic, not a tuned value.

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
