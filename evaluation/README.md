# Evaluation: Temporal RAG vs the same system without temporal reasoning

**Question this answers:** on one versioned document and 30 questions about how
it changed over time, how much more often does the temporal pipeline answer
correctly than the same system with temporal reasoning switched off?

The scripts and documents contain no results; each run writes its own file to
`results/`. Quote numbers only from runs you have made and kept.

```
evaluation/
  README.md            this file
  documents/           five versions of one policy (northwind_policy_v1.md ... v5.md)
  manifest.json        title, version and publication date of each file
  questions.json       the 30 questions + reference answers (ground truth)
  ingest_corpus.py     uploads the five versions as one lineage
  run_eval.py          asks every question both ways, grades, prints the score
  results/             one JSON per run: every answer, grade and retrieved version
  eval_data/           the evaluation's own search indexes (created on first run, git-ignored)
```

## The document

**Northwind Technologies Employee Policy**, five versions, one lineage. Every
version is written as if it were the current policy (no "changed from...", no
version or date in the text), the way real handbooks are. Version and date are
supplied at upload, which is how the system is meant to be used.

| Version | Published | Annual leave (days) | Remote (days/wk) | Notice (days) | Parental leave (wks) | Travel limit (INR) | Health cover (INR lakh) |
|---|---|---|---|---|---|---|---|
| 1.0 | 2021-04-01 | 18 | 1 | 60 | 12 | 25,000 | 3 |
| 2.0 | 2022-03-01 | 20 | 2 | 60 | 12 | 25,000 | 5 |
| 3.0 | 2023-01-01 | 20 | 3 | 45 | 12 | 40,000 | 5 |
| 4.0 | 2024-02-01 | 24 | 3 | 45 | 16 | 40,000 | 5 |
| 5.0 | 2025-03-01 | 24 | 2 | 30 | 26 | 60,000 | 10 |

Also stable in every version: head office in Pune, 9:30 AM to 6:00 PM,
6-month probation, 12 public holidays, salary on the last working day.

## The questions

| Type | n | What it tests | Example |
|---|---|---|---|
| current | 6 | old values must not beat the newest | "How many days of annual leave do Northwind employees get per year?" |
| point_in_time | 8 | the version valid on a given date | "What was the notice period on 2024-03-15?" |
| historical | 5 | earlier values, no or partial dates | "What was the earliest health insurance sum insured the company offered?" |
| conflict | 6 | old and new values both matter (change, comparison, before/after) | "How has the notice period changed over time?" |
| stable | 5 | facts that never change: both systems should get these right | "In which city is the head office?" |

Each question in `questions.json` has a reference `answer`.

## How answers are graded

An **LLM judge** (temperature 0) reads the question, the reference answer and
the candidate answer and returns correct or incorrect. It is not told which
system produced the answer. This handles wording and formatting ("24 days" vs
"twenty-four", "60,000" vs "60000").

- **Incorrect:** naming a stale value as *the* answer, hedging between several
  values without committing, leaving out a fact the reference requires, or
  saying the information is unavailable.
- **Fine:** mentioning older values as background, as long as the answer
  clearly commits to the right one.

By default the judge is the same model as the answer model. For a stricter
setup, pass a different one: `--judge-model <name>` (see
`backend/scripts/list_llm_models.py` for what your key can reach). Every verdict
comes with a one-sentence reason, saved in `results/*.json` and printed by
`--show-failures`, so you can check the judge's calls.

## Fairness: what is identical and what differs

**Identical:** documents, embeddings, BM25 + FAISS + RRF retrieval, `max_chunks`
(20), answer model, question text, database.

**The temporal run** (`no_temporal=false`) adds: one LLM call to read the
question's time intent, a filter on each chunk's validity window, intent-aware
ranking, conflict detection and arbitration, and source headers that carry each
version's validity window.

**The normal run** (`no_temporal=true`) retrieves the top chunks by relevance
alone; its sources are labelled only `[SOURCE n | title]`, and its prompt says
nothing about time. So it cannot pick the newest source by reading dates the
temporal pipeline put there. The temporal run costs two LLM calls per question
and the normal run one; `results/*.json` records latency for both.

## How to run it

The evaluation uses its own database and index folder so the five documents
never mix with anything you have ingested. It needs Postgres and a `GROQ_API_KEY`
in `.env`, like the app itself.

**1. Once: create the evaluation database**

```powershell
psql postgres -c "CREATE DATABASE temporal_rag_eval;"
```

**2. Terminal A: start a backend that uses it (port 8001, leaves your normal one alone)**

```powershell
cd backend
venv\Scripts\activate
$env:DB_NAME = "temporal_rag_eval"
$env:DATA_DIR = "..\evaluation\eval_data"
uvicorn main:app --port 8001
```

(macOS/Linux: `export DB_NAME=temporal_rag_eval DATA_DIR=../evaluation/eval_data`.)
Wait for "All services ready".

**3. Terminal B, from the project root: ingest, then evaluate**

```powershell
backend\venv\Scripts\activate
python evaluation\ingest_corpus.py --url http://localhost:8001
python evaluation\run_eval.py --url http://localhost:8001 --show-failures
```

Ingest takes a few seconds and can be repeated safely (it does nothing if the
five versions are already there). The evaluation takes a few minutes; add
`--delay 2` if your Groq key hits rate limits.

To start over from scratch: stop the backend, delete `evaluation/eval_data/`,
run `psql postgres -c "DROP DATABASE temporal_rag_eval;"`, and repeat from step 1.

## What you will see

```
Normal RAG:   XX/30 (XX.X%)
Temporal RAG: XX/30 (XX.X%)
Improvement:  +XX.X percentage points
```

followed by the questions each side got right that the other missed, a
per-type table, and warnings if anything makes the numbers unreliable
(analyzer fallbacks, judge failures, errored queries). Every
answer, grade, the intent the analyzer read, and the retrieved versions are
saved to `results/eval_<timestamp>.json`.

## Before you quote a number

- **Run it about three times and report the range.** Answers come from an LLM
  (temperature 0.1), so single runs vary by a question or two. Keep each
  `results/*.json`.
- **Say how big it is.** It is one synthetic document, five versions, 30
  questions. Two or three questions are 7-10 points, so small gaps mean little.
- **Read the wrong answers** (`--show-failures`) for both systems. Some of the
  gap may come from question types the temporal pipeline is not built for; a
  question that mixes two time frames ("higher or lower than in 2023" asks
  about today and about 2023) is one example. Those are findings, not noise.
- **Do not tune the system on these questions and then report the result.** If
  you change code or prompts after looking at failures, say so, or write a
  separate set of questions to tune on.

A truthful sentence for a resume, once you have your numbers:

> Evaluated temporal RAG against the same pipeline with temporal reasoning
> disabled on 30 questions over a 5-version policy document: X/30 vs Y/30
> correct (+Z percentage points), LLM-judged.
