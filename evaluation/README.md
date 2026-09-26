# Evaluation: Temporal RAG vs the same system without temporal reasoning

**Question this answers:** on one versioned document and 12 questions covering
every temporal intent the system implements, how much more often does the
temporal pipeline answer correctly than the same system with temporal reasoning
switched off?

This is a **small evaluation set on purpose**: 12 questions, one synthetic
document, one run per report. Each question costs three LLM calls per arm
(intent analysis, answer, judge) and the project runs on a free Groq key, so
the set is sized to be affordable and re-runnable rather than statistically
strong. Treat the numbers below as indicative, not as a benchmark result.

```
evaluation/
  README.md            this file
  documents/           five versions of one policy (northwind_policy_v1.md ... v5.md)
  manifest.json        title, version and publication date of each file
  questions.json       the 12 questions + reference answers (ground truth)
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

## The questions — one per temporal intent the system implements

`questions.json` holds 12 questions. Each carries the `type` it is expected to
be read as, a reference `answer`, and the `supporting_versions` the answer comes
from. The five types are exactly the five intents in
`backend/services/query_analyzer.py` — there is no question type the pipeline
does not implement, and no intent without questions.

| Intent | n | What it tests | Example |
|---|---|---|---|
| current | 3 | superseded values must not beat the version in force today | "How many days of annual leave do Northwind employees get per year?" |
| point_in_time | 2 | exactly the version whose validity window contains the date | "What was the notice period for resignation on 2024-03-15?" |
| range | 2 | every version valid at any point in the period, and only those | "What was the maximum business travel expense a manager could approve per trip during 2022?" |
| version | 3 | the version the question names, whatever its date | "According to version 2.0 of the policy, how many days per week could employees work remotely?" |
| historical | 2 | the earliest/original value, with every version eligible | "How many days per week could employees work remotely when the policy was first issued?" |

## Measured result

One run, 2026-09-25, against the current implementation. Answer and judge model:
`openai/gpt-oss-120b`; `max_chunks` 20 for both arms. Full output:
`results/eval_20260925_191722.json`.

| | Correct | Incorrect | Accuracy |
|---|---|---|---|
| Normal RAG (`no_temporal=true`) | 1 / 12 | 11 | **8.3%** |
| Temporal RAG | 11 / 12 | 1 | **91.7%** |

Per intent (temporal arm):

```
current         3/3
point_in_time   2/2
range           2/2
version         3/3
historical      1/2
```

The retrieved versions show the filter doing the work directly — each question
was answered from exactly the versions its reference answer comes from:

| Question | Read as | Versions used in the answer |
|---|---|---|
| Q01–Q03 | current | 5.0 |
| Q04 (2022-06-15) | point_in_time | 2.0 |
| Q05 (2024-03-15) | point_in_time | 4.0 |
| Q06 (during 2023) | range | 3.0 |
| Q07 (during 2022) | range | 1.0, 2.0 — the period spans a version change |
| Q08 / Q09 / Q10 | version | 2.0 / 4.0 / 1.0 |
| Q12 | historical | all five |

### The one failure, honestly

**Q11** ("How many days per week could employees work remotely when the policy
was first issued?") was wrong in the temporal arm, and not because of the
filter or the ranking. The analyzer's LLM call returned `point_in_time` with no
`as_of` date, which the schema correctly rejects; `analyze()` then fell back to
`intent=current`, so the question was answered from v5.0 ("2 days") instead of
v1.0 ("1 day"). The run printed this as a warning and `results/*.json` records
`analysis_source: "default"` with the validation error.

This is the documented fallback behaviour working as designed — a bad parse can
only change *which window is searched*, never invent a fact — but it is also a
real limitation: "when the policy was first issued" is a moment in time that has
no date the model can resolve, so it is genuinely ambiguous between
`point_in_time` and `historical`. Q12, phrased as "the earliest ... the company
offered", was read as `historical` and answered correctly.

### Do not compare this to the older 30-question number

An earlier, pre-refactor question set reported 27/30 vs 9/30. **Those numbers
are stale and are not comparable to these**, for two reasons: they were measured
before the reranking refactor, and that set contained five "stable" questions
(facts identical in every version) that plain RAG always answered correctly.
This set has none — every question discriminates between versions — so the
baseline has almost nothing it can get right, and its 8.3% reflects the question
mix as much as the system. The honest reading of the table is *the temporal
pipeline answers version-discriminating questions that plain RAG cannot*, not
*the temporal pipeline is eleven times better*.

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
question's time intent, a filter on each chunk's validity window, conflict
detection and arbitration, and source headers that carry each version's validity
window. Ranking is by relevance in both arms — the temporal pipeline changes
*which* chunks are eligible, not how the survivors are ordered.

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
python evaluation\run_eval.py --url http://localhost:8001 --delay 1 --show-failures
```

Ingest takes a few seconds and can be repeated safely (it does nothing if the
five versions are already there). The evaluation takes a few minutes; a free
Groq key will return HTTP 429 repeatedly and the client backs off and retries,
so raise `--delay` if a run takes too long or errors out.

To start over from scratch: stop the backend, delete `evaluation/eval_data/`,
run `psql postgres -c "DROP DATABASE temporal_rag_eval;"`, and repeat from step 1.

## What you will see

```
Questions:    12
Normal RAG:   X correct, Y incorrect   accuracy XX.X%
Temporal RAG: X correct, Y incorrect   accuracy XX.X%
Improvement:  +XX.X percentage points
```

followed by the questions each side got right that the other missed, a
per-intent table, and warnings if anything makes the numbers unreliable
(analyzer fallbacks, judge failures, errored queries). Every answer, grade, the
intent the analyzer read, and the retrieved versions are saved to
`results/eval_<timestamp>.json`.

## Before you quote a number

- **Say how small it is.** One synthetic document, five versions, 12 questions,
  one run. A single question is 8.3 points, so a one-question difference between
  runs moves the headline number by more than eight points.
- **Re-run it if you can afford the calls**, and report the range rather than a
  single figure. Answers come from an LLM, so runs vary.
- **Read the wrong answers** (`--show-failures`) for both systems, and check the
  analyzer-fallback warning. A run with fallbacks is a run where some questions
  never got temporal understanding at all, which is a different thing from the
  pipeline being wrong.
- **Do not tune the system on these questions and then report the result.** If
  you change code or prompts after looking at failures, say so, or write a
  separate set of questions to tune on.

A truthful sentence for a resume, from the run above:

> Evaluated temporal RAG against the same pipeline with temporal reasoning
> disabled, on a 12-question set covering current, point-in-time, range,
> version and historical queries over a 5-version policy document: 11/12 vs
> 1/12 correct, LLM-judged. Small single-document evaluation.
