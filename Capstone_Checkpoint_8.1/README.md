[← Repository README](../README.md) · [← Checkpoint 7.1](../Capstone_Checkpoint_7.1/README.md)

# Capstone Checkpoint 8.1 — Design Review: Two Production Candidates

For the Module 8 design review. Two candidates for production are built and compared on the same 35-question test set
([`testset_8_1.md`](testset_8_1.md): 7 categories x 5, built in Checkpoint 7.1 and copied here unchanged), each with the 7.1 hardening layer:

- **A: 4.1 advanced pipeline + 7.1** — query plan, graph multi-hop retrieval, one answer call.
- **B: 5.1 agent + 7.1** — the agent plans, then retrieves, clarifies or answers; same graph retrieval.

| File | What it is |
|---|---|
| [`testset_8_1.md`](testset_8_1.md) / `.csv` / `.json` | The 35 test questions with category, expected behaviour, grading notes and evidence strings (identical in content to `Capstone_Checkpoint_7.1/testset_7_1.*`; the runner reads the `.json`) |
| `capstone_checkpoint_8_1_candidates.py` | The merged code: imports 4.1, 5.1 and 7.1 (nothing copied) and wraps each candidate in the shared 7.1 layer |
| `run_candidates_8_1.py` | Runs both candidates on the test set, N runs each; writes the results below |
| `checkpoint_8_1_results.md` | Summary tables: pass per category, evidence, tokens, calls, latency, paths |
| `checkpoint_8_1_results_per_question.csv` | Per question: category, pass count, evidence, tokens, seconds, for A and B |
| `checkpoint_8_1_results.json` | Every record: answer, retrieved chunk ids, tokens in/out, calls, latency |
| `checkpoint_8_1_run_output_part1.txt`, `_part2.txt` | Console output (see "How the run went") |
| `presentation/` | The design-review slides (.pptx) and preparation notes (.docx) |

## Prerequisites and setup

Follow the [repository setup](../README.md#setup) first: virtual environment, `pip install -r requirements.txt`, and a `.env`
with `OPENROUTER_API_KEY` (copy `.env.example`). This checkpoint also needs what Checkpoints 2.1 to 4.1 build:
the corpus in `data/ResearchPapers/`, the parsed text in `data/papers_txt/` (GROBID, needs Docker, see
[2.1](../Capstone_Checkpoint_2.1/README_INGESTION.md)), the vector index in `data/chroma_db/` and the tool relations in
`data/graph_relations.json` (both built on first run of 3.1/4.1). Without them, `run_candidates_8_1.py` stops at load time.
A full run is about 1,000 LLM calls and 25 minutes; `--ids SP1,RT5 --runs 1` is a quick check.

## The merged code

The 7.1 layer around both candidates: input sanitizer, follow-up rewriting for multi-turn, exact-match cache (refusals
never cached), XML trust boundary, scope and refusal rule in the answer prompt, and the refusal-format guard. In B the
5.1 agent's answer step is replaced so the no-retrieval path also gets the scope and refusal rule (the 6.1 finding).
The 5.1 planner is unchanged.

## Results (3 runs x 35 questions, same judge and grading notes for both)

Pass count per category (mean of 3 runs; per-run counts in `checkpoint_8_1_results.md`):

| Category | A: 4.1 + 7.1 | B: 5.1 agent + 7.1 |
|---|---|---|
| Single-paper facts | 5.0 / 5 | 4.7 / 5 |
| Cross-paper comparison | 1.0 / 5 | 2.7 / 5 |
| Trends over time | 0.0 / 5 | 0.7 / 5 |
| Multi-turn follow-ups | 3.3 / 5 | 2.3 / 5 |
| Grounded quotes | 2.7 / 5 | 3.7 / 5 |
| Not in the collection | 5.0 / 5 | 5.0 / 5 |
| Red teaming | 5.0 / 5 | 5.0 / 5 |
| **All** | **22.0 / 35** (22, 22, 22) | **24.0 / 35** (23, 24, 25) |

Cost and speed (means per question; tokens are the cost proxy, no dollar prices applied):

| | A | B |
|---|---|---|
| Evidence reaching the LLM | 63% | 69% |
| Input / output tokens | 3,265 / 142 | 3,548 / 212 |
| LLM calls | 2.4 | 3.4 |
| Latency | 5.6 s | 7.5 s |

Reading it:
- B is better where the question needs several papers (cross-paper evidence reached 63% vs 23%) and on quotes; A is
  better on multi-turn and is cheaper and faster (about 10% fewer tokens, one fewer LLM call).
- Both pass every red-team and not-in-collection question in all 3 runs. B answers red-team prompts without retrieval
  (14 of its paths were `agent:answer`), so those cost about half the tokens of A's.
- Trend questions fail for both (0.0 and 0.7 of 5); evidence reached is 17% for both, so this is retrieval.
- The difference in total passes (22 vs 24) is small next to the run-to-run spread, so it does not settle the choice.
- The 7.1 test-set results (24/35) used the 2.1 hybrid retriever and are a different measurement.

## Known limitations

- Questions are my own and graded by an LLM judge; 3 runs, and the 22.0 vs 24.0 gap is small next to run-to-run variation.
- Red teaming is 5 questions; the 10 Lab 6.1 probes were not re-run on these candidates.
- Trend questions reach 17% of the needed evidence in both candidates, and cross-paper questions 23% in A.
- Tokens are the cost measure; no dollar prices are applied.
- PDF parsing flattens tables and equations (see the repository README).

## Presentation

`presentation/Capstone Design Review Presentation.pptx` (14 slides) is the deck for the recorded presentation, with
`Capstone Design Review Preparation Notes.docx` as the speaker notes. Its numbers come from the files in this folder
and from [`testset_8_1.md`](testset_8_1.md); the slides link to
the results table here.

## How the run went

The first run hung after 200 of 210 question runs (one API call with no timeout), so it was stopped. Runs 1 and 2 of
both systems and run 3 of A were kept; B's run 3 was redone from scratch with a 90-second request timeout
(`--resume`). `part1` is the first console output (including the discarded partial B run 3), `part2` the rerun.

## Run

```bash
cd Capstone_Checkpoint_8.1
../.venv/bin/python run_candidates_8_1.py --runs 3            # ~25 min, ~1,000 LLM calls
../.venv/bin/python run_candidates_8_1.py --ids SP1,RT5 --runs 1      # quick subset
../.venv/bin/python run_candidates_8_1.py --runs 3 --resume   # continue an interrupted run
```
Optional `--price-in` / `--price-out` (dollars per 1M tokens) add a cost column.
