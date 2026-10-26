[← Repository README](../README.md) · [← Checkpoint 6.1](../Capstone_Checkpoint_6.1/README.md)

# Capstone Checkpoint 7.1 — Production Readiness: Security and Performance Optimization

Applies safeguards and cost/latency optimizations to the Checkpoint 5.1 agent, targeting the issues found in the
Checkpoint 6.1 audit, and measures each change before vs after.

| File | What it is |
|---|---|
| `capstone_checkpoint_7_1_production_ready_starter.py` | Solution file: sanitizer, scope/refusal rule, refusal-format guard, cache and router, plus the demo and probes |
| `capstone_checkpoint_7_1_production_ready_starter_ORIGINAL.py` | The course starter, unmodified, for comparison |
| `Tarundeep_Required_Capstone_Checkpoint_7_1_Worksheet.docx` | Worksheet (completed) |
| `checkpoint_7_1_output.txt`, `checkpoint_7_1_agent.log` | Output and call log of the demo run |
| [`testset_7_1.md`](testset_7_1.md) / `.csv` / `.json` | The test set as a table: id, category, expected behaviour, question, grading notes, evidence strings |
| [`checkpoint_7_1_testset_results.md`](checkpoint_7_1_testset_results.md) / `.csv` | Per-question results of the last run: category, hardened vs plain pass/fail, path, tokens, seconds, evidence reached, answer |
| `build_testset.py`, `verify_testset.py`, `export_testset_tables.py` | 35-question test set (7 categories x 5), its generator, and a check that every evidence string occurs in the corpus text |
| `run_testset_7_1.py` | Runs the test set on the full corpus; `checkpoint_7_1_testset_results.json/.txt`, `checkpoint_7_1_testset_output.txt` hold the last run |

## What was done

Safeguards and optimizations aimed at the Checkpoint 6.1 findings, run on a three-passage sample of real paper text
(VeriFair, Hectare, Stitch). They are **not yet ported into the 5.1 agent** over the 153 PDFs.

- **Safeguards:** input sanitizer (XML escape, pasted-block removal, injection phrases) feeding retrieval, prompt and
  cache key; XML trust boundary; scope and refusal rule in the answer prompt; code-level refusal-format guard; step cap.
- **Cost:** exact-match cache (refusals never cached) and a router sending simple questions to a one-call direct path.

## Results (one run, `checkpoint_7_1_output.txt`)

| Measure | Result |
|---|---|
| Refusal format on 6.1 probes P1, P5, P7, P10 | 0/4 with the original prompt, 4/4 hardened |
| Cache (Q1) | miss 684 tokens / 1,831 ms; hit 0 tokens / 0 ms |
| Direct path | 229–402 tokens per question vs 684 on the agent path |
| Quality | VeriFair (606 s), Hectare (7x), Stitch still answered correctly |

Limits: small corpus, one run. The router sent Q1 to the agent path because it matches " and ". The raw and sanitized
injection runs gave the same correct answer because the hardened prompt was used in both.

## Run

```bash
python capstone_checkpoint_7_1_production_ready_starter.py   # needs OPENROUTER_API_KEY in .env
```


## Test set on the full corpus

`python run_testset_7_1.py --baseline` runs 35 questions (single-paper facts, cross-paper comparison, trends over time,
multi-turn follow-ups, grounded quotes, not in the collection, red teaming; 5 each) through the 7.1 pipeline over
all 8,087 chunks (`use_full_corpus()` in the solution file), and through the plain direct path for comparison.
Answer questions are graded by an LLM judge against grading notes (grounded quotes must also occur in the retrieved
text); refusal questions must match the required refusal format.

| Category | hardened | plain direct path |
|---|---|---|
| single-paper facts | 5/5 | 5/5 |
| cross-paper comparison | 1/5 | 1/5 |
| trends over time | 0/5 | 0/5 |
| multi-turn follow-ups | 3/5 | 3/5 |
| grounded quotes | 5/5 | 5/5 |
| not in the collection | 5/5 | 5/5 |
| red teaming | 5/5 | 4/5 |
| **all** | **24/35** | **23/35** |

The failures are retrieval, not safeguards: on cross-paper and trend questions only about 30% of the evidence reached
the LLM (100% on single-paper facts and quotes), usually because the question names a paper by its tool name
or needs two papers at once. The hardened path costs more tokens (4,569 vs 2,888 per question) because the router
sends comparisons to the agent loop. One LLM-judged run per path; results vary by a question or two between runs.
Two questions were rewritten after the first run because the full corpus has other correct answers (other Sketch and
SMT papers) that the grading notes did not allow.
