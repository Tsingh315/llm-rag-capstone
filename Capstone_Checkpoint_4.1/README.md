[← Repository README](../README.md) · [← Checkpoint 3.1](../Capstone_Checkpoint_3.1/README.md)

# Capstone Checkpoint 4.1 — Advanced Retrieval Implementation

Replaces the single-pass 2.1 retriever with **graph-based multi-hop retrieval, seeded by query decomposition**
(Lab 4.2 + Lab 4.1), and re-runs the 3.1 evaluation (same judge, same grading notes) against the baseline at a
matched document count.

| File | What it is |
|---|---|
| `capstone_checkpoint_4_1_advanced_retrieval_starter.py` | Solution file: paper graph, relation extraction, query planning, multi-hop retriever, evaluation of 5 systems |
| `checkpoint_4_1_responses.log` | Run-1 detail for every system and question: sub-queries, scope, hop-1 papers, matched relations, PRIMARY/CONTEXT chunks, answer, verdict |
| `checkpoint_4_1_results.json` | Every run's record (verdict, evidence found, papers found, latency, chunks) |
| `checkpoint_4_1_run_output.txt` | Console output of the full evaluation run (summary tables, judge validation) |
| `screenshots/` | Figures used in the worksheet (graph, code, Q1/Q2 traces, results) |
| `Tarundeep_Required_Capstone_Checkpoint_4_1_Worksheet.docx` | Worksheet |

## What changed

| | Baseline (2.1) | Advanced (4.1) |
|---|---|---|
| Data | flat list of 8,087 chunks | paper graph: paper / file / chunk / tool nodes; `version_of` (29 duplicate papers merged), `has_chunk` + `next_chunk` (reading order), `introduces` + `compares_against` (paper → baseline tool, with technique and claimed limitation) |
| LLM before retrieval | none | one call: 2–4 sub-queries + scope (`single` paper vs `multi` paper) |
| Retrieval | one hybrid search, top 5 | hop 1: chunks (top 50) and relations (top 10) per query, RRF-fused, vote for papers → hop 2: inside the top paper (single) or one passage per paper (multi) |
| Context | 5 chunks, duplicates possible | 5 PRIMARY + 3 CONTEXT, each labelled with how it was reached |

The `compares_against` relations are extracted once by the LLM from each paper's comparison sentences
(~1,700 characters per paper) and cached in `../data/graph_relations.json` (git-ignored); later runs make no
extraction calls.

## Results (mean of 3 runs)

| 3.1 set (Q1–Q4) | baseline k=5 | baseline k=8 | decomposition only | graph only | advanced |
|---|---|---|---|---|---|
| judge pass rate | 1.00/4 | 1.00/4 | 1.67/4 | 1.00/4 | **2.67/4** |
| answer evidence reached the LLM | 22% | 22% | 44% | 33% | **78%** |
| expected papers retrieved | 50% | 50% | 56% | 67% | **78%** |
| LLM calls / latency | 1 / 2.1 s | 1 / 2.1 s | 2 / 4.2 s | 1 / 2.6 s | 2 / 5.1 s |

Q1 (VeriFair 606 s / 28,000 samples) passes 3/3; Q2 (multi-paper SMT baselines) 1/3; Q3 retrieves its answer in
3/3 runs but passes 1/3 (answers often omit the term "thermometer continuations"). Q6 fails for every system: its
grading notes accept only two of the many papers in the corpus that use Sketch. Full tables:
`checkpoint_4_1_run_output.txt`.

## Run

From this folder, with the repo `.venv` and `.env` set up (reads the shared `../data/papers_txt` and
`../data/chroma_db` from Checkpoints 2.1/3.1):

```bash
python capstone_checkpoint_4_1_advanced_retrieval_starter.py                 # full evaluation (3 runs, ~5 min)
python capstone_checkpoint_4_1_advanced_retrieval_starter.py --runs 1 --questions Q1,Q2
python capstone_checkpoint_4_1_advanced_retrieval_starter.py --ask "your question"   # one traced retrieval
python capstone_checkpoint_4_1_advanced_retrieval_starter.py --demo          # the starter's sample-corpus demo
```

The first run extracts the tool relations (~119 small LLM calls) and embeds them once; later runs reuse both.
