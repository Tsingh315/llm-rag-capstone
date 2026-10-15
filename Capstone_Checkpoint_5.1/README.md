[← Repository README](../README.md) · [← Checkpoint 4.1](../Capstone_Checkpoint_4.1/README.md)

# Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System

An agent (LangGraph) that decides **whether** a question needs the papers, asks a clarifying question if the
wording is ambiguous, and otherwise runs the **Checkpoint 4.1 graph retrieval in one shot** and answers from
exactly what it returns. Evaluated against the fixed 2.1 and 4.1 pipelines with the 3.1 LLM judge.

| File | What it is |
|---|---|
| `capstone_checkpoint_5_1_agent_rag_starter.py` | Solution file: the agent, its evaluation, and the starter's demo (`--demo`) |
| `capstone_checkpoint_5_1_agent_rag_starter_ORIGINAL.py` | The course starter, unmodified, for comparison |
| `checkpoint_5_1_summary.txt` | Final results: 5 systems × 7 questions × 3 runs |
| `checkpoint_5_1_results.json` | Every run's record (verdict, evidence, papers, calls, tokens, latency, trace, answer) |
| `checkpoint_5_1_agent.log` | Every decision and retrieval of every run |
| `screenshots/` | Figures used in the worksheet (agent code, T3 and Q4 traces, results) |
| `experiments/` | Refinement options A (focus papers) and B (statement-style sub-queries), tested side by side |
| `results_history/` | Earlier result snapshots and run outputs (before the T3 rewording, the extra questions, the focus gate) |
| `exploration_v1_v4/` | Archived earlier agent designs (6 tools; graph hops as tools; re-retrieval loop) and their results |
| `Tarundeep_Required_Capstone_Checkpoint_5_1_Worksheet.docx` | Worksheet |

## The agent

```
question ──► PLAN ──► graph_retrieve ──► answer ──► END     research question: the happy path
                │
                ├──► clarify ──► PLAN                        ambiguous wording: ask once (batch mode: assume)
                └──► answer ──► END                          no retrieval needed (greeting, follow-up)
```

- **PLAN** is one LLM call returning a JSON-schema-enforced decision: `graph_retrieve`, `clarify` or `answer`.
- **`graph_retrieve`** passes the user's question to the 4.1 retriever unchanged: decomposition + scope, hop 1
  (passages and `compares_against` relations), hop 2, reading-order neighbours; 5 PRIMARY + 3 CONTEXT passages.
  At most once per question.
- **Refinement** (`--tag refined`, the final agent): for a question that compares two or more named papers, the
  planner lists them as `focus` and retrieval spends its 8 passages inside those papers. A code check applies it
  only when two or more papers resolve; every other question runs 4.1 unchanged. `--tag initial` runs the agent
  without it.

## Results (7 questions × 3 runs; Q1, Q2, T3 are the worksheet's test tasks, Q3–Q6 the rest of 4.1's set)

| | passes (of 7) | answer evidence reached | LLM calls | latency |
|---|---|---|---|---|
| fixed 2.1 pipeline | 2.00 | 31% | 1 | 2.2 s |
| fixed 4.1 pipeline | 3.67 | 58% | 2 | 5.6 s |
| agent, initial | 3.33 | 58% | 3 | 5.7 s |
| agent, refinement 1 (focus on any named paper) | 2.00 | 56% | 3 | 5.8 s |
| **agent, final** (focus only for multi-paper comparisons) | **3.67** | **69%** | 3 | 6.9 s |

Refinement 1 helped T3 but broke Q4 (the planner focused on DreamCoder, a baseline); the final version restricts
it and adds a code check. The agent ties the fixed 4.1 pipeline on pass rate and reaches more evidence, at one
extra LLM call. Q2 and Q6 fail for every system (grading notes accept only specific papers). Full tables:
`checkpoint_5_1_summary.txt`.

## Run

From this folder, with the repo `.venv` and `.env` set up (reads the shared `../data/` built in 2.1–4.1;
`pip install -r ../requirements.txt` adds `langgraph`):

```bash
python capstone_checkpoint_5_1_agent_rag_starter.py --chat                  # talk to the final agent
python capstone_checkpoint_5_1_agent_rag_starter.py                         # evaluate 2.1, 4.1 and the final agent
python capstone_checkpoint_5_1_agent_rag_starter.py --systems agent --tag initial --tasks Q1,T3
python capstone_checkpoint_5_1_agent_rag_starter.py --demo                  # the starter's sample-corpus demo
```
