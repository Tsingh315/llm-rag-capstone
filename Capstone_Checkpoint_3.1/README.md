[← Repository README](../README.md) · [← Checkpoint 2.1](../Capstone_Checkpoint_2.1/README.md)

# Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis

Evaluates the Checkpoint 2.1 hybrid retriever (imported unchanged) with an LLM judge that scores each
answer pass/fail against one-sentence grading notes.

| File | What it is |
|---|---|
| `capstone_checkpoint_3_1_evaluation_starter.py` | Evaluation script: 4 questions with grading notes, LLM judge, framework validation |
| `checkpoint_3_1_evaluation.log` | Retrieved chunks (with scores and chunk IDs), answers, verdicts, pass rate, validation results |
| `Tarundeep_Required_Capstone_Checkpoint_3_1_Worksheet.docx` | Completed worksheet |

**Run** (from this folder, with the repo `.venv` and `.env` set up):
`../.venv/bin/python capstone_checkpoint_3_1_evaluation_starter.py`. Add `SHOW_CHUNK_TEXT=1` to also print each
retrieved chunk's full text. Reads the shared `../data/papers_txt` and `../data/chroma_db` built in Checkpoint 2.1.

**Result:** pass rate 1/4 (passes when the answer is in a paper's abstract; fails when it is deep in a results
section or spread across papers); retrieval hit @5 3/4; framework validation 3/3 known-answer cases correct.
