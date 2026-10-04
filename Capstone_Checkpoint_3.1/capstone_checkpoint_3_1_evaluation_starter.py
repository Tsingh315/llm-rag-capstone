r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Reuse the Checkpoint 2.1 hybrid retriever as-is (no copy): put its folder on the import path.
REPO = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(REPO / "Capstone_Checkpoint_2.1"))
from capstone_checkpoint_2_1_baseline_retrieval_starter import (  # noqa: E402
    HybridRetriever, RagAssistant, build_or_load_db, load_chunks,
)

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"

# Shared, git-ignored data built in Checkpoint 2.1 (passed explicitly so nothing is re-embedded).
DATA_DIR = REPO / "data"
PAPERS_DIR = DATA_DIR / "papers_txt"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Off by default. Run with SHOW_CHUNK_TEXT=1 to also print and log the full text of each retrieved chunk,
# e.g. to see whether the passage holding the answer reached the LLM.
SHOW_CHUNK_TEXT = os.getenv("SHOW_CHUNK_TEXT", "0") == "1"

SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## The system under test: the Checkpoint 2.1 hybrid retriever (BM25 + vector, fused)
#
# Replaces the starter's 6-sentence demonstration corpus. Built once and reused by every question.

# %%
def get_system() -> tuple[HybridRetriever, RagAssistant, dict[str, str]]:
    """Returns the retriever, the assistant, and a {chunk_id: chunk text} lookup for SHOW_CHUNK_TEXT."""
    chunks = load_chunks(PAPERS_DIR)
    if not chunks:
        raise SystemExit(f"[setup] no chunks loaded from {PAPERS_DIR}; copy papers_txt/ from Checkpoint 2.1.")
    db = build_or_load_db(chunks, CHROMA_DIR)
    retriever = HybridRetriever(chunks, db)
    chunk_text = {c["chunk_id"]: c["text"] for c in chunks}
    return retriever, RagAssistant(retriever), chunk_text


def retrieved_titles(retriever: HybridRetriever, scores: dict[str, float],
                     chunk_text: dict[str, str] | None = None) -> list[str]:
    """Human-readable retrieval evidence: '0.873  <paper title> (<source file>) [<chunk id>]' per chunk.
    With chunk_text, each line is followed by that chunk's full text, indented."""
    lines = []
    for chunk_id, score in scores.items():
        meta = retriever.describe(chunk_id)
        lines.append(f"{score:.3f}  {meta['title']} ({meta['source']}) [{chunk_id}]")
        if chunk_text is not None:
            lines.append("      | " + chunk_text[chunk_id].replace("\n", "\n      | "))
    return lines


# %% [markdown]
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    # The starter used `"pass" in verdict`, which scores "does not pass" as a PASS. Match the first word only.
    return "pass" if verdict.startswith("pass") else "fail"


# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    
    return [
        {"question": "How long did the fairness verifier take to verify the deep recurrent neural network with an error probability of 10⁻¹⁰, and how many samples did it need?",
         "grading_notes": "Names VeriFair and states 606 seconds with 28,000 samples (from the evaluation section); "
                          "697 seconds from the introduction is acceptable only if the discrepancy is noted."},
        {"question": "Which papers compare their approach against an SMT-based tool and argue that SMT solving limits scalability? Name the baseline in each.",
         "grading_notes": "Identifies at least two of: VeriFair vs FairSquare, Castor vs Cozy, Hectare vs Hoogle+ — "
                          "each with its correct baseline."},
        {"question": "Is there a technique that re-runs a whole computation from the beginning, guided by a log of earlier results, to get back to a paused point? How did its speed compare with a logic-programming language on the queens puzzle?",
         "grading_notes": "Identifies thermometer continuations and states that for N=13 GNU Prolog took about 20 s "
                          "while the slowest ML versions took about 14 s."},
        {"question": "What tool did the paper on top-down synthesis for library learning introduce, and how much faster and more memory-efficient is it than DreamCoder?",
         "grading_notes": "Names the tool Stitch and states it is 3-4 orders of magnitude faster than DreamCoder "
                          "and uses 2 orders of magnitude less memory."},
    ]


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def run_evaluation(retriever: HybridRetriever, assistant: RagAssistant, chunk_text: dict[str, str]) -> None:
    llm = make_llm()                      # the judge
    eval_set = my_eval_set()
    passes = 0
    print(f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}\n")
    for i, item in enumerate(eval_set, 1):
        scores, ans = assistant.query(item["question"])
        hits = retrieved_titles(retriever, scores, chunk_text if SHOW_CHUNK_TEXT else None)
        verdict = judge(llm, ans, item["grading_notes"])
        passes += verdict == "pass"
        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print("  retrieved:\n    " + "\n    ".join(hits))
        print(f"  verdict={verdict.upper()}")
        print(f"  answer: {ans}")
        log(f"Q{i}: {item['question']}",
            "retrieved:\n  " + "\n  ".join(hits) + f"\nverdict={verdict}\nanswer={ans}")
    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")
    log("BASELINE PASS RATE", f"{passes}/{len(eval_set)}")


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%
def validate_framework(assistant: RagAssistant) -> None:
    llm = make_llm()
    item = my_eval_set()[2]               # thermometer continuations: a question the corpus can answer
    _, good = assistant.query(item["question"])
    cases = {
        # (answer, expected verdict). None = record only: the system's real answer has no known-correct verdict.
        # Reference answer: written from the paper (p.26); same wording as the subtle manipulation, correct numbers.
        "reference answer": (
            "This is thermometer continuations. For N=13 on the N-queens benchmark, GNU Prolog took about "
            "20 s while the slowest ML versions took about 14 s.", "pass"),
        "system answer": (good, None),
        # Obvious manipulation, as in the starter: wrong topic entirely.
        "obvious manipulation": (
            "The technique is BM25, a deep neural network that generates images from text prompts.", "fail"),
        # Subtle manipulation: right technique and benchmark, numbers swapped so Prolog looks faster.
        "subtle manipulation": (
            "This is thermometer continuations. On the N-queens puzzle with N=13, GNU Prolog took about "
            "14 s while the slowest ML versions took about 20 s.", "fail"),
    }
    print("\n--- Framework validation ---")
    results = []
    for name, (text, expected) in cases.items():
        verdict = judge(llm, text, item["grading_notes"])
        if expected is None:
            print(f"  {name:<22} -> {verdict.upper():<4} (recorded only)")
            results.append(f"{name}: verdict={verdict} (recorded only)")
            continue
        ok = "OK" if verdict == expected else "MISSED"
        print(f"  {name:<22} -> {verdict.upper():<4} (expected {expected.upper()})  {ok}")
        results.append(f"{name}: verdict={verdict} expected={expected} {ok}")
    print("  The framework works if it PASSES the reference answer and FAILS both manipulated ones.")
    log("FRAMEWORK VALIDATION", "\n".join(results) + f"\nsystem answer={good}")


# %%
retriever, rag_assistant, chunk_text = get_system()   # build the 2.1 system once, share it
run_evaluation(retriever, rag_assistant, chunk_text)
validate_framework(rag_assistant)

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
