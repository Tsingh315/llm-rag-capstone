r"""Capstone Checkpoint 5.1 — ARCHIVED exploration: agent versions v1-v4 (not the submitted design).

Kept for reference: v1 (6 tools), v2 (schema + answer gate), v3 (graph hops as tools), v4 (4.1 retrieval as one
tool with a re-retrieve loop). Run from this folder with --tag v1|v2|v3|v4. Results: checkpoint_5_1_*.txt/.json here.

Original header: Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools to complete user tasks.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 4. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
#
# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.
#
# ### My scenario: Research Paper Navigator (unchanged since Checkpoint 1.1)
#
# - **Corpus:** 153 research-paper PDFs (mostly program synthesis and programming languages); 148 parsed into
#   8,087 chunks (Checkpoint 2.1). 29 papers exist twice (conference + arXiv), so 119 distinct papers.
# - **Users and tasks:** researchers checking specific claims — a result buried in one paper, or evidence that
#   has to be gathered from several papers — with every claim quoted from a named source.
# - **Built so far:** 2.1 hybrid retriever (BM25 + vectors) → 3.1 LLM-judge evaluation (baseline 1/4) →
#   4.1 paper graph + two-hop retrieval (3/4 at the same 8-chunk budget).
# - **What the agent adds:** it decides each next step from what it has found so far, instead of following one
#   fixed retrieval path.
#
# **The agent's 6 actions**, mapped to the course's suggested ones:
#
# | Action | Course's suggestion | What it does | Built on |
# |---|---|---|---|
# | `search` | semantic search | find which papers/passages are relevant across the collection | 2.1 hybrid retriever |
# | `search_in_paper` | semantic search | find an exact passage inside a paper already identified | 4.1 hop 2 |
# | `read_next` | — | read the passages that follow a chunk (results often sit a few passages later) | 4.1 `next_chunk` |
# | `find_comparisons` | — | papers that compare against tools using a given technique | 4.1 `compares_against` |
# | `clarify` | ask a clarifying question | ask the user when the question is ambiguous (batch mode: state the assumption) | Lab 5.2 |
# | `answer` | — | write the grounded answer from what was collected | 2.1 answer prompt |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this course. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.
#
# **For my real agent (Step 3 below):** the repo venv (Python 3.14; 2.1–4.1 ran on it unchanged) plus
# `pip install -r ../requirements.txt` (adds `langgraph`). It reuses the git-ignored `../data/` built in 2.1–4.1
# (parsed papers, Chroma index, cached tool relations).

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 3
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECIDE_SYSTEM = (
    "You are an agent retrieving from a small document collection. Given the question, "
    "the queries already run, and the documents found so far, decide what to do next. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
    "1-2 new_queries targeting what is still missing (do not repeat past queries)."
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus + a keyword retriever (provided)

# %%
SAMPLE_DOCS = [
    {"id": "d1", "text": "Program synthesis generates programs from a specification, such as input-output examples."},
    {"id": "d2", "text": "The sketching approach lets a programmer leave holes in a program for a synthesizer to fill."},
    {"id": "d3", "text": "Retrieval-augmented generation grounds a model's answers in retrieved documents to reduce hallucination."},
    {"id": "d4", "text": "An agentic retriever decides at each step whether it has enough information or should search again."},
    {"id": "d5", "text": "A tool-using agent chooses among actions — search, look up by date, ask the user — to complete a task."},
    {"id": "d6", "text": "Evaluating an agent compares its answers and cost against a fixed single-pass pipeline."},
]
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def retrieve(query: str, k: int = 2) -> list[str]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = [(d["id"], len(q & set(re.findall(r"[a-z0-9]+", d["text"].lower())))) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, s in scored[:k] if s > 0]


def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
    docs = "\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in collected) or "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(raw)
        return {"done": bool(d.get("done", True)), "new_queries": d.get("new_queries", []) or [],
                "reasoning": d.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError):
        return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


def agentic_answer(llm: ChatOpenAI, question: str) -> str:
    """Provided: a minimal agentic loop — retrieve, decide whether to continue, repeat."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    for step in range(MAX_STEPS):
        for q in pending:
            for doc_id in retrieve(q):
                collected[doc_id] = DOC_BY_ID[doc_id]["text"]
            executed.append(q)
        d = decide(llm, question, collected, executed)
        print(f"  step {step + 1}: have {sorted(collected)}  -> done={d['done']}  ({d['reasoning'][:60]})")
        if d["done"] or not d["new_queries"]:
            break
        pending = d["new_queries"]
    context = "\n\n".join(f"[{i}] {collected[i]}" for i in collected)
    return llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                       HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """The agent design for the Research Paper Navigator (see the Step 1 table for what each tool does)."""
    return {
        "tools": [
            "search",            # find relevant papers/passages across the collection 
            "search_in_paper",   # find an exact passage inside one identified paper 
            "read_next",         # read the passages that follow a chunk 
            "find_comparisons",  # papers that compare against tools using a given technique 
            "clarify",           # ask the user when the question is ambiguous (batch mode: state the assumption)
            "answer",            # write the grounded answer from the collected passages
        ],
        "stop_condition": (
            "Answer when the collected passages contain evidence for every part of the question, with each "
            "claim traceable to a named paper; or when the step limit is reached, a tool returns nothing new "
            "twice in a row, or the plan cannot be parsed. Every failure path stops and answers with what was "
            "found, saying what is missing. Clarify is a pause, not a stop: if the question still has two or more "
            "plausible readings after searching, the agent asks a question that helps it narrow down teh remaining "
            "readings, then resumes planning with the "
            "reply; in batch mode, or if the reply is still ambiguous, it answers under an explicitly stated "
            "assumption instead of asking again."
        ),
        "system_prompt_idea": (
            "Act like a researcher: search to find the paper first, then search inside it or read the passages "
            "that follow to reach the exact result; for 'which papers' questions use find_comparisons; clarify only after searching, when the question still has two or more plausible readings; "
            "answer only from collected passages, quoting each claim with its paper."
        ),
        "test_tasks": [
            # Within a paper: the answer sits a few passages after the section that introduces the experiment.
            "How long did the fairness verifier take to verify the deep recurrent neural network with an "
            "error probability of 10⁻¹⁰, and how many samples did it need?",
            # Across papers: unrelated papers linked only by comparing against an SMT-based tool.
            "Which papers compare their approach against an SMT-based tool and argue that SMT solving limits "
            "scalability? Name the baseline in each.",
            # Comparison: two papers x (baseline, why SMT limits it, result) spread over several passages.
            "Castor and the ECTA-based Haskell synthesizer Hectare each compare against an SMT-based baseline. "
            "For each, name the baseline, explain why the paper says SMT limits it, and give the main result "
            "reported against it.",
        ],
    }


# %% [markdown]
# ## Step 3 — The agent on my real system
#
# Everything below runs on the real corpus. It reuses the Checkpoint 2.1 system (chunks, hybrid search, answer
# prompt) and the Checkpoint 4.1 paper graph (duplicate copies, reading order, `compares_against` relations)
# through imports, so nothing is re-built or re-embedded.
#
# ```
#                ┌────────────────────────── back to PLAN ──────────────────────────┐
#                ▼                                                                  │
# question ──► PLAN ──► search | search_in_paper | read_next | find_comparisons | clarify
#                │
#                └──► answer ──► END
# ```
# PLAN is the hub (Lab 5.2): the LLM returns one JSON action, Python runs it, results go into the shared state
# (Lab 5.1), and control returns to PLAN. Only `answer` ends the run.

# %%
REPO = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]  # archived one level down
sys.path.insert(0, str(REPO / "Capstone_Checkpoint_4.1"))
import capstone_checkpoint_4_1_advanced_retrieval_starter as cp41  # noqa: E402  (also imports the 2.1 system)

RESULTS_PATH = Path.cwd() / "checkpoint_5_1_results.json"

AGENT_MAX_STEPS = 6          # planning decisions before the agent is forced to answer
PLANNER_PASSAGE_CAP = 15     # passages shown (in full) to the planner; the answer step sees all of them
SEARCH_K = 5                 # passages returned by search / find_comparisons
PAPER_K = 3                  # passages returned by search_in_paper
READ_NEXT_N = 2              # passages returned by read_next
ACTIONS = ("search", "search_in_paper", "read_next", "find_comparisons", "clarify", "answer")      # v1, v2
ACTIONS_V3 = ("find_papers", "search_in_paper", "read_next", "clarify", "answer")                # v3
FIND_PAPERS_K = 5            # papers returned by find_papers (v3), one best passage each
ACTIONS_V4 = ("graph_retrieve", "clarify", "answer")                                            # v4

# The system prompt and tool descriptions are the agent's policy: they decide which action it picks at each step.
PLAN_SYSTEM = """You are a research assistant answering questions about a collection of computer-science research \
papers. You work step by step. At each step choose exactly ONE action:

- search {"query": str}: find relevant papers and passages across the whole collection. Use it first, or whenever \
you do not yet know which paper holds the answer. Phrase the query the way a paper would state it.
- search_in_paper {"paper": str, "query": str}: find an exact passage inside ONE paper you have already identified \
(give its title, tool name or a chunk id from it). Use it for a specific number or result buried in a paper.
- read_next {"chunk_id": str}: read the passages that follow a chunk. Use it when a passage introduces an \
experiment or result but stops before the numbers.
- find_comparisons {"technique": str}: find papers that compare their approach against baseline tools using that \
technique (e.g. "SMT solver"), with the passage that names the baseline. Use it for "which papers compare against \
X" questions.
- clarify {"question": str, "assumption": str}: ask the user ONE narrowing question. Only after searching, and \
only if the question still has two or more plausible readings; give the reading you would assume if no answer \
comes.
- answer {}: write the final answer. Choose it when the passages collected contain evidence for every part of \
the question, or when further steps are unlikely to find more.

Rules: do not repeat an action with the same arguments; prefer the most specific tool; never answer from memory.
Respond with ONLY a JSON object: {"action": "<one of the actions>", "args": {...}, "reasoning": "<one sentence: \
what is still missing and why this action>"}"""

ANSWER_AGENT_SYSTEM = cp41.SYSTEM_PROMPT + (
    " Cite the paper title for every claim. If part of the question is not covered by the documents, say which "
    "part is missing. If an assumption about the question's meaning is given, state it at the start."
)


# --- Refinement (v2), from the v1 evaluation: extra rules appended to the v1 policy, plus an enforced output schema.
PLAN_RULES_V2 = """

Additional rules:
- Set "missing" to the parts of the question that the collected passages do not yet support. Do not choose \
answer while "missing" is not empty, unless no tool can find it.
- One sub-question per search: for a question about two papers or tools, search each one separately.
- Pass papers and tools by the name written in the question or a passage (e.g. "Castor"), or by a chunk id you \
have seen. Never invent an id.
- When a passage introduces an experiment or result but the number is not in it, use read_next on that chunk.
- If a tool returns nothing new, change approach (another tool or another query) instead of repeating it.
- Clarify only when the wording of the question is ambiguous, never because evidence is missing."""


class PlanDecision(BaseModel):
    """The planner's decision, enforced as a JSON schema (v2): it cannot reply in prose."""
    action: Literal["search", "search_in_paper", "read_next", "find_comparisons", "clarify", "answer"]
    query: str = Field(description="search / search_in_paper: the query; otherwise empty")
    paper: str = Field(description="search_in_paper: paper title, tool name or chunk id; otherwise empty")
    chunk_id: str = Field(description="read_next: a chunk id seen in the passages; otherwise empty")
    technique: str = Field(description="find_comparisons: the technique; otherwise empty")
    question: str = Field(description="clarify: the narrowing question; otherwise empty")
    assumption: str = Field(description="clarify: the reading to assume if no reply; otherwise empty")
    missing: list[str] = Field(description="parts of the question not yet supported by the passages")
    reasoning: str = Field(description="one sentence: why this action")


class ToolDecision(PlanDecision):
    """Same schema without 'answer': used by the answer gate so the planner must pick a tool."""
    action: Literal["search", "search_in_paper", "read_next", "find_comparisons", "clarify"]


ARG_FIELDS = {"search": ["query"], "search_in_paper": ["paper", "query"], "read_next": ["chunk_id"],
              "find_comparisons": ["technique"], "clarify": ["question", "assumption"], "answer": [],
              "find_papers": ["queries"], "graph_retrieve": ["query"]}

# --- Refinement 2 (v3), from the v2 evaluation: the 4.1 graph retrieval becomes the agent's tools.
# Hop 1 (find the papers) -> find_papers; hop 2 (search inside a paper) -> search_in_paper; neighbour expansion ->
# read_next. The fixed single/multi rule of 4.1 is replaced by the planner deciding which papers to search inside.
PLAN_SYSTEM_V3 = """You are a research assistant answering questions about a collection of computer-science research \
papers. You work step by step. At each step choose exactly ONE action:

- find_papers {"queries"}: find which papers are relevant (hop 1 of the paper graph). Decompose the question into \
2-4 focused sub-queries, phrased the way a paper would state them; papers are scored on passage matches AND the \
"paper X compares against baseline tool Y" relations, summed over the sub-queries. Returns papers (score, matched \
relation) with one short preview passage each. Use it first; call it again with new sub-queries if papers are missing.
- search_in_paper {"paper", "query"}: search inside ONE paper (hop 2) for an exact passage: a number, a result or \
a reason. Give the paper's title, the tool it introduces (e.g. "Castor") or a chunk id from it.
- read_next {"chunk_id"}: read the passages that follow a chunk. Use it when a passage introduces an experiment or \
result but stops before the numbers.
- clarify {"question", "assumption"}: ask the user ONE narrowing question, only if the wording of the question is \
ambiguous (two or more plausible readings), never because evidence is missing.
- answer: write the final answer, only when the passages support every part of the question.

Rules:
- Set "missing" to the parts of the question the collected passages do not yet support; do not answer while it \
is not empty unless no tool can find it.
- find_papers only finds papers: use search_in_paper (and read_next) to get the exact passage from each paper \
whose preview does not yet contain the needed fact. For one paper, go deep; for several, search each of them.
- Pass papers by the name written in the question or a passage, or by a chunk id you have seen. Never invent ids.
- If a tool returns nothing new, change approach instead of repeating it. Never answer from memory."""


class PlanDecisionV3(BaseModel):
    """v3 planner decision: 5 actions; hop 1 / hop 2 of the paper graph are separate tools."""
    action: Literal["find_papers", "search_in_paper", "read_next", "clarify", "answer"]
    queries: list[str] = Field(description="find_papers: 2-4 sub-queries decomposing the question; otherwise empty")
    query: str = Field(description="search_in_paper: the query; otherwise empty")
    paper: str = Field(description="search_in_paper: paper title, tool name or chunk id; otherwise empty")
    chunk_id: str = Field(description="read_next: a chunk id seen in the passages; otherwise empty")
    question: str = Field(description="clarify: the narrowing question; otherwise empty")
    assumption: str = Field(description="clarify: the reading to assume if no reply; otherwise empty")
    missing: list[str] = Field(description="parts of the question not yet supported by the passages")
    reasoning: str = Field(description="one sentence: why this action")


class ToolDecisionV3(PlanDecisionV3):
    """v3 schema without 'answer', for the answer gate."""
    action: Literal["find_papers", "search_in_paper", "read_next", "clarify"]


# --- v4 (simplification): the whole 4.1 graph retrieval is ONE tool; the agent only decides whether anything is
# still missing (Lab 5.1's retrieve -> check -> retrieve-again loop, with the 4.1 retriever as the tool).
PLAN_SYSTEM_V4 = """You are a research assistant answering questions about a collection of computer-science research \
papers. You work step by step. At each step choose exactly ONE action:

- graph_retrieve {"query"}: run the paper-graph retriever. It decomposes the query into sub-queries, finds the \
relevant papers (passage matches and "paper X compares against baseline tool Y" relations), searches inside them, \
and adds the passages that follow; it returns 8 labelled passages. Call it first with the question itself. Call it \
again only for something still missing, with a query aimed at that gap (e.g. name the paper or tool).
- clarify {"question", "assumption"}: ask the user ONE narrowing question, only if the wording of the question is \
ambiguous (two or more plausible readings), never because evidence is missing.
- answer: write the final answer, only when the passages support every part of the question.

Set "missing" to the parts of the question the collected passages do not yet support; do not answer while it is \
not empty unless another retrieval is unlikely to find it. Never answer from memory."""


class PlanDecisionV4(BaseModel):
    """v4 planner decision: one retrieval tool (the 4.1 graph pipeline), clarify, or answer."""
    action: Literal["graph_retrieve", "clarify", "answer"]
    query: str = Field(description="graph_retrieve: the question, or a query aimed at what is missing; otherwise empty")
    question: str = Field(description="clarify: the narrowing question; otherwise empty")
    assumption: str = Field(description="clarify: the reading to assume if no reply; otherwise empty")
    missing: list[str] = Field(description="parts of the question not yet supported by the passages")
    reasoning: str = Field(description="one sentence: why this action")


class ToolDecisionV4(PlanDecisionV4):
    """v4 schema without 'answer', for the answer gate."""
    action: Literal["graph_retrieve", "clarify"]


class CountingLLM:
    """Wraps the chat model to count calls and characters sent/received (cost estimate = characters / 4)."""

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm
        self.calls = 0
        self.chars = 0

    def invoke(self, messages):
        out = self._llm.invoke(messages)
        self.calls += 1
        self.chars += sum(len(m.content) for m in messages) + len(out.content)
        return out

    def structured(self, schema, messages):
        """Schema-enforced call (v2 planner): the model must emit JSON matching `schema`."""
        out = structured_call(self._llm, schema, messages)
        self.calls += 1
        self.chars += sum(len(m.content) for m in messages) + len(out[1])
        return out[0]


def structured_call(llm: ChatOpenAI, schema, messages) -> tuple[Any, str]:
    """Send the JSON schema as a strict response_format, then validate the first JSON object in the reply.
    (The provider sometimes appends characters after the object, which the SDK's own parser rejects.)"""
    fmt = {"type": "json_schema",
           "json_schema": {"name": schema.__name__, "schema": to_strict_json_schema(schema), "strict": True}}
    raw = llm.bind(response_format=fmt).invoke(messages).content
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[raw.find("{"):])
        return schema.model_validate(obj), raw
    except (ValueError, json.JSONDecodeError):
        return None, raw


# %% [markdown]
# ### The tools
#
# Four retrieval tools wrap work from earlier checkpoints; `clarify` and `answer` are handled by the graph.
# All search tools run over canonical papers only (duplicate copies skipped via `duplicate_of`).

# %%
class PaperTools:
    def __init__(self, graph, retriever):
        self.g = graph                      # cp41.PaperGraph (with tool relations)
        self.r = retriever                  # cp41.GraphMultiHopRetriever: BM25 + stored vectors + relation index
        self.tool_owner = {}                # tool name -> paper that introduces it (e.g. "castor" -> Castor's paper)
        for src, dst, d in graph.G.edges(data=True):
            if d["edge_type"] == "introduces":
                self.tool_owner[dst.removeprefix("tool:")] = graph.work_of_file[src.removeprefix("paper:")]

    # --- helpers -------------------------------------------------------------------------------------
    def resolve_paper(self, name: str) -> str | None:
        """Paper from a chunk id, a file name, a tool it introduces, or (fuzzily) its title."""
        name = (name or "").strip()
        if "::" in name and name in self.g.by_id:
            return self.g.work_of_chunk(name)
        stem = name.removesuffix(".pdf")
        for f in self.g.work_of_file:
            if Path(f).stem == stem:
                return self.g.work_of_file[f]
        if name.lower() in self.tool_owner:
            return self.tool_owner[name.lower()]
        key = cp41._norm_title(name)
        for w in self.g.works:
            if key and (key in w or w in key):
                return w
        best = difflib.get_close_matches(key, self.g.works, n=1, cutoff=0.6)
        if best:
            return best[0]
        top = self._top(name, self.r.canon_rows, 1) if name else []   # e.g. "Hectare": the paper that uses the name,
        if top and name.lower() in self.g.by_id[top[0]]["text"].lower():  # but only if the name really appears there
            return self.g.work_of_chunk(top[0])
        return None

    def _top(self, query: str, rows, k: int) -> list[str]:
        fused = self.r._scores(query, rows)
        order = sorted(range(len(rows)), key=lambda i: -fused[i])[:k]
        return [self.r.ids[rows[i]] for i in order]

    # --- the four retrieval tools ---------------------------------------------------------------------
    def search(self, query: str) -> list[tuple[str, str]]:
        return [(c, f"search: {query}") for c in self._top(query, self.r.canon_rows, SEARCH_K)]

    def search_in_paper(self, paper: str, query: str) -> list[tuple[str, str]]:
        w = self.resolve_paper(paper)
        if not w:
            return []
        rows = [self.r.row[c] for c in self.g.canonical_chunks(w)]
        return [(c, f"search_in_paper: {query}") for c in self._top(query, rows, PAPER_K)]

    def read_next(self, chunk_id: str) -> list[tuple[str, str]]:
        out, c = [], chunk_id
        for step in range(1, READ_NEXT_N + 1):
            c = self.g.next_chunk(c) if c in self.g.by_id else None
            if not c:
                break
            out.append((c, f"read_next: passage {step} after {chunk_id}"))
        return out

    def graph_retrieve(self, query: str, llm) -> list[tuple[str, str]]:
        """The whole Checkpoint 4.1 pipeline in one call: its own decomposition + scope (1 LLM call), hop 1, hop 2
        and reading-order neighbours. Returns its 5 PRIMARY + 3 CONTEXT passages with 4.1's labels."""
        self.r.llm = llm                                               # 4.1's planning call is counted too
        res = self.r.retrieve(query)
        out = [(c, f"graph_retrieve PRIMARY {i} (scope {res['scope']})") for i, c in enumerate(res["primary"], 1)]
        return out + [(c, f"graph_retrieve CONTEXT ({why})") for c, why in res["context"]]

    def find_papers(self, queries: list[str]) -> list[tuple[str, str]]:
        """Hop 1 of the 4.1 graph retrieval as a tool: papers scored on passage matches and compares_against
        relations, summed over the sub-queries (as 4.1's _paper_scores). Returns papers only, each with one short
        query-matched preview; choosing the exact passage is left to search_in_paper / read_next (hop 2)."""
        by_chunk, by_rel, rel = self.r._paper_scores(list(queries))
        combined = {w: by_chunk.get(w, 0.0) + by_rel.get(w, 0.0) for w in set(by_chunk) | set(by_rel)}
        order = sorted(combined, key=lambda w: -combined[w])[:3]
        order += [w for w in sorted(by_rel, key=lambda w: -by_rel[w])[:2] if w not in order]  # keep relation hits
        out = []
        for w in order[:FIND_PAPERS_K]:
            preview = self.r._rank_in_paper(w, list(queries))[0]            # best match for the sub-queries
            r = rel.get(w)
            via = f"; relation: {r['text'][:110]}" if r else ""
            out.append((preview, f"find_papers: {self.g.title[w][:60]} (score {combined[w]:.3f}{via})"))
        return out

    def find_comparisons(self, technique: str) -> list[tuple[str, str]]:
        """Relations matching the technique (4.1 relation index); for each paper, the passage naming the baseline."""
        rel = cp41._rrf(self.r.rel_bm25.get_scores(cp41.tokenize(technique)), self.r.rel_emb @ self.r._qvec(technique))
        out, seen = [], set()
        for i in sorted(range(len(rel)), key=lambda i: -rel[i]):
            r = self.g.relations[i]
            if r["work"] in seen:
                continue
            seen.add(r["work"])
            ranked = self.r._rank_in_paper(r["work"], [technique, r["text"]])
            naming = [c for c in ranked if r["baseline"] in self.g.by_id[c]["text"]]
            out.append(((naming or ranked)[0], f"find_comparisons: {r['text'][:120]}"))
            if len(out) == SEARCH_K:
                break
        return out


# %% [markdown]
# ### The agent: state, planner, tool nodes and the LangGraph wiring

# %%
class AgentState(TypedDict):
    question: str
    mode: str                              # "chat" (clarify asks the user) or "batch" (clarify states an assumption)
    steps: int
    history: list                          # [{action, args, reasoning, result}] — what was tried
    passages: dict                         # chunk_id -> how it was found (insertion-ordered)
    observations: list                     # tool output that is not a passage (clarifications)
    assumption: str
    no_new: int                            # consecutive tool steps that added nothing new
    next_action: str
    args: dict
    answer: str


class ResearchAgent:
    def __init__(self, tools: PaperTools, llm: ChatOpenAI, max_steps: int = AGENT_MAX_STEPS, version: str = "v1"):
        self.t = tools
        self.llm = llm                      # replaced per question by a CountingLLM during evaluation
        self.max_steps = max_steps
        self.version = version              # "v1" = initial agent, "v2" = refined after the v1 evaluation
        self.max_no_new = 2 if version == "v1" else 3
        self.actions = {"v3": ACTIONS_V3, "v4": ACTIONS_V4}.get(version, ACTIONS)
        self.graph = self._build()

    # --- PLAN: the hub ---------------------------------------------------------------------------------
    def plan(self, s: AgentState) -> dict:
        if s["steps"] >= self.max_steps or s["no_new"] >= self.max_no_new:  # guards 1 and 2: stop before a call
            return {"next_action": "answer", "args": {}}
        by_id = self.t.g.by_id
        shown = list(s["passages"].items())[-PLANNER_PASSAGE_CAP:]
        label = (lambda how: f" ({how})") if self.version in ("v3", "v4") else (lambda how: "")  # show how found
        passages = "\n\n".join(f"[{cid}]{label(how)} {by_id[cid]['title']}\n{by_id[cid]['text']}" for cid, how in shown)
        tried = "\n".join(f"- {h['action']} {json.dumps(h['args'])} -> {h['result']}" for h in s["history"]) or "(none)"
        obs = "\n".join(s["observations"]) or "(none)"
        user = (f"Question: {s['question']}\n\nActions already taken:\n{tried}\n\nOther observations:\n{obs}\n\n"
                f"Passages collected ({len(s['passages'])} total, showing the latest {len(shown)}):\n{passages or '(none)'}")
        if self.version in ("v2", "v3", "v4"):
            action, args, reasoning = self._decide_v2(s, user)
        else:
            action, args, reasoning = self._decide_v1(user)
        if action not in self.actions or self._missing_args(action, args):  # incoherent plan -> answer
            action, args = "answer", {}
        if any(h["action"] == action and h["args"] == args for h in s["history"]):
            action, args = "answer", {}                                 # repeating itself -> answer
        print(f"  [plan {s['steps'] + 1}] {action} {json.dumps(args)[:90]}  — {reasoning[:110]}")
        return {"next_action": action, "args": args, "steps": s["steps"] + 1,
                "history": s["history"] + [{"action": action, "args": args, "reasoning": reasoning, "result": ""}]}

    def _decide_v1(self, user: str) -> tuple[str, dict, str]:
        """Initial version: JSON requested in the prompt and parsed by hand."""
        raw = self.llm.invoke([SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=user)]).content.strip()
        try:
            d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            action, args = d.get("action", "answer"), d.get("args") or {}
            reasoning = d.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):                     # guard 3: unparseable plan -> answer
            action, args, reasoning = "answer", {}, f"unparseable plan: {raw[:160]!r}"
        return action, args, reasoning

    def _decide_v2(self, s: AgentState, user: str) -> tuple[str, dict, str]:
        """Refined versions (v2-v4): schema-enforced decision, and an answer gate on the 'missing' list."""
        plan_schema, tool_schema, prompt = {
            "v2": (PlanDecision, ToolDecision, PLAN_SYSTEM + PLAN_RULES_V2),
            "v3": (PlanDecisionV3, ToolDecisionV3, PLAN_SYSTEM_V3),
            "v4": (PlanDecisionV4, ToolDecisionV4, PLAN_SYSTEM_V4)}[self.version]
        messages = [SystemMessage(content=prompt), HumanMessage(content=user)]
        call = self.llm.structured if hasattr(self.llm, "structured") else (
            lambda schema, m: structured_call(self.llm, schema, m)[0])
        d = call(plan_schema, messages)
        if d is None:                                                    # still unparseable -> answer
            return "answer", {}, "no valid decision"
        d.missing = d.missing + self._uncovered_tools(s)                # coverage check: don't trust "missing: []"
        if d.action == "answer" and d.missing and s["steps"] < self.max_steps - 1:
            # Answer gate (enforced in code): it says parts are missing and steps remain, so it may not answer yet;
            # the second call uses a schema without 'answer', so it must choose a tool.
            print(f"    [gate] answer blocked, missing: {d.missing}")
            messages.append(HumanMessage(content=f"You listed missing parts: {d.missing}. Answering is not allowed "
                                                 "yet: choose the tool most likely to find them."))
            d = call(tool_schema, messages) or d
        args = {k: getattr(d, k) for k in ARG_FIELDS[d.action] if getattr(d, k)}
        return d.action, args, f"{d.reasoning} | missing: {d.missing}"

    def _uncovered_tools(self, s: AgentState) -> list[str]:
        """Tools the question names that the graph knows (e.g. "Castor") whose paper has no collected passage yet."""
        have = {self.t.g.work_of_chunk(c) for c in s["passages"]}
        named = {w.lower() for w in re.findall(r"[A-Za-z][\w+-]{2,}", s["question"])} & set(self.t.tool_owner)
        how = "graph_retrieve with a query naming {n}" if self.version == "v4" else "search_in_paper with paper='{n}'"
        return [f"passages from '{self.t.g.title[self.t.tool_owner[n]]}', the paper that introduces {n} "
                f"(use {how.format(n=n)})" for n in sorted(named) if self.t.tool_owner[n] not in have]

    @staticmethod
    def _missing_args(action: str, args: dict) -> bool:
        need = {"search": ["query"], "search_in_paper": ["paper", "query"], "read_next": ["chunk_id"],
                "find_comparisons": ["technique"], "clarify": ["question"], "find_papers": ["queries"],
                "graph_retrieve": ["query"]}
        return any(not args.get(k) or (isinstance(args[k], str) and not args[k].strip())
                   for k in need.get(action, []))

    # --- tool nodes --------------------------------------------------------------------------------------
    def _passage_step(self, s: AgentState, found: list[tuple[str, str]]) -> dict:
        passages = dict(s["passages"])
        new = [(c, how) for c, how in found if c not in passages]
        passages.update(new)
        history = list(s["history"])
        note = " — nothing new: try another tool (e.g. read_next) or another query" if (
            not new and self.version == "v2") else ""                    # v2 tells the planner instead of only counting
        history[-1] = {**history[-1], "result": f"{len(found)} passages, {len(new)} new: {[c for c, _ in new]}{note}"}
        return {"passages": passages, "history": history, "no_new": 0 if new else s["no_new"] + 1}

    def search(self, s):            return self._passage_step(s, self.t.search(s["args"]["query"]))
    def search_in_paper(self, s):   return self._passage_step(s, self.t.search_in_paper(s["args"]["paper"], s["args"]["query"]))
    def read_next(self, s):         return self._passage_step(s, self.t.read_next(s["args"]["chunk_id"]))
    def find_comparisons(self, s):  return self._passage_step(s, self.t.find_comparisons(s["args"]["technique"]))
    def find_papers(self, s):       return self._passage_step(s, self.t.find_papers(s["args"]["queries"]))
    def graph_retrieve(self, s):    return self._passage_step(s, self.t.graph_retrieve(s["args"]["query"], self.llm))

    def clarify(self, s):
        q, assume = s["args"]["question"], s["args"].get("assumption", "")
        if s["mode"] == "chat" and not s["assumption"]:
            reply = input(f"\nAgent asks: {q}\nYou: ").strip()
            note = f"Clarification — asked: {q} | user replied: {reply}"
            assumption = f"(user clarified: {reply})"
        else:                                                            # batch mode, or already clarified once
            note = f"Clarification not possible (batch mode) — assuming: {assume}"
            assumption = assume or "the most common reading"
        history = list(s["history"])
        history[-1] = {**history[-1], "result": note[:160]}
        return {"observations": s["observations"] + [note], "assumption": assumption, "history": history}

    # --- ANSWER (the only exit) --------------------------------------------------------
    def answer(self, s: AgentState) -> dict:
        by_id = self.t.g.by_id
        docs = "\n\n---\n\n".join(f"[{by_id[c]['title']} | {c}]\n{by_id[c]['text']}" for c in s["passages"])
        extra = "\n".join(s["observations"])
        assume = f"Assumption about the question: {s['assumption']}\n\n" if s["assumption"] else ""
        user = (f"{assume}Documents:\n{docs or '(none)'}\n\n"
                + (f"Other findings:\n{extra}\n\n" if extra else "") + f"Question: {s['question']}")
        text = self.llm.invoke([SystemMessage(content=ANSWER_AGENT_SYSTEM), HumanMessage(content=user)]).content
        return {"answer": text}

    # --- graph ---------------------------------------------------------------------------------------------
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self.plan)
        for name in self.actions:
            g.add_node(name, getattr(self, name))
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", lambda s: s["next_action"], {a: a for a in self.actions})
        for name in self.actions:
            if name != "answer":
                g.add_edge(name, "plan")                                  # every tool returns to the hub
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str, mode: str = "batch") -> AgentState:
        init: AgentState = {"question": question, "mode": mode, "steps": 0, "history": [], "passages": {},
                            "observations": [], "assumption": "", "no_new": 0, "next_action": "", "args": {},
                            "answer": ""}
        return self.graph.invoke(init, {"recursion_limit": 4 * self.max_steps + 10})


# %% [markdown]
# ### Evaluation: the agent vs the fixed pipelines on the three test tasks
#
# Q1 and Q2 are the Checkpoint 3.1 questions with grading notes unchanged; T3 is the new comparison task. Each
# item lists the exact answer strings (`evidence`) and the papers that hold them, as in 4.1.

# %%
def my_test_tasks() -> list[dict]:
    q1, q2 = (q for q in cp41.my_eval_set() if q["id"] in ("Q1", "Q2"))
    t3 = {
        "id": "T3",
        "question": my_agent_plan()["test_tasks"][2],
        "grading_notes": "Names Cozy as Castor's baseline and says it uses an SMT solver to verify candidates, which "
                         "does not scale to the TPC-H queries; names Hoogle+ as Hectare's baseline, based on an SMT "
                         "encoding of Petri-net reachability; and gives a result for each (e.g. Castor scales to "
                         "larger queries than Cozy; Hectare solves 43 of 45 benchmarks vs 39 for Hoogle+, or a 7-8x "
                         "average speedup).",
        "evidence": ["uses an SMT solver to verify candidates", "SMT encoding of Petri-net reachability", "43 out of 45"],
        "papers": ["Deductive Optimization of Relational Data Storage", "Searching Entangled Program Spaces"],
    }
    return [q1, q2, t3]


def build_system(llm: ChatOpenAI):
    """Loads the 2.1/4.1 system once and returns the three systems under test."""
    chunks = cp41.load_chunks(cp41.PAPERS_DIR)
    db = cp41.build_or_load_db(chunks, cp41.CHROMA_DIR)
    graph = cp41.load_graph(chunks, llm)
    hybrid = cp41.HybridRetriever(chunks, db)
    retriever = cp41.GraphMultiHopRetriever(graph, db, llm)
    tools = PaperTools(graph, retriever)
    agents = {v: ResearchAgent(tools, llm, version=v) for v in ("v1", "v2", "v3", "v4")}

    def baseline(q, counter):                    # Checkpoint 2.1/3.1 fixed pipeline: one hybrid search, top 5
        scores, context = hybrid.retrievedContext(q, 5)
        ans = counter.invoke([SystemMessage(content=cp41.SYSTEM_PROMPT),
                              HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {q}")]).content
        return {"chunks": list(scores), "answer": ans, "trace": ["search (one fixed hybrid search, top 5)"]}

    def advanced(q, counter):                    # Checkpoint 4.1 fixed pipeline: plan + two hops, 8 chunks
        retriever.llm = counter
        res = retriever.retrieve(q)
        ans = counter.invoke([SystemMessage(content=cp41.GRAPH_SYSTEM), HumanMessage(
            content=f"Documents:\n{retriever.build_context(res)}\n\nQuestion: {q}")]).content
        return {"chunks": res["primary"] + [c for c, _ in res["context"]], "answer": ans,
                "trace": [f"plan scope={res['scope']}", "hop 1 + hop 2 (fixed)"]}

    def agentic(q, counter, version="v1"):       # Checkpoint 5.1 agent (initial v1 or refined v2)
        agent = agents[version]
        agent.llm = counter
        out = agent.run(q)
        return {"chunks": list(out["passages"]), "answer": out["answer"],
                "trace": [f"{h['action']} {json.dumps(h['args'])} -> {h['result']}  ({h['reasoning']})"
                          for h in out["history"]]}

    return {"baseline_2_1": baseline, "advanced_4_1": advanced, "agent": agentic}, agents, graph


def run_evaluation(systems: dict, graph, llm: ChatOpenAI, runs: int, names: list[str], tag: str) -> list[dict]:
    """Sequential on purpose: the agent and retriever hold per-question state."""
    tasks = my_test_tasks()
    records = []
    for run in range(1, runs + 1):
        for name in names:
            for item in tasks:
                counter = CountingLLM(llm)
                t0 = time.perf_counter()
                print(f"\n=== run {run} | {name}{'/' + tag if name == 'agent' else ''} | {item['id']}")
                out = (systems[name](item["question"], counter, tag) if name == "agent"
                       else systems[name](item["question"], counter))
                latency = time.perf_counter() - t0
                verdict = cp41.judge(llm, out["answer"], item["grading_notes"])
                rec = {"run": run, "system": name if name != "agent" else f"agent_{tag}", "id": item["id"],
                       "verdict": verdict, "llm_calls": counter.calls, "est_tokens": round(counter.chars / 4),
                       "latency": round(latency, 1), "n_chunks": len(out["chunks"]), "steps": len(out["trace"]),
                       **cp41.score_retrieval(item, out["chunks"], graph),
                       "trace": out["trace"], "chunks": out["chunks"],
                       "answer": out["answer"]}
                print(f"    {verdict.upper()}  evidence {rec['evidence_found']}/{rec['evidence_total']}  papers "
                      f"{rec['papers_found']}/{rec['papers_total']}  calls {rec['llm_calls']}  ~{rec['est_tokens']} tok  "
                      f"{rec['latency']}s  chunks {rec['n_chunks']}")
                log(f"{rec['system']} {item['id']} run{run} verdict={verdict}",
                    f"Q: {item['question']}\ntrace:\n  " + "\n  ".join(rec["trace"])
                    + f"\nchunks: {rec['chunks']}\nanswer: {rec['answer']}")
                records.append(rec)
    return records


def summarize(records: list[dict]) -> str:
    systems = list(dict.fromkeys(r["system"] for r in records))
    ids = list(dict.fromkeys(r["id"] for r in records))
    head = (f"{'system':<16} {'pass':>8} {'evidence':>9} {'papers':>7} {'calls':>6} {'tokens':>8} {'latency':>8} "
            f"{'chunks':>7} {'steps':>6}   per task (passes/runs)")
    rows = [head]
    for s in systems:
        rs = [r for r in records if r["system"] == s]
        n_runs = len({r["run"] for r in rs})
        mean = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
        per = "  ".join(f"{i} {sum(r['verdict'] == 'pass' for r in rs if r['id'] == i)}/{n_runs}" for i in ids)
        rows.append(f"{s:<16} {sum(r['verdict'] == 'pass' for r in rs) / n_runs:>5.2f}/{len(ids)} "
                    f"{sum(r['evidence_found'] for r in rs) / sum(r['evidence_total'] for r in rs):>8.0%} "
                    f"{sum(r['papers_found'] for r in rs) / sum(r['papers_total'] for r in rs):>7.0%} "
                    f"{mean('llm_calls'):>6.1f} {mean('est_tokens'):>8,.0f} {mean('latency'):>7.1f}s "
                    f"{mean('n_chunks'):>7.1f} {mean('steps'):>6.1f}   {per}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Checkpoint 5.1: agent-based RAG over the research papers.")
    ap.add_argument("--demo", action="store_true", help="the starter's sample-corpus demo")
    ap.add_argument("--ask", default="", help="run the agent on one question (chat mode: it may ask you)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--systems", default="baseline_2_1,advanced_4_1,agent")
    ap.add_argument("--tag", default="v4", choices=["v1", "v2", "v3", "v4"],
                    help="agent version: v1 initial, v2 refined, v3 graph hops as tools, v4 graph retrieval as one tool")
    args = ap.parse_args(argv)

    if args.demo:
        run()
        return
    llm = cp41.make_llm()
    print("Your agent plan:\n" + json.dumps(my_agent_plan(), indent=2, ensure_ascii=False))
    systems, agents, graph = build_system(llm)
    if args.ask:
        out = agents[args.tag].run(args.ask, mode="chat")
        print(f"\nAnswer:\n{out['answer']}")
        log("ASK", f"Q: {args.ask}\ntrace: {out['history']}\nanswer: {out['answer']}")
        return
    names = [n for n in args.systems.split(",") if n]
    records = run_evaluation(systems, graph, llm, args.runs, names, args.tag)
    old = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    keep = [r for r in old if not any(r["system"] == n["system"] for n in records)]
    RESULTS_PATH.write_text(json.dumps(keep + records, indent=1, ensure_ascii=False), encoding="utf-8")
    report = summarize(records)
    print("\n" + report)
    log("SUMMARY", report)


# %% [markdown]
# ## Starter demo — the provided agentic loop on the tiny sample corpus (run with --demo)
#
# (Originally the starter's Step 3; kept unchanged so the provided code still runs.)
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run() -> None:
    llm = make_llm()
    question = "How does an agentic retriever differ from a fixed pipeline, and how is it evaluated?"
    print(f"Checkpoint 5.1 — agentic RAG demo  |  scenario: {SCENARIO}")
    print(f"Question: {question}\n")
    answer = agentic_answer(llm, question)
    print(f"\nAgent answer:\n{answer}\n")
    log("AGENTIC", f"Q: {question}\nA: {answer}")
    try:
        print("Your agent plan:")
        print(json.dumps(my_agent_plan(), indent=2))
    except NotImplementedError as e:
        print(f"[my_agent_plan not done yet] {e}")
    print("=" * 72)
    print("Done. Build this agent for your real system and compare it to your 2.1 baseline.")



# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **Capstone Checkpoint 5.1 worksheet** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Refinements**: Describe any refinements you made to the agent (for example,
#   retrieval strategy, prompting, tool selection, stopping criteria, or other
#   workflow improvements) and explain how they affected system performance.
# 6. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).


if __name__ == "__main__":
    main()
