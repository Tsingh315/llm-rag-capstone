r"""Capstone Checkpoint 5.1 — EXPERIMENT (Option A: focus papers). A copy of the main agent where the planner
names the papers the question asks about and graph_retrieve spends its 8 passages only on those.
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
# - **What the agent adds:** it decides whether a question needs the papers at all, whether it must ask the user
#   first, and only then runs the 4.1 graph retrieval — once — and answers from what it returns.
#
# **The agent's actions**, mapped to the course's suggested ones:
#
# | Action | Course's suggestion | What it does | Built on |
# |---|---|---|---|
# | `graph_retrieve` | semantic search; follow links | the whole 4.1 graph retrieval in one call: decomposition, papers via passages and `compares_against` relations, search inside them, reading-order neighbours | 4.1 |
# | `clarify` | ask a clarifying question | ask one narrowing question if the wording is ambiguous (batch mode: state the assumption) | Lab 5.2 |
# | `answer` | — | answer from the retrieved passages, or directly when no retrieval is needed | 4.1 answer prompt |

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
# `pip install -r ../requirements.txt` (adds `langgraph`). It reuses the git-ignored `../data/` built in 2.1–4.1.

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
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent_option_a.log"

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
    """The agent design for the Research Paper Navigator (see the Step 1 table for what each action does)."""
    return {
        "tools": [
            "graph_retrieve",    # the whole Checkpoint 4.1 graph retrieval in one call, with the user's question
            "clarify",           # ask one narrowing question if the wording is ambiguous (batch mode: state assumption)
            "answer",            # answer from the retrieved passages, or directly when no retrieval is needed
        ],
        "stop_condition": (
            "Answer is the only exit. For a research question the agent calls graph_retrieve once and answers from "
            "exactly what it returns (no second retrieval). Questions that need no papers (a greeting, what the "
            "agent can do, a follow-up already answered) are answered directly. Clarify is a pause, not a stop: "
            "if the wording has two or more plausible readings, the agent asks one narrowing question and plans "
            "again; in batch mode it states an assumption instead. A step limit and an unparseable decision both "
            "fall back to retrieving once and answering."
        ),
        "system_prompt_idea": (
            "Decide first whether the question needs the papers: if it asks about research content, call "
            "graph_retrieve once with the question as written and answer only from what it returns, quoting each "
            "claim with its paper; answer directly only when no paper content is needed; clarify only when the "
            "wording has two or more plausible readings."
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
# ```
# question ──► PLAN ──► graph_retrieve ──► answer ──► END        (research question: the happy path)
#                 │
#                 ├──► clarify ──► PLAN                          (ambiguous wording: ask once, then plan again)
#                 └──► answer ──► END                            (no retrieval needed)
# ```
# PLAN decides **whether retrieval is needed**. If it is, `graph_retrieve` passes the user's question, as written,
# to the Checkpoint 4.1 graph retrieval, which does everything internally in one call (decomposition + scope,
# hop 1, hop 2, reading-order neighbours) and returns 5 PRIMARY + 3 CONTEXT passages. The agent then answers from
# exactly those passages. Retrieval runs at most once per question.

# %%
REPO = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]  # moved to experiments/
sys.path.insert(0, str(REPO / "Capstone_Checkpoint_4.1"))
import capstone_checkpoint_4_1_advanced_retrieval_starter as cp41  # noqa: E402  (also imports the 2.1 system)

RESULTS_PATH = Path.cwd() / "checkpoint_5_1_results_option_a.json"
AGENT_MAX_STEPS = 3          # planning decisions; enough for clarify -> plan -> retrieve

# The system prompt is the agent's policy: it decides when retrieval is needed.
PLAN_SYSTEM = """You are the Research Paper Navigator, an assistant for a collection of about 120 computer-science \
research papers (mostly program synthesis and programming languages). Decide the next action:

- graph_retrieve: search the paper collection with the user's question. Use it for any question about research \
content (papers, tools, methods, results, comparisons). It can be used once; the answer is written from what it \
returns.
- clarify: ask the user ONE narrowing question, only if the question's wording has two or more plausible readings \
(e.g. "Sketch" could be the program-synthesis tool or CAD sketches). Give the reading you would assume if no \
reply comes.
- answer: reply without retrieval, only when the papers are not needed (a greeting, what you can do, or a \
follow-up already answered earlier in the conversation).

Also fill focus: the tools or papers the question explicitly names (e.g. ["Castor", "Hectare"]), written as in the question; leave it empty if the question names none. Retrieval then reads only those papers."""

DIRECT_SYSTEM = ("You are the Research Paper Navigator, an assistant that answers questions about a collection of "
                 "computer-science research papers by searching them. Reply briefly. Do not state facts about the "
                 "papers from memory: if the user needs paper content, say you will search the collection.")


class Decision(BaseModel):
    """The planner's decision, enforced as a JSON schema so it cannot reply in prose."""
    action: Literal["graph_retrieve", "clarify", "answer"]
    question: str = Field(description="clarify: the narrowing question; otherwise empty")
    assumption: str = Field(description="clarify: the reading to assume if no reply; otherwise empty")
    reasoning: str = Field(description="one sentence: why this action")
    focus: list[str] = Field(description="tools or papers explicitly named in the question, e.g. "
                                         "[\"Castor\", \"Hectare\"]; empty if none")


def structured_call(llm: ChatOpenAI, schema, messages) -> tuple[Any, str]:
    """Send the schema as a strict response_format, then validate the first JSON object in the reply
    (the provider sometimes appends characters after it, which the SDK's own parser rejects)."""
    fmt = {"type": "json_schema",
           "json_schema": {"name": schema.__name__, "schema": to_strict_json_schema(schema), "strict": True}}
    raw = llm.bind(response_format=fmt).invoke(messages).content
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[raw.find("{"):])
        return schema.model_validate(obj), raw
    except (ValueError, json.JSONDecodeError):
        return None, raw


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

    def bind(self, **kwargs):                       # used by structured_call
        parent = self

        class _Bound:
            def invoke(self, messages):
                out = parent._llm.bind(**kwargs).invoke(messages)
                parent.calls += 1
                parent.chars += sum(len(m.content) for m in messages) + len(out.content)
                return out
        return _Bound()


# %% [markdown]
# ### The agent: state, nodes and the LangGraph wiring

# %%
class AgentState(TypedDict):
    question: str
    mode: str              # "chat" (clarify asks the user) or "batch" (clarify states an assumption)
    conversation: list     # earlier turns, for follow-ups answered without retrieval
    steps: int
    ask: dict              # the clarifying question and the reading to assume, set by plan
    clarification: str     # the user's reply, or the stated assumption
    retrieval: dict        # the 4.1 retriever's result (empty until graph_retrieve runs)
    focus: list            # Option A: tools/papers the question names (from plan)
    next_action: str
    trace: list            # one line per decision/action, for the log
    answer: str


class ResearchAgent:
    def __init__(self, retriever, llm: ChatOpenAI):
        self.r = retriever                  # cp41.GraphMultiHopRetriever — the Checkpoint 4.1 graph retrieval
        self.llm = llm                      # replaced per question by a CountingLLM during evaluation
        self.graph = self._build()
        g = retriever.g
        self.tool_owner = {}                # tool name -> paper that introduces it (e.g. "castor" -> Castor's paper)
        for src, dst, d in g.G.edges(data=True):
            if d["edge_type"] == "introduces":
                self.tool_owner[dst.removeprefix("tool:")] = g.work_of_file[src.removeprefix("paper:")]

    def resolve_paper(self, name: str) -> str | None:
        """Paper from a chunk id, a file name, a tool it introduces, or (fuzzily) its title; last, the top passage
        for the name, if the name really appears in it (copied from exploration_v1_v4 PaperTools.resolve_paper)."""
        g, r = self.r.g, self.r
        name = (name or "").strip()
        if not name:
            return None
        if "::" in name and name in g.by_id:
            return g.work_of_chunk(name)
        stem = name.removesuffix(".pdf")
        for f in g.work_of_file:
            if Path(f).stem == stem:
                return g.work_of_file[f]
        if name.lower() in self.tool_owner:
            return self.tool_owner[name.lower()]
        key = cp41._norm_title(name)
        for w in g.works:
            if key and (key in w or w in key):
                return w
        best = difflib.get_close_matches(key, g.works, n=1, cutoff=0.6)
        if best:
            return best[0]
        fused = r._scores(name, r.canon_rows)
        top = r.ids[r.canon_rows[int(fused.argmax())]]
        if name.lower() in g.by_id[top]["text"].lower():
            return g.work_of_chunk(top)
        return None

    def focus_retrieve(self, question: str, works: list[str]) -> dict:
        """Option A: 4.1's plan_query for the sub-queries, then the 8-passage budget split across the focus papers
        (hop 2 inside each paper with the question + sub-queries). Same result shape as 4.1's retrieve()."""
        scope, subs = cp41.plan_query(self.r.llm, question)
        queries = [question] + [q for q in subs if q.strip().lower() != question.strip().lower()]
        works = works[:cp41.N_PRIMARY + cp41.N_CONTEXT]
        budget = cp41.N_PRIMARY + cp41.N_CONTEXT
        quota = [budget // len(works) + (i < budget % len(works)) for i in range(len(works))]
        picks = [self.r._rank_in_paper(w, queries)[:k] for w, k in zip(works, quota)]
        order = []                                                      # round-robin by rank across papers
        for rank in range(max(quota)):
            for wi, p in enumerate(picks):
                if rank < len(p):
                    order.append((p[rank], wi, rank))
        primary = [c for c, _, _ in order[:cp41.N_PRIMARY]]
        context = [(c, f"focus paper {self.r.g.title[works[wi]][:60]}, in-paper rank {rank + 1}")
                   for c, wi, rank in order[cp41.N_PRIMARY:]]
        return {"scope": f"focus({scope})", "sub_queries": subs, "papers": [self.r.g.title[w] for w in works],
                "relations": [], "primary": primary, "context": context}

    def plan(self, s: AgentState) -> dict:
        if s["steps"] >= AGENT_MAX_STEPS:                               # guard: never loop
            action, reasoning, q, assume = "graph_retrieve", "step limit", "", ""
            focus = s["focus"]
        else:
            convo = "\n".join(s["conversation"][-6:]) or "(none)"
            clar = f"\nClarification: {s['clarification']}" if s["clarification"] else ""
            d, _ = structured_call(self.llm, Decision, [SystemMessage(content=PLAN_SYSTEM), HumanMessage(
                content=f"Earlier conversation:\n{convo}\n\nQuestion: {s['question']}{clar}")])
            if d is None:                                               # unparseable -> the happy path
                action, reasoning, q, assume = "graph_retrieve", "no valid decision", "", ""
                focus = []
            else:
                action, reasoning, q, assume = d.action, d.reasoning, d.question, d.assumption
                focus = [f for f in d.focus if f.strip()]
            if action == "clarify" and (s["clarification"] or not q):  # clarify at most once
                action = "graph_retrieve"
        print(f"  [plan] {action} focus={focus} — {reasoning[:120]}")
        return {"next_action": action, "steps": s["steps"] + 1, "ask": {"q": q, "a": assume}, "focus": focus,
                "trace": s["trace"] + [f"plan -> {action} focus={focus} ({reasoning})"]}

    def clarify(self, s: AgentState) -> dict:
        ask = s["ask"]
        if s["mode"] == "chat":
            reply = input(f"\nAgent asks: {ask['q']}\nYou: ").strip()
            note = f"user clarified: {reply}"
        else:                                                           # batch mode never blocks
            note = f"assumed: {ask['a'] or 'the most common reading'}"
        return {"clarification": note, "trace": s["trace"] + [f"clarify -> {note}"]}

    def graph_retrieve(self, s: AgentState) -> dict:
        query = s["question"] + (f" ({s['clarification']})" if s["clarification"] else "")
        self.r.llm = self.llm                                           # 4.1's own planning call is counted too
        works, resolved = [], []
        for name in s["focus"]:                                         # Option A: the papers the question names
            w = self.resolve_paper(name)
            resolved.append(f"{name} -> {self.r.g.title[w] if w else None}")
            if w and w not in works:
                works.append(w)
        if works:
            res = self.focus_retrieve(query, works)                     # 8 passages split across the focus papers
        else:
            res = self.r.retrieve(query)                                # the whole 4.1 pipeline, one shot
        n = len(res["primary"]) + len(res["context"])
        print(f"  [graph_retrieve] focus {resolved or '[]'} -> scope {res['scope']}")
        return {"retrieval": res, "trace": s["trace"] + [f"focus resolved: {resolved}"] * bool(resolved) + [
            f"graph_retrieve -> scope {res['scope']}, {n} passages from: {'; '.join(res['papers'])[:200]}"]}

    def answer(self, s: AgentState) -> dict:
        if s["retrieval"]:                                              # answer from what the retrieval returned
            note = f"Clarification: {s['clarification']}\n\n" if s["clarification"] else ""
            msgs = [SystemMessage(content=cp41.GRAPH_SYSTEM), HumanMessage(
                content=f"{note}Documents:\n{self.r.build_context(s['retrieval'])}\n\nQuestion: {s['question']}")]
        else:                                                           # no retrieval needed
            convo = "\n".join(s["conversation"][-6:]) or "(none)"
            msgs = [SystemMessage(content=DIRECT_SYSTEM),
                    HumanMessage(content=f"Earlier conversation:\n{convo}\n\nQuestion: {s['question']}")]
        text = self.llm.invoke(msgs).content
        return {"answer": text, "trace": s["trace"] + ["answer"]}

    def _build(self):
        g = StateGraph(AgentState)
        for name in ("plan", "graph_retrieve", "clarify", "answer"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", lambda s: s["next_action"],
                                {"graph_retrieve": "graph_retrieve", "clarify": "clarify", "answer": "answer"})
        g.add_edge("graph_retrieve", "answer")                          # happy path: retrieve once, then answer
        g.add_edge("clarify", "plan")
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str, mode: str = "batch", conversation: list | None = None) -> AgentState:
        return self.graph.invoke({"question": question, "mode": mode, "conversation": conversation or [],
                                  "steps": 0, "ask": {}, "clarification": "", "retrieval": {}, "focus": [], "next_action": "",
                                  "trace": [], "answer": ""})

    def chat(self) -> None:
        """Interactive loop with memory of earlier turns (follow-ups can be answered without retrieval)."""
        conversation: list[str] = []
        print("Research Paper Navigator agent. Ask a question; 'exit' to stop.\n")
        while (q := input("You: ").strip()).lower() not in {"exit", "quit"}:
            out = self.run(q, mode="chat", conversation=conversation)
            print(f"\nAssistant: {out['answer']}\n")
            conversation += [f"User: {q}", f"Assistant: {out['answer'][:600]}"]


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
    agent = ResearchAgent(retriever, llm)

    def baseline(q, counter):                    # Checkpoint 2.1/3.1 fixed pipeline: one hybrid search, top 5
        scores, context = hybrid.retrievedContext(q, 5)
        ans = counter.invoke([SystemMessage(content=cp41.SYSTEM_PROMPT),
                              HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {q}")]).content
        return {"chunks": list(scores), "answer": ans, "trace": ["fixed: one hybrid search, top 5"]}

    def advanced(q, counter):                    # Checkpoint 4.1 fixed pipeline: plan + two hops, 8 chunks
        retriever.llm = counter
        res = retriever.retrieve(q)
        ans = counter.invoke([SystemMessage(content=cp41.GRAPH_SYSTEM), HumanMessage(
            content=f"Documents:\n{retriever.build_context(res)}\n\nQuestion: {q}")]).content
        return {"chunks": res["primary"] + [c for c, _ in res["context"]], "answer": ans,
                "trace": [f"fixed: plan scope={res['scope']}, hop 1 + hop 2"]}

    def agentic(q, counter):                     # Checkpoint 5.1 agent
        agent.llm = counter
        out = agent.run(q)
        res = out["retrieval"]
        chunks = (res["primary"] + [c for c, _ in res["context"]]) if res else []
        return {"chunks": chunks, "answer": out["answer"], "trace": out["trace"]}

    return {"baseline_2_1": baseline, "advanced_4_1": advanced, "agent": agentic}, agent, graph


def run_evaluation(systems: dict, graph, llm: ChatOpenAI, runs: int, names: list[str], tag: str) -> list[dict]:
    """Sequential on purpose: the agent and the retriever hold per-question state."""
    records = []
    for run in range(1, runs + 1):
        for name in names:
            for item in my_test_tasks():
                counter = CountingLLM(llm)
                t0 = time.perf_counter()
                print(f"\n=== run {run} | {name} | {item['id']}")
                out = systems[name](item["question"], counter)
                latency = time.perf_counter() - t0
                verdict = cp41.judge(llm, out["answer"], item["grading_notes"])
                rec = {"run": run, "system": f"agent_{tag}" if name == "agent" else name, "id": item["id"],
                       "verdict": verdict, "llm_calls": counter.calls, "est_tokens": round(counter.chars / 4),
                       "latency": round(latency, 1), "n_chunks": len(out["chunks"]),
                       **cp41.score_retrieval(item, out["chunks"], graph),
                       "trace": out["trace"], "chunks": out["chunks"], "answer": out["answer"]}
                print(f"    {verdict.upper()}  evidence {rec['evidence_found']}/{rec['evidence_total']}  papers "
                      f"{rec['papers_found']}/{rec['papers_total']}  calls {rec['llm_calls']}  "
                      f"~{rec['est_tokens']} tok  {rec['latency']}s")
                log(f"{rec['system']} {item['id']} run{run} verdict={verdict}",
                    f"Q: {item['question']}\ntrace:\n  " + "\n  ".join(rec["trace"])
                    + f"\nchunks: {rec['chunks']}\nanswer: {rec['answer']}")
                records.append(rec)
    return records


def summarize(records: list[dict]) -> str:
    systems = list(dict.fromkeys(r["system"] for r in records))
    ids = list(dict.fromkeys(r["id"] for r in records))
    rows = [f"{'system':<14} {'pass':>8} {'evidence':>9} {'papers':>7} {'calls':>6} {'tokens':>8} {'latency':>8}"
            "   per task (passes/runs)"]
    for s in systems:
        rs = [r for r in records if r["system"] == s]
        n_runs = len({r["run"] for r in rs})
        mean = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
        per = "  ".join(f"{i} {sum(r['verdict'] == 'pass' for r in rs if r['id'] == i)}/{n_runs}" for i in ids)
        rows.append(f"{s:<14} {sum(r['verdict'] == 'pass' for r in rs) / n_runs:>5.2f}/{len(ids)} "
                    f"{sum(r['evidence_found'] for r in rs) / sum(r['evidence_total'] for r in rs):>8.0%} "
                    f"{sum(r['papers_found'] for r in rs) / sum(r['papers_total'] for r in rs):>7.0%} "
                    f"{mean('llm_calls'):>6.1f} {mean('est_tokens'):>8,.0f} {mean('latency'):>7.1f}s   {per}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Checkpoint 5.1: agent-based RAG over the research papers.")
    ap.add_argument("--demo", action="store_true", help="the starter's sample-corpus demo")
    ap.add_argument("--chat", action="store_true", help="talk to the agent (it may ask you a question)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--systems", default="baseline_2_1,advanced_4_1,agent")
    ap.add_argument("--tag", default="initial", help="label for this agent version in the results")
    args = ap.parse_args(argv)

    if args.demo:
        run()
        return
    llm = cp41.make_llm()
    print("Your agent plan:\n" + json.dumps(my_agent_plan(), indent=2, ensure_ascii=False))
    systems, agent, graph = build_system(llm)
    if args.chat:
        agent.chat()
        return
    records = run_evaluation(systems, graph, llm, args.runs, [n for n in args.systems.split(",") if n], args.tag)
    old = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    keep = [r for r in old if not any(r["system"] == n["system"] for n in records)]
    RESULTS_PATH.write_text(json.dumps(keep + records, indent=1, ensure_ascii=False), encoding="utf-8")
    report = summarize(keep + records)
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
