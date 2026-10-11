r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (solution).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.

Run (from this folder, with the repo .venv):
    python capstone_checkpoint_4_1_advanced_retrieval_starter.py                  # full evaluation, 3 runs
    python capstone_checkpoint_4_1_advanced_retrieval_starter.py --runs 1 --questions Q1,Q2
    python capstone_checkpoint_4_1_advanced_retrieval_starter.py --ask "your question"   # show one retrieval
    python capstone_checkpoint_4_1_advanced_retrieval_starter.py --demo           # the starter's sample-corpus demo

Reuses the Checkpoint 2.1 system (chunks, BM25 tokenizer, Chroma index) unchanged and the Checkpoint 3.1
judge and grading notes unchanged, so the before/after comparison measures only the retrieval redesign.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# The baseline (Checkpoint 2.1) ranks every chunk in isolation with one hybrid search and keeps the top 5.
# Checkpoint 3.1 showed it finds the right *paper* but not the *evidence* when that evidence is spread out:
# deeper in the same paper (Q1, Q3) or across unrelated papers (Q2). This checkpoint replaces it with:
#
# - **Primary (Lab 4.2): graph-based multi-hop retrieval.** A paper graph (paper / file / chunk / tool nodes)
#   walked in hops: find candidate papers (by passages and by "compares_against" relations) -> search inside
#   them -> add the best following passages, labelled PRIMARY / CONTEXT.
# - **Supporting (Lab 4.1): query decomposition.** The LLM rewrites the question into sub-queries and plans the
#   retrieval: one paper in depth ("single") or one passage from each of several papers ("multi").
# - **Minor fix:** reciprocal rank fusion (RRF) instead of min-max score blending (removes the 0.500 ties).

# %% [markdown]
# ## Setup
#
# 1. Python 3.11+ (the repo venv runs 3.14; 2.1 and 3.1 ran on it unchanged).
# 2. `pip install -r ../requirements.txt` (adds `networkx` for this checkpoint).
# 3. `OPENROUTER_API_KEY=sk-or-v1-...` in a `.env` file (this folder, the repo root, or your home folder).
# 4. The shared, git-ignored `../data/papers_txt` and `../data/chroma_db` built in Checkpoints 2.1/3.1.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

# Reuse the Checkpoint 2.1 system as-is (no copy): put its folder on the import path.
REPO = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(REPO / "Capstone_Checkpoint_2.1"))
from capstone_checkpoint_2_1_baseline_retrieval_starter import (  # noqa: E402
    SYSTEM_PROMPT, HybridRetriever, build_or_load_db, get_embeddings, load_chunks, make_llm, tokenize,
)

# %%
LLM_MODEL = "openai/gpt-5.4-mini"  # same model as 2.1/3.1 (set inside make_llm)
LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"
RESULTS_PATH = Path.cwd() / "checkpoint_4_1_results.json"

DATA_DIR = REPO / "data"
PAPERS_DIR = DATA_DIR / "papers_txt"
CHROMA_DIR = DATA_DIR / "chroma_db"

SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

# Retrieval budget. The advanced system sends 5 PRIMARY + up to 3 CONTEXT chunks, so it is compared with the
# baseline at k=8 (matched document count) as well as the original k=5.
N_PRIMARY = 5
N_CONTEXT = 3
RRF_K = 60                 # standard reciprocal-rank-fusion constant
HOP1_POOL = 50             # fused candidates per sub-query when looking for papers (2.1 used 20 per retriever)
NEIGHBOUR_WINDOW = 3       # follow next_chunk up to 3 passages after a PRIMARY chunk (Lab 4.2 thread neighbours)
RELATION_POOL = 10         # compares_against relations matched per query in hop 1
MULTI_FROM_CHUNKS = 4      # multi-paper questions: papers taken from the chunk channel ...
MULTI_FROM_RELATIONS = 6   # ... and from the compares_against relation channel

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system over computer-science research "
    "papers. Break the user's question into 2-4 focused sub-queries that together cover everything needed to "
    "answer it. Each sub-query should target a distinct aspect. Use the technical terms a research paper "
    "would use rather than the user's wording. If the question asks for a result, make one sub-query a "
    "declarative sentence written the way the paper would state that result (for example 'the tool terminated "
    "in N seconds and used M samples'). "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

# Same decomposition, plus a retrieval plan: does the answer live in one paper (search it in depth) or in
# several papers (one strong passage from each)?
PLAN_SYSTEM = DECOMPOSE_SYSTEM.replace(
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].',
    'Also decide the scope: "multi" only if the question explicitly asks for several papers (e.g. "which '
    'papers...", "compare papers X and Y"); otherwise "single", including questions about one technique, tool '
    'or result whose paper is not named. Return ONLY a JSON object: '
    '{"scope": "single" or "multi", "sub_queries": ["sub-query 1", "sub-query 2"]}.')

# The 2.1 answer prompt, unchanged, plus one sentence explaining the PRIMARY / CONTEXT labels.
GRAPH_SYSTEM = SYSTEM_PROMPT + (
    " Documents are labelled PRIMARY (retrieved for the question) or CONTEXT (neighbouring passages or related "
    "papers added through the paper graph). Rely mainly on PRIMARY documents and use CONTEXT to complete or "
    "connect them."
)

# Checkpoint 3.1 judge, copied verbatim (importing the 3.1 script would run its whole evaluation).
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def log_response(label: str, prompt: str, response: str) -> None:
    """Same entry format as the starter's log."""
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _chunk_no(chunk_id: str) -> int:
    return int(chunk_id.rsplit("::", 1)[1])


def _rrf(bm25: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Reciprocal rank fusion of two score arrays over the same candidates. BM25 zeros (no shared term) get no
    BM25 contribution. Unlike min-max blending, a chunk found by only one retriever does not get a fixed 0.5."""
    def ranks(scores: np.ndarray) -> np.ndarray:
        r = np.empty(len(scores))
        r[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
        return r
    fused = 1.0 / (RRF_K + ranks(cos))
    has_term = bm25 > 0
    fused[has_term] += 1.0 / (RRF_K + ranks(bm25)[has_term])
    return fused


# %% [markdown]
# ## Step 1 — Data redesign: the paper graph (Lab 4.2 applied to papers)
#
# Lab 4.2 organised emails as email / thread / person / topic nodes. The same ideas map onto papers:
#
# | Lab 4.2 (emails) | This corpus (papers) |
# |---|---|
# | email | **chunk** (a ~1,500-character passage) |
# | thread + sequence number | **paper**, with chunk order as the sequence (`has_chunk`, `next_chunk`) |
# | — | **file**: 29 papers exist twice (conference + arXiv); `version_of` merges them into one paper |
# | person / topic (who / what) | **tool**: the system a paper `introduces` and the baselines it `compares_against`, with the baseline's technique and the limitation the paper claims stored on the edge |
#
# The `compares_against` relation is extracted once by the LLM from each paper's comparison sentences (the
# same role as Lab 4.2's pre-annotated email JSON) and cached in `data/`. Authors and citations were not used:
# the GROBID step kept neither author lists nor reference lists, and one author appears on most papers, so
# author edges would connect almost everything.

# %%
class PaperGraph:
    """Builds the graph once at startup (pure in-memory computation, no API calls), plus lookup tables."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.G = nx.DiGraph()

        # paper (work) <- file (version); one canonical file per paper: the published one, else the longest.
        files_by_work: dict[str, list[str]] = defaultdict(list)
        chunks_by_file: dict[str, list[str]] = defaultdict(list)
        self.title: dict[str, str] = {}
        for c in chunks:
            chunks_by_file[c["source"]].append(c["chunk_id"])
            work = _norm_title(c["title"])
            if c["source"] not in files_by_work[work]:
                files_by_work[work].append(c["source"])
            self.title[work] = c["title"]
        self.work_of_file = {f: w for w, files in files_by_work.items() for f in files}
        self.canonical: dict[str, str] = {}
        for work, files in files_by_work.items():
            self.canonical[work] = max(files, key=lambda f: ("_corr_" not in f, len(chunks_by_file[f])))
        self.chunks_by_file = {f: sorted(ids, key=_chunk_no) for f, ids in chunks_by_file.items()}

        for work, files in files_by_work.items():
            self.G.add_node(f"paper:{work}", node_type="paper", title=self.title[work])
            for f in files:
                self.G.add_node(f"file:{f}", node_type="file", canonical=(f == self.canonical[work]))
                self.G.add_edge(f"file:{f}", f"paper:{work}", edge_type="version_of")
                prev = None
                for cid in self.chunks_by_file[f]:
                    self.G.add_node(f"chunk:{cid}", node_type="chunk", seq=_chunk_no(cid))
                    self.G.add_edge(f"file:{f}", f"chunk:{cid}", edge_type="has_chunk")
                    if prev:
                        self.G.add_edge(f"chunk:{prev}", f"chunk:{cid}", edge_type="next_chunk")
                    prev = cid

        self.works = sorted(files_by_work)

        self.relations: list[dict] = []    # one record per compares_against edge (filled by add_relations)

    def summary(self) -> str:
        n_types = Counter(d["node_type"] for _, d in self.G.nodes(data=True))
        e_types = Counter(d["edge_type"] for _, _, d in self.G.edges(data=True))
        return (f"Paper graph: {self.G.number_of_nodes()} nodes {dict(n_types)}; "
                f"{self.G.number_of_edges()} edges {dict(e_types)}.")

    def add_relations(self, extracted: dict[str, dict]) -> None:
        """Tool nodes and introduces / compares_against edges from the cached LLM extraction."""
        for work, rec in extracted.items():
            own = (rec.get("own_tool") or "").strip()
            if own:
                self.G.add_node(f"tool:{own.lower()}", node_type="tool", name=own)
                self.G.add_edge(f"paper:{work}", f"tool:{own.lower()}", edge_type="introduces")
            for b in rec.get("baselines") or []:
                name = (b.get("name") or "").strip()
                if not name or name.lower() == own.lower():
                    continue
                technique, limitation = b.get("technique") or "", b.get("limitation") or ""
                self.G.add_node(f"tool:{name.lower()}", node_type="tool", name=name)
                self.G.add_edge(f"paper:{work}", f"tool:{name.lower()}", edge_type="compares_against",
                                technique=technique, limitation=limitation)
                subject = own or self.title[work]
                self.relations.append({
                    "work": work, "baseline": name,
                    "text": f"{subject} is compared against the baseline {name}, which uses {technique}. "
                            f"{limitation}".strip()})

    # --- traversal helpers -------------------------------------------------------------------------------
    def work_of_chunk(self, chunk_id: str) -> str:
        return self.work_of_file[self.by_id[chunk_id]["source"]]

    def is_canonical(self, chunk_id: str) -> bool:
        src = self.by_id[chunk_id]["source"]
        return self.canonical[self.work_of_file[src]] == src

    def canonical_chunks(self, work: str) -> list[str]:
        return self.chunks_by_file[self.canonical[work]]

    def next_chunk(self, chunk_id: str) -> str | None:
        for _, tgt, d in self.G.out_edges(f"chunk:{chunk_id}", data=True):
            if d["edge_type"] == "next_chunk":
                return tgt.removeprefix("chunk:")
        return None


# %% [markdown]
# ### Relation extraction (runs once, cached)
#
# Lab 4.2 was given annotated JSON (sender, receivers, topics). Papers have no such metadata, so the LLM
# annotates each paper once: only its comparison sentences are sent (about 1,700 characters per paper), and the
# result is cached in `data/graph_relations.json`, keyed by a hash of the input, so later runs make no calls.

# %%
RELATIONS_PATH = DATA_DIR / "graph_relations.json"
COMPARISON_CUES = re.compile(
    r"compar|baseline|state[- ]of[- ]the[- ]art|outperform|\bversus\b|\bvs\.?\s|than (?:ours|our)|does not scale|"
    r"less scalable|existing (?:tools?|approach(?:es)?|systems?)|prior (?:work|tools?)", re.I)
RELATION_SYSTEM = (
    "You annotate research papers for a knowledge graph. From the paper's title, abstract and comparison "
    "sentences, return ONLY a JSON object: {\"own_tool\": name of the system/tool the paper introduces or null, "
    "\"baselines\": [{\"name\": tool or system the paper experimentally compares against or explicitly "
    "contrasts with, \"technique\": how that baseline works in a few words (e.g. 'SMT solver', 'enumerative "
    "search'), \"limitation\": the paper's stated reason the baseline falls short, or null}]}. Use names exactly "
    "as written in the text. Use an empty list if the paper compares against no named tool."
)


def extract_relations(graph: PaperGraph, llm: ChatOpenAI, workers: int = 8) -> dict[str, dict]:
    cache = json.loads(RELATIONS_PATH.read_text(encoding="utf-8")) if RELATIONS_PATH.exists() else {}

    def excerpt(work: str) -> str:
        text = (PAPERS_DIR / f"{Path(graph.canonical[work]).stem}.txt").read_text(encoding="utf-8", errors="replace")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if COMPARISON_CUES.search(s) and len(s) < 700]
        return f"TITLE: {graph.title[work]}\n\nABSTRACT: {text[:1500]}\n\nCOMPARISON SENTENCES:\n" + \
            "\n".join(sentences)[:9000]

    todo = {}
    for w in graph.works:
        ex = excerpt(w)
        key = hashlib.sha256((RELATION_SYSTEM + ex).encode()).hexdigest()
        if cache.get(w, {}).get("key") != key:
            todo[w] = (ex, key)

    def one(item):
        w, (ex, key) = item
        raw = llm.invoke([SystemMessage(content=RELATION_SYSTEM), HumanMessage(content=ex)]).content
        try:
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except ValueError:
            data = {"own_tool": None, "baselines": []}
        return w, {"key": key, "own_tool": data.get("own_tool"), "baselines": data.get("baselines") or []}

    if todo:
        print(f"Extracting tool relations for {len(todo)} papers (one-time, cached)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for w, rec in pool.map(one, todo.items()):
                cache[w] = rec
        RELATIONS_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return {w: cache[w] for w in graph.works}


def load_graph(chunks: list[dict], llm: ChatOpenAI) -> PaperGraph:
    graph = PaperGraph(chunks)
    graph.add_relations(extract_relations(graph, llm))
    print(graph.summary())
    return graph


# %% [markdown]
# ## Step 2 — Query decomposition (Lab 4.1, adapted)
#
# The LLM splits the question into sub-queries in the papers' own vocabulary. The original question is always
# kept as the first query, and a parsing failure falls back to the question alone.

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    raw = llm.invoke([SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]).content.strip()
    if "[" in raw:
        raw = raw[raw.find("["): raw.rfind("]") + 1]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
            return parsed[:4]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def plan_query(llm: ChatOpenAI, query: str) -> tuple[str | None, list[str]]:
    """Returns (scope, sub-queries); scope is None if the reply cannot be parsed."""
    raw = llm.invoke([SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=query)]).content.strip()
    try:
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        subs = [x for x in data.get("sub_queries", []) if isinstance(x, str)][:4]
        scope = data.get("scope") if data.get("scope") in ("single", "multi") else None
        return scope, subs
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None, []


# %% [markdown]
# ## Step 3 — Graph-based multi-hop retrieval
#
# 1. **Hop 1 — find papers (over-fetch).** Each query (question + sub-queries) is scored with BM25 + vectors, fused
#    with RRF, against every canonical chunk (top 50) and every `compares_against` relation (top 10). Two
#    channels of paper scores result, summed over queries; duplicate copies count as one paper.
# 2. **Hop 2 — narrow down, following the plan.**
#    - *single*: search inside the top paper only; 5 PRIMARY chunks round-robin across the queries; then the
#      passages that follow PRIMARY 1 in reading order (`next_chunk`, up to 3 ahead) as CONTEXT — Lab 4.2's
#      closest thread neighbours.
#    - *multi*: interleave papers from both channels (relations first); the best passage of each of the first 5
#      papers is PRIMARY, of the next 3 is CONTEXT. For a paper reached through a relation, the passage that
#      names the baseline is preferred (following the edge to its evidence), labelled with that relation.
# 3. The result is always 8 chunks (5 PRIMARY + 3 CONTEXT), matching the k=8 baseline.

# %%
class GraphMultiHopRetriever:
    def __init__(self, graph: PaperGraph, db, llm: ChatOpenAI, decompose: bool = True, expand: bool = True):
        self.g = graph
        self.llm = llm
        self.decompose = decompose
        self.expand = expand
        self.ids = [c["chunk_id"] for c in graph.chunks]
        self.row = {cid: i for i, cid in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in graph.chunks])
        stored = db._collection.get(include=["embeddings"])   # embeddings already built in 2.1: no API cost
        emb = np.zeros((len(self.ids), len(stored["embeddings"][0])), dtype=np.float32)
        for cid, vec in zip(stored["ids"], stored["embeddings"]):
            emb[self.row[cid]] = vec
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.canon_rows = np.array([i for i, cid in enumerate(self.ids) if graph.is_canonical(cid)])
        self._embedder = get_embeddings()

        # Relation index: one short text per compares_against edge, searchable like the chunks.
        texts = [r["text"] for r in graph.relations]
        self.rel_bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.rel_emb = self._embed_cached(texts)

    def _embed_cached(self, texts: list[str]) -> np.ndarray:
        """Embeds the relation texts once and caches them next to the relation file."""
        path = DATA_DIR / "graph_relations_emb.npz"
        key = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        if path.exists():
            cached = np.load(path)
            if str(cached["key"]) == key:
                return cached["emb"]
        vecs = np.asarray(self._embedder.embed_documents(texts), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        np.savez(path, key=key, emb=vecs)
        return vecs

    @lru_cache(maxsize=512)
    def _qvec(self, text: str) -> np.ndarray:
        v = np.asarray(self._embedder.embed_query(text), dtype=np.float32)
        return v / np.linalg.norm(v)

    def _scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        bm = self.bm25.get_scores(tokenize(query))[rows]
        cos = self.emb[rows] @ self._qvec(query)
        return _rrf(bm, cos)

    def _paper_scores(self, queries: list[str]) -> tuple[dict, dict, dict]:
        """Hop 1, two channels, each fused with RRF and summed over queries:
        chunk channel    — best matching chunk per paper among the top HOP1_POOL chunks;
        relation channel — best matching compares_against relation per paper among the top RELATION_POOL."""
        by_chunk: dict[str, float] = defaultdict(float)
        by_relation: dict[str, float] = defaultdict(float)
        relation_text: dict[str, str] = {}
        for q in queries:
            fused = self._scores(q, self.canon_rows)
            best: dict[str, float] = {}
            for i in np.argsort(-fused)[:HOP1_POOL]:
                w = self.g.work_of_chunk(self.ids[self.canon_rows[i]])
                best[w] = max(best.get(w, 0.0), float(fused[i]))
            for w, sc in best.items():
                by_chunk[w] += sc
            rel = _rrf(self.rel_bm25.get_scores(tokenize(q)), self.rel_emb @ self._qvec(q))
            best = {}
            for i in np.argsort(-rel)[:RELATION_POOL]:
                r = self.g.relations[i]
                if float(rel[i]) > best.get(r["work"], 0.0):
                    best[r["work"]] = float(rel[i])
                    relation_text.setdefault(r["work"], r)
            for w, sc in best.items():
                by_relation[w] += sc
        return by_chunk, by_relation, relation_text

    def _rank_in_paper(self, work: str, queries: list[str]) -> list[str]:
        rows = np.array([self.row[cid] for cid in self.g.canonical_chunks(work)])
        score = sum(self._scores(q, rows) for q in queries)
        return [self.ids[rows[i]] for i in np.argsort(-score)]

    def retrieve(self, question: str) -> dict:
        scope, subs = plan_query(self.llm, question) if self.decompose else (None, [])
        queries = [question] + [s for s in subs if s.strip().lower() != question.strip().lower()]
        by_chunk, by_relation, relation_text = self._paper_scores(queries)
        combined = {w: by_chunk.get(w, 0.0) + by_relation.get(w, 0.0) for w in set(by_chunk) | set(by_relation)}
        ranked_papers = sorted(combined, key=lambda w: -combined[w])
        if scope is None:   # no plan (graph-only run): one dominant paper means a single-paper question
            top2 = [combined[w] for w in ranked_papers[:2]] + [0.0]
            scope = "single" if top2[0] >= 2 * top2[1] else "multi"

        primary: list[str] = []
        context: list[tuple[str, str]] = []
        if scope == "single":
            # Depth: hop 2 inside the top paper, round-robin across queries, then the passages that follow.
            papers = ranked_papers[:1]
            lists = [self._rank_in_paper(papers[0], [q]) for q in queries]
            cursors = [0] * len(lists)
            while len(primary) < N_PRIMARY and any(c < len(l) for c, l in zip(cursors, lists)):
                for qi, l in enumerate(lists):
                    while cursors[qi] < len(l) and l[cursors[qi]] in primary:
                        cursors[qi] += 1
                    if cursors[qi] < len(l) and len(primary) < N_PRIMARY:
                        primary.append(l[cursors[qi]])
            if self.expand:
                context = self._neighbours(primary)[:N_CONTEXT]
        else:
            # Breadth: interleave the two hop-1 channels, then one passage per paper (hop 2 inside each paper).
            chunk_side = sorted(by_chunk, key=lambda w: -by_chunk[w])[:MULTI_FROM_CHUNKS]
            rel_side = sorted(by_relation, key=lambda w: -by_relation[w])[:MULTI_FROM_RELATIONS]
            papers = []
            for w in [x for pair in zip(rel_side, chunk_side) for x in pair] + rel_side + chunk_side:
                if w not in papers:
                    papers.append(w)
            papers = papers[:N_PRIMARY + N_CONTEXT]
            for i, w in enumerate(papers):
                rel = relation_text.get(w)
                ranked = self._rank_in_paper(w, queries + ([rel["text"]] if rel else []))
                if rel:   # follow the edge to its evidence: prefer a passage that names the baseline
                    naming = [c for c in ranked if rel["baseline"] in self.g.by_id[c]["text"]]
                    ranked = naming or ranked
                best = ranked[0]
                if i < N_PRIMARY:
                    primary.append(best)
                elif self.expand:
                    why = (f"paper reached via relation: {rel['text']}" if rel
                           else "next candidate paper")
                    context.append((best, why[:160]))
        return {"scope": scope, "sub_queries": subs, "papers": [self.g.title[w] for w in papers],
                "relations": [relation_text[w]["text"] for w in papers if w in relation_text],
                "primary": primary, "context": context}

    def _neighbours(self, primary: list[str]) -> list[tuple[str, str]]:
        """Lab 4.2's closest thread neighbours, applied to papers: the passages that follow PRIMARY 1 in reading
        order (then those after PRIMARY 2, ...), up to NEIGHBOUR_WINDOW ahead of each. A result sentence often
        sits a few chunks after the passage that introduces the experiment."""
        out: list[tuple[str, str]] = []
        for i, p in enumerate(primary, 1):
            n = p
            for step in range(1, NEIGHBOUR_WINDOW + 1):
                n = self.g.next_chunk(n)
                if not n:
                    break
                if n not in primary and n not in [c for c, _ in out]:
                    out.append((n, f"passage {step} after PRIMARY {i}"))
        return out

    def build_context(self, result: dict) -> str:
        parts = []
        for i, cid in enumerate(result["primary"], 1):
            c = self.g.by_id[cid]
            parts.append(f"=== PRIMARY {i} [{c['title']} | {c['source']}] ===\n{c['text']}")
        for cid, why in result["context"]:
            c = self.g.by_id[cid]
            parts.append(f"--- CONTEXT ({why}) [{c['title']} | {c['source']}] ---\n{c['text']}")
        return "\n\n".join(parts)


# %% [markdown]
# ## Step 4 — Advanced-retrieval plan (TODO from the starter)

# %%
def my_advanced_plan() -> dict[str, Any]:
    return {
        "technique": "both",
        "node_types": ["paper", "file", "chunk", "tool"],
        "edge_types": ["version_of", "has_chunk", "next_chunk", "introduces", "compares_against"],
        "test_queries": [q["question"] for q in my_eval_set()[:3]],
        "rationale": (
            "The 3.1 failures were evidence spread out within a paper (answer a few chunks after the best match) "
            "and across unrelated papers, made worse by 29 duplicate papers. The graph merges duplicates, lets "
            "retrieval hop from a paper to its passages and neighbours, and links papers to the baseline tools "
            "they compare against, so cross-paper questions can be answered through that relationship; "
            "decomposition gives the graph better starting points and fixes paraphrased wording."
        ),
    }


# %% [markdown]
# ## Step 5 — Re-run the 3.1 evaluation: baseline vs advanced
#
# Q1-Q4 are the Checkpoint 3.1 set with grading notes unchanged (the headline comparison). Q5-Q6 come from the
# golden set in `corpus_analysis/eval_queries.json` as extra evidence. Besides the judge verdict, each item has
# the exact answer sentences (`evidence`) and the papers that hold them (`papers`), so retrieval is scored
# automatically: did the evidence reach the LLM at all?

# %%
def my_eval_set() -> list[dict]:
    return [
        {"id": "Q1",
         "question": "How long did the fairness verifier take to verify the deep recurrent neural network with an error probability of 10⁻¹⁰, and how many samples did it need?",
         "grading_notes": "Names VeriFair and states 606 seconds with 28,000 samples (from the evaluation section); "
                          "697 seconds from the introduction is acceptable only if the discrepancy is noted.",
         "evidence": ["terminated in 606 seconds", "uses 28,000 samples"],
         "papers": ["Probabilistic Verification of Fairness Properties via Concentration"]},
        {"id": "Q2",
         "question": "Which papers compare their approach against an SMT-based tool and argue that SMT solving limits scalability? Name the baseline in each.",
         "grading_notes": "Identifies at least two of: VeriFair vs FairSquare, Castor vs Cozy, Hectare vs Hoogle+ — "
                          "each with its correct baseline.",
         "evidence": ["FairSquare", "Cozy", "Hoogle+"],
         "papers": ["Probabilistic Verification of Fairness Properties via Concentration",
                    "Deductive Optimization of Relational Data Storage", "Searching Entangled Program Spaces"]},
        {"id": "Q3",
         "question": "Is there a technique that re-runs a whole computation from the beginning, guided by a log of earlier results, to get back to a paused point? How did its speed compare with a logic-programming language on the queens puzzle?",
         "grading_notes": "Identifies thermometer continuations and states that for N=13 GNU Prolog took about 20 s "
                          "while the slowest ML versions took about 14 s.",
         "evidence": ["GNU Prolog takes around 20", "slowest ML versions run in 14"],
         "papers": ["Capturing the Future by Replaying the Past Functional Pearl"]},
        {"id": "Q4",
         "question": "What tool did the paper on top-down synthesis for library learning introduce, and how much faster and more memory-efficient is it than DreamCoder?",
         "grading_notes": "Names the tool Stitch and states it is 3-4 orders of magnitude faster than DreamCoder "
                          "and uses 2 orders of magnitude less memory.",
         "evidence": ["3-4 orders of magnitude", "orders of magnitude less memory"],
         "papers": ["Top-Down Synthesis for Library Learning"]},
        {"id": "Q5",
         "question": "On the Hoogle+ benchmark suite alone, what average speedup did the ECTA-based Haskell synthesizer achieve, and how many of the 45 benchmarks did each tool solve?",
         "grading_notes": "States a 7x average (geometric mean) speedup on the Hoogle+ suite (not the abstract's 8x) "
                          "and that Hectare solved 43 of 45 benchmarks while Hoogle+ solved 39.",
         "evidence": ["geometric mean) of 7", "43 out of 45"],
         "papers": ["Searching Entangled Program Spaces"]},
        {"id": "Q6",
         "question": "Which papers use the Sketch synthesis tool as their solver backend, and what language were their implementations written in?",
         "grading_notes": "Identifies both the paper on synthesizing theories of human language (Ellis et al.) and the "
                          "entity-matching rule synthesis paper (RuleSynth), and states both were implemented in "
                          "Python 2.7 calling Sketch.",
         "evidence": ["RuleSynth in Python 2.7", "Python 2.7 scripts that invoke Sketch"],
         "papers": ["Synthesizing theories of human language with Bayesian program induction",
                    "Synthesizing Entity Matching Rules by Examples"]},
    ]


def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    """Checkpoint 3.1 judge, unchanged (first word only, so 'does not pass' is a FAIL)."""
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if verdict.startswith("pass") else "fail"


def _flat(text: str) -> str:
    return " ".join(text.split())


class System:
    """One configuration under test: retrieve -> labelled context -> answer."""

    def __init__(self, name: str, retrieve, system_prompt: str, llm_calls_retrieval: int):
        self.name = name
        self._retrieve = retrieve
        self.prompt = system_prompt
        self.llm_calls_retrieval = llm_calls_retrieval

    def run(self, llm: ChatOpenAI, question: str) -> dict:
        t0 = time.perf_counter()
        r = self._retrieve(question)
        t_retrieve = time.perf_counter() - t0
        answer = llm.invoke([SystemMessage(content=self.prompt),
                             HumanMessage(content=f"Documents:\n{r['context']}\n\nQuestion: {question}")]).content
        r.update(answer=answer, t_retrieve=t_retrieve, t_total=time.perf_counter() - t0)
        return r


def make_systems(chunks, db, graph: PaperGraph, llm: ChatOpenAI) -> dict[str, System]:
    hybrid = HybridRetriever(chunks, db)
    by_id = graph.by_id

    def baseline(k):
        def retrieve(q):
            scores, context = hybrid.retrievedContext(q, k)
            return {"chunks": list(scores), "labels": ["retrieved"] * len(scores), "context": context}
        return retrieve

    def decomposition_only(q):
        """Lab 4.1 as provided: sub-queries through the 2.1 retriever, scores summed, top 8."""
        subs = decompose_query(llm, q) or [q]
        total: dict[str, float] = defaultdict(float)
        for sq in subs:
            for cid, _, s in hybrid.getTopK(sq, 3 * (N_PRIMARY + N_CONTEXT)):
                total[cid] += s
        ids = sorted(total, key=lambda c: -total[c])[:N_PRIMARY + N_CONTEXT]
        context = "\n\n---\n\n".join(f"[{by_id[c]['title']} | {by_id[c]['source']}]\n{by_id[c]['text']}" for c in ids)
        return {"chunks": ids, "labels": ["retrieved"] * len(ids), "context": context, "sub_queries": subs}

    def graph_system(decompose: bool):
        g = GraphMultiHopRetriever(graph, db, llm, decompose=decompose)
        def retrieve(q):
            res = g.retrieve(q)
            ids = res["primary"] + [c for c, _ in res["context"]]
            labels = ["PRIMARY"] * len(res["primary"]) + [f"CONTEXT ({why})" for _, why in res["context"]]
            return {"chunks": ids, "labels": labels, "context": g.build_context(res),
                    "sub_queries": res["sub_queries"], "papers_hop1": res["papers"],
                    "relations": res["relations"], "scope": res["scope"]}
        return retrieve

    return {
        "baseline_k5": System("baseline_k5", baseline(5), SYSTEM_PROMPT, 0),
        "baseline_k8": System("baseline_k8", baseline(8), SYSTEM_PROMPT, 0),
        "decomposition_only": System("decomposition_only", decomposition_only, SYSTEM_PROMPT, 1),
        "graph_only": System("graph_only", graph_system(False), GRAPH_SYSTEM, 0),
        "advanced": System("advanced", graph_system(True), GRAPH_SYSTEM, 1),
    }


def score_retrieval(item: dict, chunk_ids: list[str], graph: PaperGraph) -> dict:
    texts = [_flat(graph.by_id[c]["text"]) for c in chunk_ids]
    found = [e for e in item["evidence"] if any(e in t for t in texts)]
    titles = {_norm_title(graph.by_id[c]["title"]) for c in chunk_ids}
    papers = [p for p in item["papers"] if _norm_title(p) in titles]
    distinct = len({_flat(graph.by_id[c]["text"].split("\n\n", 1)[-1]) for c in chunk_ids})
    return {"evidence_found": len(found), "evidence_total": len(item["evidence"]),
            "papers_found": len(papers), "papers_total": len(item["papers"]),
            "distinct_passages": distinct, "n_chunks": len(chunk_ids)}


def run_evaluation(systems: dict[str, System], graph: PaperGraph, llm: ChatOpenAI, runs: int,
                   question_ids: list[str], workers: int = 4) -> list[dict]:
    items = [q for q in my_eval_set() if not question_ids or q["id"] in question_ids]
    tasks = [(run, name, item) for run in range(1, runs + 1) for name in systems for item in items]

    def one(task):
        run, name, item = task
        sysm = systems[name]
        out = sysm.run(llm, item["question"])
        verdict = judge(llm, out["answer"], item["grading_notes"])
        rec = {"run": run, "system": name, "id": item["id"], "verdict": verdict,
               "llm_calls": sysm.llm_calls_retrieval + 1,
               "t_retrieve": round(out["t_retrieve"], 2), "t_total": round(out["t_total"], 2),
               **score_retrieval(item, out["chunks"], graph),
               "chunks": out["chunks"], "labels": out["labels"], "sub_queries": out.get("sub_queries", []),
               "papers_hop1": out.get("papers_hop1", []), "relations": out.get("relations", []),
               "scope": out.get("scope"),
               "answer": out["answer"]}
        print(f"  run {run}  {name:<19} {item['id']}  {verdict.upper():<4}  "
              f"evidence {rec['evidence_found']}/{rec['evidence_total']}  "
              f"papers {rec['papers_found']}/{rec['papers_total']}  {rec['t_total']:.1f}s", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, tasks))

    question_text = {q["id"]: q["question"] for q in items}
    for rec in records:
        if rec["run"] != 1:
            continue
        lines = [f"{lab:<58} {graph.by_id[c]['title'][:60]} [{c}]" for c, lab in zip(rec["chunks"], rec["labels"])]
        detail = ""
        if rec["scope"]:
            detail += f"scope: {rec['scope']}\n"
        if rec["sub_queries"]:
            detail += "sub-queries: " + json.dumps(rec["sub_queries"], ensure_ascii=False) + "\n"
        if rec["papers_hop1"]:
            detail += "hop-1 papers: " + "; ".join(rec["papers_hop1"]) + "\n"
        if rec["relations"]:
            detail += "matched relations: " + " | ".join(rec["relations"]) + "\n"
        detail += ("retrieved:\n  " + "\n  ".join(lines)
                   + f"\nevidence {rec['evidence_found']}/{rec['evidence_total']}, papers "
                     f"{rec['papers_found']}/{rec['papers_total']}, {rec['t_total']}s\nanswer: {rec['answer']}")
        log_response(f"{rec['system']} {rec['id']} run1 verdict={rec['verdict']}", question_text[rec["id"]], detail)
    return records


def summarize(records: list[dict], ids: list[str], systems: list[str]) -> str:
    """Mean over runs; one row per system."""
    rows = [f"{'system':<19} {'pass rate':>10} {'evidence':>9} {'papers':>7} {'distinct':>9} {'chunks':>7} "
            f"{'LLM calls':>9} {'latency s':>9}"]
    for name in systems:
        rs = [r for r in records if r["system"] == name and r["id"] in ids]
        if not rs:
            continue
        n_runs = len({r["run"] for r in rs})

        def mean(key):
            return sum(r[key] for r in rs) / len(rs)
        rows.append(
            f"{name:<19} {sum(r['verdict'] == 'pass' for r in rs) / n_runs:>6.2f}/{len(ids):<3}"
            f" {sum(r['evidence_found'] for r in rs) / sum(r['evidence_total'] for r in rs):>8.0%}"
            f" {sum(r['papers_found'] for r in rs) / sum(r['papers_total'] for r in rs):>7.0%}"
            f" {mean('distinct_passages'):>9.1f} {mean('n_chunks'):>7.1f} {mean('llm_calls'):>9.1f}"
            f" {mean('t_total'):>9.1f}")
    return "\n".join(rows)


def per_question(records: list[dict], systems: list[str]) -> str:
    ids = sorted({r["id"] for r in records})
    rows = [f"{'':<19} " + " ".join(f"{i:>12}" for i in ids)]
    for name in systems:
        cells = []
        for i in ids:
            rs = [r for r in records if r["system"] == name and r["id"] == i]
            if rs:
                cells.append(f"{sum(r['verdict'] == 'pass' for r in rs)}/{len(rs)} ev"
                             f"{sum(r['evidence_found'] for r in rs) / sum(r['evidence_total'] for r in rs):.0%}")
        rows.append(f"{name:<19} " + " ".join(f"{c:>12}" for c in cells))
    return "\n".join(rows)


def validate_framework(llm: ChatOpenAI) -> str:
    """Re-run the 3.1 known-answer validation on the same judge (Q3 grading notes)."""
    notes = my_eval_set()[2]["grading_notes"]
    cases = {
        "reference answer": ("This is thermometer continuations. For N=13 on the N-queens benchmark, GNU Prolog took "
                             "about 20 s while the slowest ML versions took about 14 s.", "pass"),
        "obvious manipulation": ("The technique is BM25, a deep neural network that generates images from text "
                                 "prompts.", "fail"),
        "subtle manipulation": ("This is thermometer continuations. On the N-queens puzzle with N=13, GNU Prolog took "
                                "about 14 s while the slowest ML versions took about 20 s.", "fail"),
    }
    out = []
    for name, (text, expected) in cases.items():
        v = judge(llm, text, notes)
        out.append(f"{name}: verdict={v} expected={expected} {'OK' if v == expected else 'MISSED'}")
    return "\n".join(out)


# %% [markdown]
# ## Starter demonstration (sample corpus, kept from the starter; run with --demo)

# %%
SAMPLE_DOCS = [
    {"id": "d1", "text": "Paper on retrieval-augmented generation: Grounding LLM answers in retrieved documents reduces hallucination.",
     "topics": ["RAG", "hallucination"], "links": ["d2", "d3"]},
    {"id": "d2", "text": "Study of hallucination in language models: Models fabricate specifics when they lack grounding.",
     "topics": ["hallucination"], "links": ["d1"]},
    {"id": "d3", "text": "Hybrid retrieval combines keyword and vector search to improve recall over either alone.",
     "topics": ["RAG", "retrieval"], "links": ["d1", "d4"]},
    {"id": "d4", "text": "Query decomposition breaks a complex question into sub-queries, improving multi-aspect retrieval.",
     "topics": ["retrieval", "decomposition"], "links": ["d3"]},
    {"id": "d5", "text": "Graph-based retrieval traverses relationships between documents to add related context.",
     "topics": ["retrieval", "graph"], "links": ["d4", "d6"]},
    {"id": "d6", "text": "Evaluation of RAG systems uses an LLM judge to score answers against grading notes.",
     "topics": ["evaluation", "RAG"], "links": ["d5"]},
]


def run_starter_demo(llm: ChatOpenAI) -> None:
    def kw(q, t):
        return len(set(re.findall(r"[a-z0-9]+", q.lower())) & set(re.findall(r"[a-z0-9]+", t.lower())))

    def base(q, k=3):
        s = sorted(((d["id"], kw(q, d["text"])) for d in SAMPLE_DOCS), key=lambda x: -x[1])
        return [(i, v) for i, v in s[:k] if v > 0]

    G = nx.DiGraph()
    for d in SAMPLE_DOCS:
        for o in d["links"]:
            G.add_edge(f"doc:{d['id']}", f"doc:{o}", edge_type="links_to")
        for t in d["topics"]:
            G.add_edge(f"doc:{d['id']}", f"topic:{t}", edge_type="relates_to")
    query = "How does breaking a question into parts and following document links help retrieval?"
    seeds = [i for i, _ in base(query)]
    union = {s: "seed" for s in seeds}
    for s in seeds:
        for _, tgt, ed in G.out_edges(f"doc:{s}", data=True):
            if ed["edge_type"] == "links_to":
                union.setdefault(tgt.removeprefix("doc:"), "linked")
            else:
                for sib, _ in G.in_edges(tgt):
                    union.setdefault(sib.removeprefix("doc:"), "topic")
    subs = decompose_query(llm, query) or [query]
    total: dict[str, float] = defaultdict(float)
    for sq in subs:
        for i, v in base(sq, 9):
            total[i] += v
    print(f"Query: {query}\nBASELINE: {seeds}\nMULTI-STEP {subs}: {sorted(total, key=lambda i: -total[i])[:3]}"
          f"\nGRAPH: {union}")


# %% [markdown]
# ## Run

# %%
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Checkpoint 4.1: baseline vs graph-based multi-hop retrieval.")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--questions", default="", help="comma-separated ids, e.g. Q1,Q2 (default: all)")
    ap.add_argument("--systems", default="", help="comma-separated system names (default: all)")
    ap.add_argument("--ask", default="", help="show the advanced retrieval for one question and answer it")
    ap.add_argument("--demo", action="store_true", help="run the starter's sample-corpus demonstration")
    args = ap.parse_args(argv)

    llm = make_llm()
    if args.demo:
        run_starter_demo(llm)
        return

    print(f"Checkpoint 4.1 — scenario: {SCENARIO}")
    chunks = load_chunks(PAPERS_DIR)
    db = build_or_load_db(chunks, CHROMA_DIR)
    graph = load_graph(chunks, llm)

    print("\nYour advanced-retrieval plan:\n" + json.dumps(my_advanced_plan(), indent=2, ensure_ascii=False))

    if args.ask:
        g = GraphMultiHopRetriever(graph, db, llm)
        res = g.retrieve(args.ask)
        print(f"\nQuestion: {args.ask}\nscope: {res['scope']}\nsub-queries: {res['sub_queries']}"
              f"\nhop-1 papers: {res['papers']}"
              f"\nmatched relations: {res['relations']}")
        for i, cid in enumerate(res["primary"], 1):
            print(f"  PRIMARY {i}: {graph.by_id[cid]['title'][:70]} [{cid}]")
        for cid, why in res["context"]:
            print(f"  CONTEXT ({why}): {graph.by_id[cid]['title'][:60]} [{cid}]")
        answer = llm.invoke([SystemMessage(content=GRAPH_SYSTEM), HumanMessage(
            content=f"Documents:\n{g.build_context(res)}\n\nQuestion: {args.ask}")]).content
        print(f"\nAnswer:\n{answer}")
        log_response("ASK advanced", args.ask, answer)
        return

    systems = make_systems(chunks, db, graph, llm)
    if args.systems:
        systems = {k: v for k, v in systems.items() if k in args.systems.split(",")}
    qids = [q for q in args.questions.split(",") if q]
    print(f"\nEvaluating {list(systems)} x {qids or 'all questions'} x {args.runs} run(s)...")
    records = run_evaluation(systems, graph, llm, args.runs, qids)

    names = list(systems)
    ids_31 = [i for i in ("Q1", "Q2", "Q3", "Q4") if not qids or i in qids]
    ids_all = sorted({r["id"] for r in records})
    report = (f"\n3.1 evaluation set (Q1-Q4), mean over {args.runs} run(s):\n{summarize(records, ids_31, names)}"
              f"\n\nExtended set ({', '.join(ids_all)}):\n{summarize(records, ids_all, names)}"
              f"\n\nPer question (passes/runs, evidence found):\n{per_question(records, names)}")
    print(report)
    log_response("SUMMARY", f"{names} x {ids_all} x {args.runs} runs", report)

    validation = validate_framework(llm)
    print(f"\nFramework validation (3.1 judge):\n{validation}")
    log_response("FRAMEWORK VALIDATION", "3.1 known-answer cases on Q3 grading notes", validation)

    RESULTS_PATH.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nPer-run records written to {RESULTS_PATH.name}; run-1 detail appended to {LOG_PATH.name}.")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Step 6 — Written report
#
# The deliverable is the Capstone Checkpoint 4.1 worksheet (in this folder), written from the log and results:
# system overview, failure diagnosis, data redesign, advanced retrieval with log evidence, before/after
# comparison at a matched document count, and analysis & reflection.
