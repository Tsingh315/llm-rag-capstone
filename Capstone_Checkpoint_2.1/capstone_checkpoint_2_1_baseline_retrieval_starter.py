r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

Run:   python capstone_checkpoint_2_1_baseline_retrieval_starter.py            # runs your 3 queries
       python capstone_checkpoint_2_1_baseline_retrieval_starter.py --chat     # interactive demo

Prerequisite: run `python ingest_papers.py` once. It parses the PDFs into papers_txt/ (see README_INGESTION.md).
This script only READS papers_txt/; it never parses PDFs itself.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your
# retrieval system implementation and testing. This script implements a **hybrid baseline**
# (BM25 + vector search, fused) over the research-paper corpus.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core langchain-chroma langchain-text-splitters rank-bm25 python-dotenv requests pypdf`
# 3. Use the OpenRouter API key provided for this program (`OPENROUTER_API_KEY=sk-or-v1-...` in a `.env`
#    file next to this script or in your home folder). The LLM is `openai/gpt-5.4-mini` and embeddings are
#    `openai/text-embedding-3-small`, both covered by the course credits.
# 4. Run `python ingest_papers.py` once to parse the PDFs into `papers_txt/` (needs Docker for GROBID).
#
# The first run of this script embeds every chunk (a few minutes) and saves the vector DB in
# `chroma_db/`; later runs reuse it.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# %%
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
EMBEDDING_MODEL = "openai/text-embedding-3-small"

TOP_K = 5                  # chunks sent to the LLM as context
CANDIDATE_POOL = 20        # candidates pulled from EACH retriever before fusion
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

# Chunking: papers are far longer than an embedding model's input limit, so each is split into pieces.
CHUNK_SIZE = 1500          # characters (~250 words)
CHUNK_OVERLAP = 200
# Files above this many characters are whole proceedings volumes (books), not papers: the largest real
# paper is ~180k chars and the smallest volume ~800k. They would dominate retrieval, so they are skipped.
MAX_DOC_CHARS = 400_000

PAPERS_DIR = HERE / "papers_txt"      # written by ingest_papers.py
CHROMA_DIR = HERE / "chroma_db"
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set. Use the OpenRouter API key provided for this "
            "course, put it in a .env file next to this script (or in your home folder), and rerun.\n"
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
        timeout=60,
        max_retries=3,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
        chunk_size=100,          # texts per embedding request
        timeout=60,
        max_retries=5,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


def chat_loop(response):
    print("Chat over the research papers. Type your question; 'exit'/'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        try:
            result = response(user_input)
        except Exception as e:
            print(f"Error: {e}")
            continue
        print(f"\nAssistant: {result}\n")


# %% [markdown]
# ## Step 2 — The baseline retriever: chunked papers, BM25 + vector, fused
#
# `ingest_papers.py` (run once, separately) turns the PDFs into `papers_txt/*.txt` plus a
# `manifest.json` holding each paper's title. This script:
#
# 1. **Loads** the parsed papers, skips failed files and whole-volume "books", and **splits** each
#    paper into overlapping chunks (each chunk starts with the paper title so it keeps its context).
# 2. Indexes the **same chunks** twice: BM25 (keyword) and Chroma (vector embeddings).
# 3. For a query, pulls a candidate pool from each, min-max normalises both score lists (vector
#    *distance* is inverted, since lower = better), and combines them with a weighted sum.
#    Chunks are fused by `chunk_id`.
# 4. Sends the top chunks to the LLM, which must answer only from them.
#
# Retrieval (this class) knows nothing about the LLM; the answer step is separate.

# %%
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale a list of scores to [0, 1]. If invert is True, flip the scores so a LOW raw
    value (e.g., a small vector distance = very similar) becomes a HIGH normalized score."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [0.5] * len(scores)
    normalized = [(s - lo) / (hi - lo) for s in scores]
    return [1 - n for n in normalized] if invert else normalized


def load_chunks(papers_dir: Path = PAPERS_DIR) -> list[dict]:
    """Read the ingestion output and split each paper into chunks. Never parses PDFs."""
    manifest_path = papers_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"\n[setup] {manifest_path} not found. Run ingestion first:  python ingest_papers.py\n")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks: list[dict] = []
    skipped_volumes, papers = [], 0
    for pdf_name, rec in sorted(manifest.items()):
        txt_path = papers_dir / f"{Path(pdf_name).stem}.txt"
        if rec.get("status") != "ok" or not txt_path.exists():
            continue
        if rec.get("chars", 0) > MAX_DOC_CHARS:
            skipped_volumes.append(pdf_name)
            continue
        title = rec.get("title") or Path(pdf_name).stem
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        papers += 1
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append({
                "chunk_id": f"{Path(pdf_name).stem}::{i}",
                "source": pdf_name,
                "title": title,
                "text": f"{title}\n\n{piece}",       # title prefix keeps each chunk's context
            })
    print(f"Loaded {len(chunks)} chunks from {papers} papers "
          f"(skipped {len(skipped_volumes)} whole-volume files over {MAX_DOC_CHARS:,} chars).")
    return chunks


def _fingerprint(chunks: list[dict]) -> str:
    """Identifies exactly what the vector DB was built from, so a stale DB is rebuilt automatically."""
    h = hashlib.sha256()
    h.update(f"{EMBEDDING_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(hashlib.sha256(c["text"].encode()).digest())
    return h.hexdigest()


def build_or_load_db(chunks: list[dict], chroma_dir: Path = CHROMA_DIR) -> Chroma:
    """Reuse the saved vector DB only if it was fully built from these exact chunks; otherwise rebuild."""
    marker = chroma_dir / ".build_complete"
    fingerprint = _fingerprint(chunks)
    if marker.exists() and marker.read_text().strip() == fingerprint:
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=str(chroma_dir), embedding_function=get_embeddings())

    if chroma_dir.exists():
        if not (chroma_dir / "chroma.sqlite3").exists():
            raise SystemExit(f"[setup] {chroma_dir} exists but is not a Chroma DB; refusing to delete it.")
        print("Vector DB is missing, incomplete or out of date; rebuilding.")
        shutil.rmtree(chroma_dir)

    print(f"Building vector DB (first run: embedding {len(chunks)} chunks)...")
    db = Chroma(persist_directory=str(chroma_dir), embedding_function=get_embeddings())
    batch = 200
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        db.add_documents(
            [Document(page_content=c["text"],
                      metadata={"chunk_id": c["chunk_id"], "source": c["source"], "title": c["title"]})
             for c in part],
            ids=[c["chunk_id"] for c in part],
        )
        print(f"  embedded {min(start + batch, len(chunks))}/{len(chunks)}")
    marker.write_text(fingerprint)          # written last: a crash mid-build leaves no marker
    return db


class BaseRetriever(ABC):
    @abstractmethod
    def retrievedContext(self, query: str, k: int = TOP_K) -> tuple[dict[str, float], str]: ...


class HybridRetriever(BaseRetriever):
    def __init__(self, chunks: list[dict], db: Chroma):
        self._chunks = chunks
        self._by_id = {c["chunk_id"]: c for c in chunks}
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self._db = db
        print(f"Hybrid retriever ready over {len(chunks)} chunks.")

    def describe(self, chunk_id: str) -> dict:
        c = self._by_id[chunk_id]
        return {"title": c["title"], "source": c["source"]}

    # Single-retriever helpers. BM25: higher score = better. Vector: returns a DISTANCE, lower = better.
    def _bm25_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._chunks[i]["chunk_id"], self._chunks[i]["text"], float(scores[i])) for i in top]

    def _vector_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [(d.metadata["chunk_id"], d.page_content, float(s)) for d, s in results]

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Fuse the two retrievers into one ranking of (chunk_id, text, fused_score)."""
        # BM25 chunks with score 0 share no term with the query; they are noise, not candidates.
        bm_candidates = [c for c in self._bm25_topk(query, CANDIDATE_POOL) if c[2] > 0]
        vector_candidates = self._vector_topk(query, CANDIDATE_POOL)

        bm_norm = _normalize([score for _, _, score in bm_candidates])
        vector_norm = _normalize([score for _, _, score in vector_candidates], invert=True)

        fused: dict[str, dict] = {}
        for (cid, content, _), score in zip(bm_candidates, bm_norm):
            fused[cid] = {"content": content, "bm": score, "vec": 0.0}
        for (cid, content, _), score in zip(vector_candidates, vector_norm):
            entry = fused.setdefault(cid, {"content": content, "bm": 0.0, "vec": 0.0})
            entry["vec"] = score

        results = sorted(
            ((cid, d["content"], WEIGHT_BM25 * d["bm"] + WEIGHT_VECTOR * d["vec"]) for cid, d in fused.items()),
            key=lambda r: r[2], reverse=True)
        return results[:k]

    def retrievedContext(self, query: str, k: int = TOP_K) -> tuple[dict[str, float], str]:
        """Returns ({chunk_id: fused_score}, context_text). The dict is ordered best-first."""
        results = self.getTopK(query, k)
        scores = {cid: round(score, 4) for cid, _, score in results}
        context = "\n\n---\n\n".join(
            f"[{self._by_id[cid]['title']} | {self._by_id[cid]['source']}]\n{content}"
            for cid, content, _ in results)
        return scores, context


# %% [markdown]
# ## Step 3 — Your representative queries
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    return [
        "How long did the fairness verifier take to verify the deep recurrent neural network with an error probability of 10⁻¹⁰, and how many samples did it need?",
        "Which papers compare their approach against an SMT-based tool and argue that SMT solving limits scalability? Name the baseline in each.",
        "Is there a technique that re-runs a whole computation from the beginning, guided by a log of earlier results, to get back to a paused point? How did its speed compare with a logic-programming language on the queens puzzle?"
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the hybrid retriever and the LLM, printing the
# retrieved chunks with their fused scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
class RagAssistant:
    """The answer step: retrieval (any BaseRetriever) plus the LLM. Kept separate from retrieval."""

    def __init__(self, retriever: BaseRetriever, llm: ChatOpenAI | None = None):
        self._retriever = retriever
        self._llm = llm or make_llm()
        self._history = [SystemMessage(content=SYSTEM_PROMPT)]

    @staticmethod
    def _user_message(question: str, context: str) -> str:
        return f"Documents:\n{context}\n\nQuestion: {question}"

    def query(self, question: str) -> tuple[dict[str, float], str]:
        """Stateless: returns (retrieved chunk scores, answer)."""
        scores, context = self._retriever.retrievedContext(question)
        response = self._llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                                     HumanMessage(content=self._user_message(question, context))])
        return scores, response.content

    def queryWHistory(self, question: str) -> str:
        _, context = self._retriever.retrievedContext(question)
        self._history.append(HumanMessage(content=self._user_message(question, context)))
        try:
            answer = self._llm.invoke(self._history).content
        except Exception:
            self._history.pop()
            raise
        self._history.append(AIMessage(content=answer))
        return answer

    def chat(self) -> None:
        chat_loop(self.queryWHistory)


def run_queries(assistant: RagAssistant, retriever: HybridRetriever) -> None:
    print(f"Checkpoint 2.1 — hybrid retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(my_representative_queries(), 1):
        scores, ans = assistant.query(query)
        hits = "\n".join(
            f"    {score:.3f}  {retriever.describe(cid)['title'][:80]}  [{retriever.describe(cid)['source']}]"
            for cid, score in scores.items())
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print(f"  retrieved (fused score, title, file):\n{hits}")
        print(f"  answer: {ans}\n")
        log(f"QUERY {i}: {query}", f"retrieved=\n{hits}\nanswer={ans}")
    print("=" * 72)
    print("Done. Use the retrieved document results above as evidence in your writeup, and "
          "describe your REAL baseline (over your full corpus) in the submission.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Hybrid (BM25 + vector) retrieval over the parsed research papers.")
    ap.add_argument("--papers-dir", type=Path, default=PAPERS_DIR, help="output folder of ingest_papers.py")
    ap.add_argument("--chat", action="store_true", help="interactive chat instead of the 3 representative queries")
    args = ap.parse_args(argv)

    check_api_key()                                   # fail fast, before any work
    chunks = load_chunks(args.papers_dir)
    if not chunks:
        raise SystemExit("[setup] no chunks loaded; check that ingestion produced papers_txt/.")
    db = build_or_load_db(chunks)
    retriever = HybridRetriever(chunks, db)
    assistant = RagAssistant(retriever)
    if args.chat:
        assistant.chat()
    else:
        run_queries(assistant, retriever)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.

# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository.
# Upload the completed worksheet only to the learning platform as your graded submission.
