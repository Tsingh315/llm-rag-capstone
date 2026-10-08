# Retrieval-Augmented LLM System — MIT xPRO Capstone

Capstone project for the MIT xPRO LLM / RAG program. This single repository tracks the
system as it evolves from a baseline (no-retrieval) LLM into a full retrieval-augmented
generation (RAG) pipeline, one capstone checkpoint per module.

## Selected scenario

**Research Paper Navigator** — a conversational interface over a collection of ~153 research
papers (roughly 2009–2025, plus at least one 2026 preprint). Researchers and engineers use it to find and check specific claims
without reading every paper. It must support:

- **Single-paper questions**: a finding, method, or result from one named paper, backed by a direct quote.
- **Cross-paper comparison**: how papers define a problem, differ in approach, or conflict in results.
- **Trend questions**: how an idea evolved across the collection over time.
- **Grounded answers**: every substantive claim traced to a named source and quoted verbatim.
- **Multi-turn conversation**: follow-ups that refer back to papers already discussed.
- **Corpus-boundary honesty**: saying plainly when the answer isn't in the collection.

## Purpose of the system

Build an LLM-powered assistant for the scenario above that gives **grounded, trustworthy**
answers. So far:

- **1.1, no retrieval:** the base model either refuses or, in one case, fabricates a convincing
  quote. Retrieval is required.
- **2.1, retrieval:** the PDFs are parsed once (GROBID) into 8,087 chunks from 148 papers; a hybrid
  retriever (BM25 + vector search, blended 50/50) returns the top 5 chunks, and the LLM answers
  only from them.
- **3.1, evaluation:** an LLM judge scores answers pass/fail against grading notes, with a
  validation step that checks the judge itself. Baseline: 1/4 pass. The system answers facts
  stated in abstracts but misses facts buried in results sections or spread across papers.

Later checkpoints improve retrieval and keep re-running the same evaluation to measure progress.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── Capstone_Checkpoint_1.1/   # LLM without retrieval — baseline evaluation
│   ├── README.md
│   ├── capstone_checkpoint_1_1_baseline_starter.py   # solution file
│   ├── checkpoint_1_1_responses.log                  # prompt/response evidence
│   └── Tarundeep_Required_Capstone_Checkpoint_1_1_Worksheet.pdf
├── Capstone_Checkpoint_2.1/   # hybrid retrieval (BM25 + vector) over chunked papers
│   ├── README.md              # folder index and results
│   ├── README_RETRIEVAL.md    # retrieval code: design, bugs fixed, first-run results
│   ├── README_INGESTION.md    # PDF -> text (GROBID): design, sanity tests, corpus findings
│   ├── capstone_checkpoint_2_1_baseline_retrieval_starter.py   # solution file
│   ├── ingest_papers.py, Dockerfile.grobid
│   └── checkpoint_2_1_retrieval.log
├── Capstone_Checkpoint_3.1/   # evaluation: LLM judge + framework validation
│   ├── README.md
│   ├── capstone_checkpoint_3_1_evaluation_starter.py   # solution file (imports the 2.1 retriever)
│   ├── checkpoint_3_1_evaluation.log
│   └── Tarundeep_Required_Capstone_Checkpoint_3_1_Worksheet.docx
├── data/                      # git-ignored, shared by all checkpoints
│   ├── ResearchPapers/        # the course PDFs
│   ├── papers_txt/            # parsed text (written by ingest_papers.py)
│   └── chroma_db/             # vector index (built on first run, then reused)
└── ...                        # one folder per module checkpoint
```

Each checkpoint folder has its own README: [Checkpoint 1.1](Capstone_Checkpoint_1.1/README.md) ·
[Checkpoint 2.1](Capstone_Checkpoint_2.1/README.md) ·
[Checkpoint 3.1](Capstone_Checkpoint_3.1/README.md).

## Setup

```bash
git clone <this-repo-url>
cd llm-rag-capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenRouter key (`.env` is git-ignored — never commit it).
The scripts use Jupytext `# %%` cells, so they also open as notebooks in VS Code.

**Dataset:** the ResearchPapers corpus (~500 MB) is provided by the course and is not committed.
Place the PDFs at `data/ResearchPapers/` (git-ignored). Everything generated from it also lives in
`data/`, so every checkpoint shares one copy.

**Run order:**

```bash
# 1.1: no retrieval (does not need the corpus)
python Capstone_Checkpoint_1.1/capstone_checkpoint_1_1_baseline_starter.py

# 2.1: parse the PDFs once (needs Docker for GROBID), then run the retriever
cd Capstone_Checkpoint_2.1
python ingest_papers.py --pdf-dir ../data/ResearchPapers --out-dir ../data/papers_txt
python capstone_checkpoint_2_1_baseline_retrieval_starter.py --papers-dir ../data/papers_txt

# 3.1: evaluation (imports the 2.1 retriever; builds data/chroma_db on its first run, then reuses it)
cd ../Capstone_Checkpoint_3.1
python capstone_checkpoint_3_1_evaluation_starter.py              # add SHOW_CHUNK_TEXT=1 to print chunk text
```

Building a vector index embeds all 8,087 chunks once (an embedding API cost); later runs reuse it unless the
chunks change. 2.1 keeps its own index in `Capstone_Checkpoint_2.1/chroma_db/` (git-ignored); 3.1 uses
`data/chroma_db/`. To avoid embedding twice, copy the 2.1 index to `data/chroma_db/` before running 3.1.

## Progress

| Checkpoint | Topic | Status |
|---|---|---|
| 1.1 | Testing the LLM without retrieval | ✅ Complete |
| 2.1 | Retrieval strategy design and baseline (hybrid BM25 + vector) | ✅ Complete |
| 3.1 | Evaluation infrastructure and baseline diagnosis (LLM judge) | ✅ Complete |
