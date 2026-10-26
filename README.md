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
- **4.1, advanced retrieval:** a paper graph (paper / chunk / tool nodes: reading order, duplicate copies,
  paper → baseline-tool relations) walked in two hops, seeded by LLM query decomposition. Same 3.1 evaluation at
  a matched 8-chunk budget: 3.00/4 pass (mean of 3 runs) vs 1.00/4 for the baseline.
- **5.1, agent:** a LangGraph agent decides whether a question needs the papers, may ask one clarifying question,
  and otherwise runs the 4.1 graph retrieval once and answers from it. On 7 questions × 3 runs it ties the fixed
  4.1 pipeline (3.67/7 passes) with the most evidence reaching the LLM (69%), at one extra LLM call.
- **6.1, security and cost audit:** the 5.1 agent, unchanged, under 10 probes and the test tasks, twice, with real
  token counts. The research path resisted command injection and a poisoned passage; the no-retrieval path did not
  (role-play accepted, refusals that add offers, a promised search never run). The answer call is 65% of input tokens.
- **7.1, production readiness:** safeguards and cost optimizations aimed at the 6.1 findings, demonstrated on a
  three-passage paper corpus (not yet ported into the 5.1 agent). Refusal format 0/4 → 4/4 on the failed 6.1 probes;
  a repeated question costs 0 tokens instead of 684; the direct path uses 229–402 tokens.
- **8.1, design review:** the 4.1 pipeline and the 5.1 agent, each wrapped in the 7.1 hardening, on the 35-question test
  set (3 runs). Close overall (22.0/35 vs 24.0/35); the agent is better on cross-paper questions and quotes, the pipeline
  on multi-turn and cost. Both pass all red-team questions.

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
├── Capstone_Checkpoint_4.1/   # advanced retrieval: paper graph + multi-hop + query decomposition
│   ├── README.md
│   ├── capstone_checkpoint_4_1_advanced_retrieval_starter.py   # solution file (imports the 2.1 system)
│   ├── checkpoint_4_1_responses.log, checkpoint_4_1_results.json, checkpoint_4_1_run_output.txt
│   ├── screenshots/
│   └── Tarundeep_Required_Capstone_Checkpoint_4_1_Worksheet.docx
├── Capstone_Checkpoint_5.1/   # agent-based RAG: plan -> graph_retrieve (4.1, one shot) -> answer
│   ├── README.md
│   ├── capstone_checkpoint_5_1_agent_rag_starter.py   # solution file (imports the 4.1 system)
│   ├── checkpoint_5_1_summary.txt, checkpoint_5_1_results.json, checkpoint_5_1_agent.log
│   ├── screenshots/, experiments/, results_history/, exploration_v1_v4/
│   └── Tarundeep_Required_Capstone_Checkpoint_5_1_Worksheet.docx
├── Capstone_Checkpoint_6.1/   # security and cost audit of the 5.1 agent (probes, real token split)
│   ├── README.md
│   ├── capstone_checkpoint_6_1_security_cost_starter.py   # solution file (imports the 5.1 agent)
│   ├── checkpoint_6_1_summary.txt, checkpoint_6_1_results_run*.json, checkpoint_6_1_agent_run*.log
│   ├── screenshots/
│   └── Tarundeep_Required_Capstone_Checkpoint_6_1_Worksheet.docx
├── Capstone_Checkpoint_7.1/   # production readiness: safeguards + cost optimizations (small-corpus demo)
│   ├── README.md
│   ├── capstone_checkpoint_7_1_production_ready_starter.py   # solution file
│   ├── checkpoint_7_1_output.txt, checkpoint_7_1_agent.log
│   └── Tarundeep_Required_Capstone_Checkpoint_7_1_Worksheet.docx
├── Capstone_Checkpoint_8.1/   # design review: 4.1 + 7.1 vs 5.1 + 7.1 on the 35-question test set
│   ├── README.md
│   ├── capstone_checkpoint_8_1_candidates.py, run_candidates_8_1.py
│   └── checkpoint_8_1_results.*, checkpoint_8_1_run_output_part*.txt
├── data/                      # git-ignored, shared by all checkpoints
│   ├── ResearchPapers/        # the course PDFs
│   ├── papers_txt/            # parsed text (written by ingest_papers.py)
│   └── chroma_db/             # vector index (built on first run, then reused)
└── ...                        # one folder per module checkpoint
```

Each checkpoint folder has its own README: [Checkpoint 1.1](Capstone_Checkpoint_1.1/README.md) ·
[Checkpoint 2.1](Capstone_Checkpoint_2.1/README.md) ·
[Checkpoint 3.1](Capstone_Checkpoint_3.1/README.md) ·
[Checkpoint 4.1](Capstone_Checkpoint_4.1/README.md) ·
[Checkpoint 5.1](Capstone_Checkpoint_5.1/README.md) ·
[Checkpoint 6.1](Capstone_Checkpoint_6.1/README.md) ·
[Checkpoint 7.1](Capstone_Checkpoint_7.1/README.md) ·
[Checkpoint 8.1](Capstone_Checkpoint_8.1/README.md).

## Setup

```bash
git clone <this-repo-url>
cd llm-rag-capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Developed and run with Python 3.14; other recent Python 3 versions should work but are untested.

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

# 4.1: advanced retrieval vs baseline (reuses data/; first run extracts tool relations once, ~119 LLM calls)
cd ../Capstone_Checkpoint_4.1
python capstone_checkpoint_4_1_advanced_retrieval_starter.py      # or --ask "question" for one traced retrieval

# 5.1: the agent (needs langgraph); --chat to talk to it, no flags to run the evaluation
cd ../Capstone_Checkpoint_5.1
python capstone_checkpoint_5_1_agent_rag_starter.py --chat

# 6.1: security probes + token split on the 5.1 agent (~40 LLM calls)
cd ../Capstone_Checkpoint_6.1
python capstone_checkpoint_6_1_security_cost_starter.py

# 8.1: compare the two production candidates on the 35-question test set
#      (~1,000 LLM calls, about 25 minutes; needs data/papers_txt and data/chroma_db from the steps above)
cd ../Capstone_Checkpoint_8.1
python run_candidates_8_1.py --runs 3

# 7.1: hardening and cost demo on a small paper corpus (a handful of LLM calls)
cd ../Capstone_Checkpoint_7.1
python capstone_checkpoint_7_1_production_ready_starter.py
```

Building a vector index embeds all 8,087 chunks once (an embedding API cost); later runs reuse it unless the
chunks change. 2.1 keeps its own index in `Capstone_Checkpoint_2.1/chroma_db/` (git-ignored); 3.1 uses
`data/chroma_db/`. To avoid embedding twice, copy the 2.1 index to `data/chroma_db/` before running 3.1.

## Known limitations and notes for reviewers

- **Setup is not self-contained.** The 153-paper corpus (~500 MB) is not in the repo, parsing it needs Docker (GROBID),
  and every LLM step needs an OpenRouter API key. The only parts that run without the corpus are 1.1 and the 7.1 demo.
- **Cost and time.** Building the index embeds 8,087 chunks once. The 8.1 comparison makes about 1,000 LLM calls and takes
  about 25 minutes; the 7.1 test-set run is about 70 questions.
- **Parsing.** 152 of 153 PDFs parsed and 148 papers were indexed (five proceedings volumes lost). Tables and equations are
  flattened, sections are not saved, and the quality checks only sample the start of each file. At least six filenames do
  not match the paper inside (titles come from the text).
- **Evaluation.** Questions were written by me and graded by an LLM judge (4 questions in 3.1/4.1, 7 in 5.1, 35 in 7.1/8.1).
  Results move by a question or two between runs; in 8.1 the gap between the candidates (22.0 vs 24.0 of 35) is small.
- **What 8.1 does and does not show.** Red-team pass is measured on 5 questions; the 10 Lab 6.1 probes were not re-run on
  the 8.1 candidates. The 7.1 demo and its 24/35 test-set result used a 3-passage sample and the 2.1 retriever
  respectively, not the 4.1 graph.
- **Known weak spots.** Trend questions reach only 17% of the needed evidence; cross-paper questions reach 23% (A) and 63% (B).
- **Not built.** The cache is exact-match (not semantic) and the 7.1 router is not part of the 8.1 candidates.
- **Presentation.** The slides and prep notes are in [`Capstone_Checkpoint_8.1/presentation/`](Capstone_Checkpoint_8.1/presentation);
  the test set and results they cite are in [8.1](Capstone_Checkpoint_8.1/testset_8_1.md) and
  [results](Capstone_Checkpoint_8.1/checkpoint_8_1_results.md).

## Progress

| Checkpoint | Topic | Status |
|---|---|---|
| 1.1 | Testing the LLM without retrieval | ✅ Complete |
| 2.1 | Retrieval strategy design and baseline (hybrid BM25 + vector) | ✅ Complete |
| 3.1 | Evaluation infrastructure and baseline diagnosis (LLM judge) | ✅ Complete |
| 4.1 | Advanced retrieval: graph-based multi-hop + query decomposition | ✅ Complete |
| 5.1 | Agent-based RAG: plan, clarify, one-shot graph retrieval; evaluated vs 2.1 and 4.1 | ✅ Complete |
| 6.1 | Security and performance audit of the 5.1 agent (probes, token cost) | ✅ Complete |
| 7.1 | Production readiness: safeguards and cost/latency optimizations, measured before vs after | ✅ Complete |
| 8.1 | Design review: the 4.1 pipeline and the 5.1 agent, each with the 7.1 hardening, compared on a 35-question test set | 🟡 In progress |
