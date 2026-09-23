# Retrieval-Augmented LLM System — MIT xPRO Capstone

Capstone project for the MIT xPRO LLM / RAG program. This single repository tracks the
system as it evolves from a baseline (no-retrieval) LLM into a full retrieval-augmented
generation (RAG) pipeline, one capstone checkpoint per module.

## Selected scenario

**Research Paper Navigator** — a conversational interface over a collection of ~153 research
papers (roughly 2009–2024). Researchers and engineers use it to find and check specific claims
without reading every paper. It must support:

- **Single-paper questions**: a finding, method, or result from one named paper, backed by a direct quote.
- **Cross-paper comparison**: how papers define a problem, differ in approach, or conflict in results.
- **Trend questions**: how an idea evolved across the collection over time.
- **Grounded answers**: every substantive claim traced to a named source and quoted verbatim.
- **Multi-turn conversation**: follow-ups that refer back to papers already discussed.
- **Corpus-boundary honesty**: saying plainly when the answer isn't in the collection.

## Purpose of the system

Build an LLM-powered assistant for the scenario above that gives **grounded, trustworthy**
answers. Checkpoint 1.1 measures how the base model behaves *without* retrieval: where it is
reliable, where it hallucinates or goes stale, and where external retrieval is needed.
Later checkpoints add retrieval, grounding, and evaluation.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── Capstone_Checkpoint_1.1/   # LLM without retrieval — baseline evaluation
│   ├── capstone_checkpoint_1_1_baseline_starter.py   # solution file
│   ├── checkpoint_1_1_responses.log                  # prompt/response evidence
│   └── Tarundeep_Required_Capstone_Checkpoint_1_1_Worksheet.pdf
├── Capstone_Checkpoint_2.1/   # added in Module 2
└── ...                        # one folder per module checkpoint
```

## Setup

```bash
git clone <this-repo-url>
cd llm-rag-capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenRouter key (`.env` is git-ignored — never commit it).
Then run the checkpoint script (it uses Jupytext `# %%` cells, so it also opens as a notebook in VS Code):

```bash
python Capstone_Checkpoint_1.1/capstone_checkpoint_1_1_baseline_starter.py
```

**Dataset:** the ResearchPapers corpus (~450 MB) is provided by the course and is not committed.
Place it at `data/ResearchPapers/` locally (git-ignored). Checkpoint 1.1 does not need it;
retrieval over it starts in Checkpoint 2.1.

## Progress

| Checkpoint | Topic | Status |
|---|---|---|
| 1.1 | Testing the LLM without retrieval | ✅ Complete |
