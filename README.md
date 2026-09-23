# Retrieval-Augmented LLM System — MIT xPRO Capstone

Capstone project for the MIT xPRO LLM / RAG program. This single repository tracks the
system as it evolves from a baseline (no-retrieval) LLM into a full retrieval-augmented
generation (RAG) pipeline, one capstone checkpoint per module.

## Selected scenario

> **TODO:** Name the scenario chosen from the Capstone Project Overview page and describe it
> in 2–3 sentences (the domain, the users, and the kinds of questions the system must answer).

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
├── Capstone_Checkpoint_1.1/   # LLM without retrieval — baseline evaluation
│   ├── solution file (notebook / script)
│   └── worksheet (completed)
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

Set any API keys as environment variables (e.g. `export OPENAI_API_KEY=...`) — never commit them.
Then open the notebook in the checkpoint folder:

```bash
jupyter notebook Capstone_Checkpoint_1.1/
```

## Progress

| Checkpoint | Topic | Status |
|---|---|---|
| 1.1 | Testing the LLM without retrieval | In progress |
