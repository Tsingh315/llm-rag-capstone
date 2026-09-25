[← Repository README](../README.md) · [← Checkpoint 1.1](../Capstone_Checkpoint_1.1/README.md)

# Capstone Checkpoint 2.1 — Hybrid Retrieval over the Research Papers

Baseline retrieval for the Research Paper Navigator scenario: **BM25 + vector search, fused**, over
chunked papers, with an LLM that answers only from the retrieved chunks.

| File | What it is |
|---|---|
| `capstone_checkpoint_2_1_baseline_retrieval_starter.py` | Solution file: loads the parsed papers, chunks them, indexes the same chunks in BM25 and Chroma, fuses by `chunk_id`, answers the 3 representative queries |
| `checkpoint_2_1_retrieval.log` | Retrieved chunks (with fused scores) and answers from the run |
| `ingest_papers.py` | One-time PDF -> text step using GROBID (separate script by design), with a pre-flight size report and sanity tests |
| `Dockerfile.grobid` | Builds the GROBID image that `ingest_papers.py` starts and stops itself |
| [`README_RETRIEVAL.md`](README_RETRIEVAL.md) | Notes on the retrieval code, design decisions, bugs fixed, first-run results (section 9) |
| [`README_INGESTION.md`](README_INGESTION.md) | Notes on the ingestion script, its sanity tests and their limits, corpus data-quality findings |

**Run** (from this folder): `python ingest_papers.py --pdf-dir ../data/ResearchPapers` once (needs Docker), then
`python capstone_checkpoint_2_1_baseline_retrieval_starter.py`. Without `--pdf-dir`, the ingestion script looks for
`CapstoneDatasets/ResearchPapers/` next to itself.
`papers_txt/`, `chroma_db/` and the PDFs are not committed; they are regenerated from the course dataset.

**Result:** 152 of 153 PDFs parsed (one 939-page proceedings volume exceeds GROBID's limit); 8,087 chunks
from 148 papers indexed. Queries 1 and 3 retrieved the right paper; query 1's exact numbers were not found;
query 2 (multi-paper) was the weakest. Repeated copies of the same paper crowd the top results.
Details in `README_RETRIEVAL.md`, section 9.
