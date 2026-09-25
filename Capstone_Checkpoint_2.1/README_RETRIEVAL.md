[← Checkpoint 2.1 README](README.md) · [Ingestion notes](README_INGESTION.md) · [Repository README](../README.md)

# Capstone Checkpoint 2.1 — Hybrid Retrieval Code: Notes & Design Decisions

Notes on `capstone_checkpoint_2_1_baseline_retrieval_starter.py`: how the code works, why it is written
the way it is, and what changes for production. The PDF -> text step is a separate script, documented in
[README_INGESTION.md](README_INGESTION.md). Line numbers refer to the file at the time of writing and may drift.

## Project context

- **Goal:** build a **Hybrid Retrieval** system (BM25 keyword search + vector/semantic search, fused into
  one ranking), adapted from the Module 2 Lab 2 code.
- **Chosen project / scenario:** **Research papers** (`SCENARIO = "research_papers"`). The corpus is 153
  academic PDFs in `CapstoneDatasets/ResearchPapers/` (152 parsed; see README_INGESTION.md), so the
  retriever must work on long-form papers and surface the real paper **title**, not just a file name.
- **Status:** **implemented and run end to end.** The file was restructured along the lab's lines (main
  guard, retriever classes, retrieval separate from the LLM step), reads the parsed papers from
  `papers_txt/`, chunks them, indexes the same chunks in BM25 and Chroma, and fuses by `chunk_id`. The
  three bugs in section 2 are fixed. Results of the first run are in section 8.

---

## 1. How the script runs

```bash
python capstone_checkpoint_2_1_baseline_retrieval_starter.py           # runs the 3 representative queries
python capstone_checkpoint_2_1_baseline_retrieval_starter.py --chat    # interactive demo
```

- The file has a `main()` guarded by `if __name__ == "__main__":`, so importing it has no side effects.
  It is still a notebook-style file (`# %%` markers are VS Code / Jupyter cell delimiters).
- `main()` order: check the API key (fail fast) -> `load_chunks()` (reads `papers_txt/`, skips volumes,
  splits into chunks) -> `build_or_load_db()` (embeds on the first run, reuses `chroma_db/` afterwards,
  rebuilds automatically if the chunks or chunk settings changed) -> `HybridRetriever` -> `RagAssistant`
  -> `run_queries()` (prints retrieved chunks with scores and the answer, appends to
  `checkpoint_2_1_retrieval.log`).
- Running it makes **real API calls** (OpenRouter key from a `.env` next to the script or in your home
  folder): about 8,000 embedding requests' worth of text on the first run, then one LLM call per query.
- It only **reads** `papers_txt/`; if that folder is missing it stops with a message to run
  `ingest_papers.py`.

---

## 2. Hybrid retriever: notes and known issues

**All three issues below (2.1-2.3) are fixed in the current file;** they are kept here as a record.

`HybridRetriever.getTopK` fuses BM25 and vector search: take the top candidates from each, min-max
normalize each score list (vector distances are **inverted**, since lower distance = better), merge by file
name, and combine with `WEIGHT_BM25 * bm + WEIGHT_VECTOR * vec`.

### 2.1 `retrievedContext` should return `(scores, context)`

The annotation says `tuple[Dict, str]` but the body returned only a string. To return a sorted map of
document name -> score plus the joined context string:

```python
def retrievedContext(self, query: str) -> tuple[Dict[str, float], str]:
    results = self.getTopK(query, TOP_K)          # already sorted, highest score first
    scores = {name: score for name, _, score in results}   # dicts keep insertion order
    context = "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)
    return scores, context
```

Callers unpack with `scores, context = retriever.retrievedContext(query)`. `BaseRetriever.query()` /
`queryWHistory()` currently treat the return value as a plain `str`, so they must be updated to unpack the
tuple (and the abstract method signature changed to match).

### 2.2 Bug: wrong list zipped on the vector pass

```python
for (fname, content, _), score in zip(bm_candidates, vector_norm):     # WRONG
for (fname, content, _), score in zip(vector_candidates, vector_norm): # CORRECT
```

Zipping `bm_candidates` with `vector_norm` attaches each vector score to a BM25 file name, so the fused
ranking is wrong.

### 2.3 Duplicate `getTopK`

A module-level `getTopK` (outside the class, with `self`) exists in addition to the method inside
`HybridRetriever`. The module-level copy is dead code and carries the same bug; keep only the method.

---

## 3. Reading the ingested papers (contract with the ingestion script)

The retrieval script only **reads** `papers_txt/`, produced by `ingest_papers.py`. If the folder is missing
it must fail with a clear message telling you to run ingestion; it must never trigger it.

- `papers_txt/<pdf name>.txt` is the text; `papers_txt/manifest.json` maps each PDF to its `title`, `status`
  and more. **Titles come from the manifest** (`manifest["<pdf name>.pdf"]["title"]`), not a separate titles
  file. Load only entries with `status == "ok"`.
- **Vector side:** in `build_or_load_db`, add the title to metadata:
  `metadata={"source": fname, "title": title}`. In `_vector_topk`, read `d.metadata.get("title")` as well as
  `source`.
- **BM25 side:** in `HybridRetriever.__init__`, load the titles from the manifest and look them up in
  `getTopK` / `retrievedContext` when building output. Index the title with the body:
  `tokenize(title + " " + doc)` (titles are dense with key terms). Prepending the title to `page_content`
  gives the vector side the same benefit, at the cost of embedding the title in every document.
- **Titles are not always reliable.** Some files fall back to the file name, and volumes/front matter have
  wrong titles (see README_INGESTION.md, section 7). Decide how to treat those (exclude, or show the file
  name).
- The existing loaders filter on `.endswith(".txt")`, so `manifest.json` in the same folder is ignored by them.

### 3.1 Gotchas

- **Delete `chroma_db/` after adding titles or changing chunking.** `build_or_load_db` reuses an existing
  non-empty directory, so an old DB would lack the new metadata.
- **Whole-paper embeddings.** The current code embeds each paper as one document. Papers are long and
  embedding models have token limits; chunking (section 4) is the likely improvement.
- **Repeated papers.** Many papers appear twice (conference and arXiv copies, by identical extracted title;
  not verified). Retrieval would return both and crowd the top-k; decide whether to deduplicate.
- **Large volumes.** Five files are 350-939-page proceedings volumes (four are in `papers_txt/`). Decide
  whether to exclude them from the index; they would dominate retrieval.

---

## 4. Chunking and vector search (implemented; section 4.1 lists the design points)

Chunking splits each paper into smaller, overlapping pieces; each piece becomes its own retrievable
"document" in BM25 and in the vector DB.

**Why:** embedding models have token limits and one vector for 15 pages is a blurry average; a question
about one experiment should match that paragraph, not the whole paper; and the LLM gets 3 relevant chunks
instead of 3 entire papers.

**Parameters:** size typically 200-500 words (~300-800 tokens); overlap 10-20% so a boundary-straddling
sentence appears whole in at least one chunk; split on paragraph, then sentence, then word boundaries. These
are common starting points, **not tuned values**: check against the three representative queries whether
the right chunk is in the top results.

**GROBID and `RecursiveCharacterTextSplitter` do not compete; they are different pipeline stages.** GROBID
parses PDFs (ingestion); the splitter only cuts a string that is already text (retrieval side).

```
PDF -> GROBID (ingestion) -> text per section -> RecursiveCharacterTextSplitter (chunk) -> BM25 + vectors
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter   # pip install langchain-text-splitters

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)   # characters, ~250 words
chunks = splitter.split_text(section_text)    # then attach title, section heading, paper_id as metadata
```

**As implemented:** each paper's flat `.txt` is split with `RecursiveCharacterTextSplitter` (1,500 chars,
200 overlap) and every chunk is prefixed with the paper title: 8,087 chunks from 148 papers. Chunks are
**not** aligned to sections (the ingestion script flattens sections; splitting within sections would need it
to save them as structured data; see README_INGESTION.md checklist). Whole-volume files over 400,000
characters are skipped (`MAX_DOC_CHARS`; the largest real paper is ~180k, the smallest volume ~800k).

**Vector search works fine with parsed text:** the embedding model only sees the plain-text strings you give
it. One `Document` per chunk, `page_content` = chunk text (title-prefixed), metadata
`{"chunk_id", "paper_id", "title", "section"}`, embedded with `get_embeddings()` into Chroma:

```python
docs = [Document(page_content=c["text"],
                 metadata={"chunk_id": c["chunk_id"], "paper_id": c["paper_id"],
                           "title": c["title"], "section": c["section"]})
        for c in chunks]
db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
```

`tokenize` (`[a-z0-9]+`) drops non-ASCII, so "10⁻¹⁰" in a query loses its superscripts and math symbols
vanish; equations and tables are often garbled and contribute little to either retriever.

### 4.1 Impact on the hybrid retriever code

Moving to chunks is a bigger refactor than swapping the parser, so decide up front whether the capstone
stays at whole-paper granularity (simpler, weaker) or moves to chunks (better retrieval). If chunking:

1. **Both retrievers index the same chunk list**: `BM25Okapi([tokenize(c["text"]) for c in chunks])` and the
   Chroma documents above. Same text, same IDs.
2. **Fuse by `chunk_id`, not file name.** Otherwise BM25 returns papers while vector search returns chunks
   and fusion breaks. In `getTopK`, `fname` becomes `chunk_id`.
3. Show paper title and section with each chunk in the LLM context.
4. A paper can now appear several times in the top-k; decide whether to allow that or keep only the best
   chunk per paper.
5. `TOP_K = 3` may now be too small, since three chunks carry less than three papers.
6. Delete and rebuild `chroma_db/` after any change to parsing or chunking.

---

## 5. The three representative queries and the corpus

Checked by searching the 152 parsed texts (the excluded 939-page volume could not be parsed, so this is not
proof about its contents):

- **Query 1** (fairness verifier, deep recurrent neural network) matches `DBLP_journals_pacmpl_Bastani0S19`.
- **Query 3** (replaying computation from a log; queens puzzle vs a logic-programming language) matches
  `DBLP_journals_pacmpl_KoppelSS18`.
- **Query 2** (papers comparing against an SMT-based tool) is broad: 54 parsed files mention "SMT".

Nothing suggests any query needs the excluded volume. The exact numbers asked for in query 1 (verification
time, sample count) were not checked against the text.

---

## 6. What changes in production (retrieval side)

| Concern | Capstone | Production |
|---|---|---|
| **BM25 index** | Rebuilt in memory at every startup | Persistent search engine (Elasticsearch/OpenSearch or Postgres full-text); in-memory `rank_bm25` neither scales nor shares across instances |
| **Vector index** | Local Chroma folder | Managed or server-mode store (pgvector, Qdrant, Pinecone, ...) |
| **Consistency** | Both read the same folder | BM25 and vector indexes updated together under a shared doc ID; otherwise a doc exists in one and not the other and fusion scores go wrong |
| **Versioning** | Delete `chroma_db/` by hand | Version the embedding model and chunking config; a change means re-embedding into a new index, then switching over |
| **Chunking** | Whole paper per document | Chunk to fit embedding limits and improve precision; each chunk carries doc ID and title |
| **Incremental updates** | Rebuild everything | Add, change and delete documents in both indexes when the ingestion job reports changes |

**Guiding principle:** treat the retrieval indexes as **derived data**: the source of truth is the original
PDFs plus the ingestion metadata, so the text cache, BM25 and vectors can all be rebuilt from them, which makes
recovery, migration and model upgrades routine. Ingestion-side production notes are in README_INGESTION.md.

### Known limitations to mention in the submission writeup

- BM25 index is in-memory and rebuilt at startup.
- No incremental updates (new/changed/deleted PDFs).
- Whole-document embeddings, no chunking (unless section 4 is implemented).
- Titles come from GROBID; some are wrong or missing and were only checked by automated tests.
- BM25 and vector indexes are only kept consistent because they read the same folder.
- One corpus file (`DBLP_conf_cav_BraggFRS21.pdf`, 939 pages) is a proceedings volume that exceeds GROBID's
  1M-token limit; it was excluded, leaving 152 documents.

---

## 7. Checklist

- [x] Restructured along the lab (main guard, classes, retrieval separate from the LLM step)
- [x] Reads `papers_txt/` and titles from `manifest.json`; fails clearly if the folder is missing
- [x] Chunking; same chunks in BM25 and Chroma; fused by `chunk_id`
- [x] Fixed the `zip` bug, removed the duplicate `getTopK`, `retrievedContext` returns `(scores, context)`
- [x] Vector DB rebuilds itself when chunks or chunk settings change (no manual `chroma_db/` delete)
- [x] Ran the three queries (section 8)
- [ ] Decide whether to deduplicate repeated papers (section 8 shows why it matters)
- [ ] Complete the worksheet using the evidence in `checkpoint_2_1_retrieval.log`

---

## 8. First run on the research-paper corpus

Command: `python capstone_checkpoint_2_1_baseline_retrieval_starter.py` (first run embeds 8,087 chunks, then
reuses `chroma_db/`). Top 5 chunks go to the LLM. Full output: `checkpoint_2_1_retrieval.log`.

| Query | Right paper retrieved? | Answer quality |
|---|---|---|
| 1. Fairness verifier, deep RNN, time and samples | Yes: "Probabilistic Verification of Fairness Properties via Concentration" (both copies) | **Did not answer the specific numbers.** The model said the time and sample count were not in the text it was given: the paper was found but the chunk holding the numbers was not retrieved (or the numbers are in a table GROBID garbled; not checked) |
| 2. Papers comparing against an SMT-based tool | Partly | Named ReaS (dReal, Z3) and Metric Program Synthesis (Regel); the model itself said only ReaS explicitly argues SMT does not scale. 3 of the 5 slots went to one paper ("MANTRA...") the answer did not use |
| 3. Replay a computation from a log; queens vs a logic-programming language | Yes: "Capturing the Future by Replaying the Past" (both copies) | Grounded answer: replay-based technique, N-queens compared against GNU Prolog, optimized version "just as fast" as replay-nondeterminism |

**Observations for the writeup**

- **Duplicate papers crowd the top-k.** Query 1's five chunks came from just two copies of one paper (the
  conference and arXiv versions), so they used every slot on one paper. Deduplicating by title, or capping
  chunks per paper, would free slots for other evidence.
- **Queries needing several papers (Query 2) are the weakest**, because a fixed top 5 chunks can be filled by
  one or two papers.
- **Paper found, exact detail missed (Query 1):** chunk-level retrieval returned the right paper but not the
  passage with the numbers. Larger `TOP_K`, or a different chunk size, are the levers to try.
- **A title/file mismatch to check:** `DBLP_conf_icse_JeonQFFS16.pdf` is titled "MANTRA: Synthesizing
  SMT-Validated Compliance Benchmarks for Tool-Using LLM Agents", which does not sound like a 2016 ICSE paper.
  Either GROBID picked the wrong title or the PDF is not the paper its file name suggests; not investigated.
- Results are from one run at `TOP_K=5`, weights 0.5/0.5, chunk size 1,500: no tuning was done.
