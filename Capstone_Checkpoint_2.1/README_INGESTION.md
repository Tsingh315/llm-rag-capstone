# PDF Ingestion (`ingest_papers.py`) — Notes & Design Decisions

Covers only the one-time PDF -> text step. For the retrieval code see [README_RETRIEVAL.md](README_RETRIEVAL.md).

**What it does:** parses the 153 research-paper PDFs in `CapstoneDatasets/ResearchPapers/` with
**GROBID**, writes clean text and titles to `papers_txt/`, reports unusually large documents before
parsing, and sanity-tests the result afterwards.

---

## 1. Decision: ingestion is a separate script

Parsing lives in its own script, not in the retrieval script.

**Why**

- **Different rates of change.** Parsing is slow, needs Docker/GROBID, and changes only when the PDFs
  change. The retrieval code is edited and re-run constantly; coupling them would re-parse 153 PDFs on
  every test run.
- **Matches production.** Ingestion is a batch job with its own deploy cadence; retrieval is the
  service that reads its output. The output folder is the contract between them.
- **Failure isolation.** If GROBID is down, retrieval keeps working over the last good output.

**Rules**

- Ingestion writes `papers_txt/`; retrieval only reads it. Retrieval must fail with a clear message if
  the folder is missing and must never trigger ingestion.
- Chunking stays on the retrieval side (size/overlap are tuned during testing), so ingestion should keep
  sections as structured data rather than flattening them. **Not done yet:** it currently stores only a
  section count in the manifest.
- **GROBID runs only during ingestion.** The script starts the container (building the image from
  `Dockerfile.grobid` on first use), waits until it is ready, and always stops it afterwards, even on
  Ctrl-C or error. If GROBID is already running it is reused and left alone. If every PDF is already up
  to date, GROBID is not started at all.

---

## 2. Running it

```bash
cd "<project folder>"
.venv/bin/python ingest_papers.py --workers 2      # all PDFs; starts and stops GROBID itself
```

| Flag | Meaning |
|---|---|
| `--workers N` | Parallel requests (default 4; **2 recommended**, see memory below) |
| `--limit N` | Only the first N PDFs (quick trial) |
| `--force` | Re-parse even PDFs that are unchanged and already OK |
| `--no-docker` | Do not manage a container; expect GROBID already running |
| `--image NAME` | Docker image to run (default `grobid-local`) |
| `--max-pages N` | Pre-flight report flags PDFs with more pages than N (default 100) |
| `--verify-only` | Skip parsing and GROBID; run only the pre-flight report and sanity tests (~21 s) |
| `--skip-verify` | Do not run the sanity tests at the end |
| `--pdf-dir`, `--out-dir`, `--grobid-url` | Override input folder, output folder (`papers_txt/`), GROBID URL |

Dependencies: `requests` (installed); `pypdf` for the independent cross-check (installed into `.venv`;
without it the report and cross-checks are skipped with a notice). The TEI XML is parsed with the standard
library, so `lxml` is not needed.

**Output contract** (what the retrieval side reads), in `papers_txt/`:

- `<pdf name>.txt`: title, abstract and body sections as plain text (references excluded).
- `manifest.json`: per PDF `pdf`, `sha256`, `status` (`ok`/`failed`), `title`, `sections` (count),
  `chars`, `error`. Re-runs skip PDFs whose sha256 is unchanged and already OK, and retry failed ones. The
  manifest is written after each paper, so an interrupted run keeps its progress.
- A PDF whose extraction is under 500 characters is recorded as failed.

**Exit code** is non-zero if a PDF failed unexpectedly or a sanity test FAILed (known issues excluded).

---

## 3. Choosing the PDF parser: GROBID vs the alternatives

**Where plain-text extractors (PyMuPDF, pypdf, pdfminer) fall short on papers:** two-column text read
across columns; headers/footers/page numbers mixed into the body; garbled equations, tables and captions;
hyphenated line breaks ("retrie-\nval") that split words and hurt BM25 tokens; the references list
drowning out the paper's own content; and no reliable title (a font-size heuristic is fragile).

| Tool | Strength | Trade-off |
|---|---|---|
| **GROBID** | Built for scholarly PDFs; returns title, authors, abstract, sections, references, clean reading order | Separate Java service (Docker); slower than PyMuPDF |
| **Docling / Marker / MinerU** | ML layout analysis, tables/equations, Markdown output | Heavier (models, ideally GPU), slower |
| **Cloud document AI** (Azure Document Intelligence, AWS Textract, Mistral OCR) | Strong on scans and tables, managed | Cost, data leaves your environment, lock-in |
| **pypdf / pdfminer.six / pypdfium2** | Permissive licences, simple | Same layout weaknesses as PyMuPDF, or worse |
| **PyMuPDF** | Very fast, easy, exposes font sizes | Weak layout handling; **AGPL** licence (paid commercial licence) |

**Licensing:** PyMuPDF is AGPL, a real concern for a closed-source production service. GROBID is
**Apache 2.0**: free for commercial use, modification and redistribution, no per-document fees; you pay
only for the infrastructure it runs on.

**Decision:** GROBID, because it identifies the paper title directly and gives sections and abstract for
sensible chunking. This was reasoned from the tools' documented behaviour and has not been compared
head-to-head on these PDFs.

**Ways to get a title** (GROBID is used; the others are fallbacks, not implemented): PDF metadata
`/Title` (often junk such as `Microsoft Word - paper.docx`, or empty); the largest text on page 1 (e.g.
PyMuPDF font sizes); an LLM on first-page text; or a bibliographic source of truth (the file names carry
DBLP keys). All of the first-page approaches assume the title is on page 1, which fails for front matter.

### 3.1 Docker (`Dockerfile.grobid`)

Builds a local image from the official `grobid/grobid` image and adds a healthcheck.

```bash
docker build -f Dockerfile.grobid -t grobid-local .                       # default 0.9.1-crf
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid-local          # manual run
curl http://localhost:8070/api/isalive                                    # prints "true"
```

- **Variants** via `--build-arg GROBID_VERSION=...`: `0.9.1-crf` (default) is CPU-only, lighter and
  faster, slightly less accurate; `0.9.1-full` uses deep-learning models, more accurate but heavier and
  best with a GPU. Both tags were confirmed on Docker Hub when this was written.
- **Startup** takes about 20-30 s to load models; the healthcheck allows a 90 s start period.
- **Pin the tag** so extraction stays reproducible. **No authentication:** keep it on a private network.
  Do not send the corpus to the public GROBID demo server.

### 3.2 Why a REST API, and is that a production problem?

GROBID is a Java application with no native Python library; its HTTP API is the official interface, and it
keeps the heavy Java process separate so it can be scaled, restarted and upgraded independently. It also
sits in the **offline ingestion job, not the query path**: users never wait on GROBID, and if it is down,
search over already-indexed documents keeps working.

| Concern | Mitigation (in the script unless noted) |
|---|---|
| Service unavailable or crashes | Retries with backoff (5 tries, 2 s factor); run under Docker/Kubernetes with health checks and auto-restart (production) |
| Busy (HTTP **503** by design when its worker pool is full) | 503 is retried; cap client concurrency with `--workers` |
| Timeouts on large or odd PDFs | Per-request timeout (10 s connect, 300 s read); a failure is recorded and the batch continues |
| Throughput | Thread pool; several GROBID instances behind a load balancer (production) |
| Exposure/security | Private network only |
| Version drift | Pin the image tag |

Threads are the right tool here: each job mostly waits on GROBID's HTTP response, so Python's GIL is not a
bottleneck. Work is balanced dynamically (each free thread takes the next PDF from the queue). Alternatives
to the hand-rolled client: `grobid_client_python` (official; batch, concurrency, retries) or GROBID's
directory batch mode.

### 3.3 From GROBID's TEI XML to text

GROBID returns **TEI XML**, not plain text. The script extracts the title, abstract and body sections and
skips the `<back>` element (references). Never embed raw XML: the tags would pollute the vectors. TEI paths
can vary between papers, so titles/sections were checked by the sanity tests, not assumed.

---

## 4. Memory and stability

GROBID idles at ~3.2 GiB and peaked at ~6.75 GiB (2 workers) of the 7.75 GiB Docker Desktop had. Two
earlier runs saw the container die mid-run. Cause **not proven**: likely memory pressure from other
containers sharing Docker's VM (spark, Kubernetes node, Odoo, etc. were running earlier in the session). A
clean run with other containers stopped and `--workers 2` finished all 153 PDFs in about 4 minutes. Close
other Docker workloads or raise Docker Desktop's memory before ingesting. A dead container makes every
remaining PDF fail with "connection refused" after long retries; Ctrl-C is safe (progress is saved and the
container is stopped). Not yet done: make the script stop early if GROBID dies.

---

## 5. Pre-flight report: unusually large documents

Runs before parsing (and before GROBID starts). It flags PDFs by **page count** (`--max-pages`, default
100), not file size: the two biggest files (48 MB, 45 MB) are ordinary 21-page papers.

Corpus: median 16 pages, p90 46, largest normal paper 76, then a cliff: **939, 620, 558, 553 and 350 pages**.
Those five are proceedings volumes (books). Only the 939-page one failed in GROBID (`TOO_MANY_TOKENS`); the
other four parsed (0.8-1.2M characters each, just under the limit) but would dominate retrieval.

---

## 6. Sanity tests (run at the end, or with `--verify-only`)

### 6.1 What each check looks at

Each test examines one file and yields one result; **a verdict is per file, never per line**. Some tests look
at only part of a file, and none can say which paragraph is wrong.

| Check | What it looks at | Level |
|---|---|---|
| `txt_exists`, `utf8` | The `.txt` exists and is valid UTF-8 | FAIL |
| `stale` | PDF's sha256 still matches the manifest | FAIL |
| `markup_leak` | TEI/XML markers (`<TEI>`, `<ref>`, `xml:id`, ...) left in the text | FAIL |
| `encoding` | More than 0.5% replacement characters | FAIL |
| `title_shape` | Title 10-300 chars, at least 2 words, not just the file name | FAIL |
| `title_on_first_pages` (needs pypdf) | Share of title words found on PDF pages 1-3 (< 60%) | FAIL if also not on pages 1-6, else WARN |
| `text_matches_pdf` (needs pypdf) | First ~4,000 chars of extracted text: share of words (5+ letters) that occur in the PDF's first 6 pages read by a *different* parser | FAIL under 70%, WARN under 85% |
| `chars_per_page` (needs pypdf) | Whole file: characters divided by page count (< 1,000) to spot truncation | WARN |
| `structure` | GROBID found fewer than 2 sections | WARN |
| `short` | Whole text under 3,000 characters | WARN |
| `alpha_ratio` | Under 60% letters: garbled or very math-heavy | WARN |
| `duplicate_title` | Another file has the same extracted title | WARN |
| `orphan_txt`, `count` | Corpus level: `.txt` files with no OK manifest entry, or counts that disagree | FAIL |

**What FAIL and WARN mean.** Neither means the parse failed: every file with an entry in `papers_txt/`
parsed. They grade the *quality* of the output. **WARN** = suspicious but probably usable. **FAIL** = the
output is probably wrong or unreliable in some respect (e.g. the title is wrong). A title FAIL does not mean
the body text is bad. The FAIL/WARN split and every threshold above are my design choices, not standards.

`KNOWN_ISSUES` in the script lists files already understood; they are reported as KNOWN, not failures.

### 6.2 Limitations of the tests

- **They do not prove the whole body was captured.** They verify the title, the first ~4,000 characters and
  an overall characters-per-page ratio. A paper missing its middle sections could still pass.
- **They cannot localise a problem** to a section or line; the unit is the file.
- **The cross-check only sees the start of each PDF** (pages 1-6) and relies on a second parser (pypdf) that
  has its own errors (two-column ordering, ligatures, hyphenation), so thresholds trade false alarms against
  misses. Volumes and front matter fail the text match because their first pages differ from the extracted
  text.
- **Without `pypdf`, a wrong title cannot be caught**: the APLAS file passed every self-consistency check.
- **Thresholds are untuned guesses.** A first version flagged 31 files for "markup leaks"; all were false
  alarms (papers legitimately contain prompt templates like `<User>`, `<answer>`), so the check now looks
  only for TEI markers.
- **Equations, tables and figures** are not checked and are often garbled by GROBID.
- Findings are only as good as my reading of them: the grouping below was done by hand from file names.

---

## 7. Results on the corpus

**Parsing:** 152 of 153 PDFs parsed; the one failure is the 939-page volume.

**Sanity tests (152 parsed):** 29 FAIL and 57 WARN findings. By distinct file: **72 clean, 63 with warnings
only, 16 with at least one FAIL, 1 known-issue** (the APLAS front matter). Run time about 21 s.

- The 16 FAIL files include no garbled normal paper. 13 are volumes or front matter: the four large volumes
  above plus nine with year-only DBLP keys (e.g. `DBLP_conf_cav_2011`); that these are front matter is
  inferred from the key pattern and low text match, **not checked file by file**.
- 3 are single papers where GROBID found no title and the file name was used:
  `DBLP_conf_nips_NyeS0L20`, `DBLP_journals_corr_GopalakrishnanH17`, `DBLP_journals_sttt_Solar-Lezama13`.
- WARNs: 29 groups of files with an identical extracted title (likely the conference and arXiv versions of
  the same paper, **not verified**; retrieval would return both), 11 low chars-per-page, 10 few sections,
  5 short texts, 2 titles not clearly on the first pages.

### 7.1 Data-quality findings

At least six files are not single papers. Treated as bad data, not parser bugs.

| File | What it is | Outcome |
|---|---|---|
| `DBLP_conf_cav_BraggFRS21.pdf` | A **939-page proceedings volume** (CAV 2021; embedded title `516232_1_En_Print.indd`), effectively a book | **Excluded.** GROBID rejects it with HTTP 500 `TOO_MANY_TOKENS` (1,166,139 tokens vs a 1,000,000 limit). Recorded as failed. Corpus = 152 documents. |
| `DBLP_conf_aplas_2009.pdf` | 8-page front matter (table of contents / organizers) | **Kept.** Parsed, but its title ("Local Arrangements Chair") is wrong; the real title, "Programming Languages and Systems", is on page 2 and is the volume name. |

**Why exclude the volume:** it breaks the one-paper-per-document model, and one giant document would match
many queries weakly and crowd out real papers.

**Writeup line:** "One corpus file (`DBLP_conf_cav_BraggFRS21.pdf`, 939 pages) is a full proceedings volume
that exceeds GROBID's 1M-token limit; it was excluded, leaving 152 documents."

**Do the three representative queries depend on the excluded volume?** Checked by searching the 152 parsed
texts (the volume itself could not be parsed, so this is not proof about its contents). Query 1 (fairness
verifier, deep RNN) matches `DBLP_journals_pacmpl_Bastani0S19`; Query 3 (replaying computation, queens vs a
logic-programming language) matches `DBLP_journals_pacmpl_KoppelSS18`; Query 2 (papers comparing against an
SMT-based tool) is broad (54 files mention "SMT"). Nothing suggests any query needs the excluded volume; the
exact numbers asked for in query 1 were not checked against the text.

**Open decisions:** exclude or flag the 13 volume/front-matter files; deduplicate the 29 repeated titles;
fix or accept the 3 missing titles.

---

## 8. What changes in production (ingestion side)

The convert-once-and-cache pattern **is** the production pattern; the difference is that it becomes a
separate, repeatable job and the cache becomes durable, versioned storage.

| Concern | Capstone | Production |
|---|---|---|
| **Where it runs** | Manually, before retrieval | Separate offline job (batch or event-triggered: file lands in S3 -> queue -> workers), never in the request path |
| **Where GROBID runs** | A container started by the script | Long-running container service (ECS/Fargate, Kubernetes, Cloud Run) behind a load balancer, several replicas. **Lambda is a poor fit** for GROBID itself (heavy Java service, ~30 s startup, 15-minute cap); it suits only the small trigger step |
| **Skip logic** | sha256 in `manifest.json` | Content hash per file: an updated PDF is re-ingested, a deleted one is removed from both indexes |
| **Storage** | Local folder + `manifest.json` | Object storage for text + a metadata table (Postgres, etc.): doc ID, title, source path, hash, ingest time. A JSON file breaks with concurrent writers |
| **Failures** | Recorded in the manifest; rerun retries them | Queue with retries and a dead-letter queue; one corrupt PDF never blocks the rest |
| **Bad PDFs** | Flagged by tests | OCR fallback for scans; volumes routed to a fallback parser or excluded |
| **Titles** | GROBID, tests flag misses | Same, plus an LLM fallback for low confidence, or DBLP records as source of truth |
| **Reliability** | One local container | Stateless replicas across availability zones, health checks (`/api/isalive`), auto-restart |
| **Calling APIs** | `requests` session with retries | Timeouts always set (`requests` has none by default), retries with backoff only on transient errors, connection reuse, concurrency limits, logging of status and latency; or `grobid_client_python` |

**Guiding principle:** treat the retrieval indexes as **derived data**. The source of truth is the original
PDFs plus the metadata; the text cache, BM25 and vectors can all be rebuilt from them.

---

## 9. Checklist

- [x] `Dockerfile.grobid` builds; `ingest_papers.py` starts and stops the container itself
- [x] Full corpus parsed (152 of 153); sanity tests run
- [ ] Decide what to do with the 13 volume/front-matter files, the 29 duplicate titles and the 3 missing titles
- [ ] Save sections as structured data (so chunking can be redone without re-parsing)
- [ ] Make the script stop early if GROBID dies
- [ ] Label oversized failures as `too_large` instead of a bare HTTP 500
