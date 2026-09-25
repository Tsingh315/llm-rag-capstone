"""One-time ingestion: research-paper PDFs -> clean text + titles, via a running GROBID service.

Kept separate from the retrieval script on purpose: parsing is slow and changes rarely, while the
retrieval code is edited and re-run constantly. Run this once (and again only when the PDFs
change); the retrieval script just reads the output folder.

    python ingest_papers.py                 # all PDFs; starts GROBID in Docker, stops it when done
    python ingest_papers.py --limit 3       # quick trial
    python ingest_papers.py --no-docker     # use a GROBID you already started yourself

GROBID runs only for the duration of the run: the script starts the container (building the image from
Dockerfile.grobid on first use), waits until it is ready, and always stops it afterwards, even on Ctrl-C or
an error. If GROBID is already reachable at --grobid-url it is used as-is and left running.

Output (default: papers_txt/):
    <paper>.txt      title, abstract and body sections as plain text (references excluded)
    manifest.json    per-paper record: title, sha256 of the PDF, status, sections, error
Re-running skips PDFs whose sha256 is unchanged and that already succeeded; failed ones are retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import signal
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:                                              # independent PDF reader used only for cross-checks
    from pypdf import PdfReader
except ImportError:                               # pip install pypdf
    PdfReader = None
logging.getLogger("pypdf").setLevel(logging.ERROR)   # pypdf logs a warning per odd font; keep the report readable

HERE = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = HERE / "CapstoneDatasets" / "ResearchPapers"
DEFAULT_OUT_DIR = HERE / "papers_txt"
DEFAULT_GROBID_URL = "http://localhost:8070"
MANIFEST = "manifest.json"
DOCKERFILE = HERE / "Dockerfile.grobid"
DEFAULT_IMAGE = "grobid-local"
CONTAINER_NAME = "grobid-ingest"
STARTUP_TIMEOUT_S = 180
MAX_PAGES = 100            # typical paper: median 16 pages, p90 46. Above this it is almost surely a volume/book.

# Files we already know are not single papers; reported as KNOWN, never as failures.
KNOWN_ISSUES = {
    "DBLP_conf_cav_BraggFRS21.pdf": "939-page proceedings volume; exceeds GROBID's 1M-token limit; excluded",
    "DBLP_conf_aplas_2009.pdf": "front matter (table of contents), not a paper; extracted title is wrong",
}
TEI = {"tei": "http://www.tei-c.org/ns/1.0"}


def make_session() -> requests.Session:
    # GROBID answers 503 when its worker pool is busy; back off and retry. POST is safe to repeat.
    retry = Retry(total=5, backoff_factor=2.0, status_forcelist=[429, 502, 503, 504],
                  allowed_methods=["GET", "POST"])
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def grobid_ready(session: requests.Session, url: str) -> bool:
    try:
        return session.get(f"{url}/api/isalive", timeout=5).text.strip() == "true"
    except requests.RequestException:
        return False


@contextmanager
def grobid_container(url: str, image: str):
    """Start GROBID in Docker for the duration of the block and always stop it afterwards.

    If GROBID is already reachable at `url` it is used as-is and NOT stopped (it is not ours).
    """
    session = requests.Session()               # no retries here: we poll ourselves
    if grobid_ready(session, url):
        print(f"GROBID already running at {url}; using it and leaving it running.")
        yield
        return

    port = urlparse(url).port or 8070
    if _docker("image", "inspect", image, check=False).returncode != 0:
        print(f"Docker image '{image}' not found; building from {DOCKERFILE.name} (first run only)...")
        _docker("build", "-f", str(DOCKERFILE), "-t", image, str(HERE))
    _docker("rm", "-f", CONTAINER_NAME, check=False)          # clear a stale container of ours
    print(f"Starting GROBID container '{CONTAINER_NAME}' on port {port}...")
    _docker("run", "-d", "--rm", "--init", "--ulimit", "core=0",
            "-p", f"{port}:8070", "--name", CONTAINER_NAME, image)

    # Turn SIGTERM into an exception so the finally block below still stops the container.
    def _terminate(signum, frame):
        raise KeyboardInterrupt
    old_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not grobid_ready(session, url):
            if time.monotonic() > deadline:
                raise RuntimeError(f"GROBID not ready after {STARTUP_TIMEOUT_S}s")
            time.sleep(3)
        print("GROBID is ready.")
        yield
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        print(f"Stopping GROBID container '{CONTAINER_NAME}'...")
        _docker("stop", CONTAINER_NAME, check=False)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_tei(xml: str) -> dict:
    """TEI XML -> {"title", "abstract", "sections": [{"heading", "text"}]}. References (<back>) are skipped."""
    root = ET.fromstring(xml)
    title = _text(root.find(".//tei:teiHeader//tei:titleStmt/tei:title", TEI))
    abstract = _text(root.find(".//tei:profileDesc/tei:abstract", TEI))
    sections = []
    for div in root.findall(".//tei:text/tei:body/tei:div", TEI):
        heading = _text(div.find("tei:head", TEI))
        body = " ".join(_text(p) for p in div.findall("tei:p", TEI)).strip()
        if body:
            sections.append({"heading": heading, "text": body})
    return {"title": title, "abstract": abstract, "sections": sections}


def render_text(parsed: dict) -> str:
    parts = [parsed["title"]]
    if parsed["abstract"]:
        parts.append("Abstract\n" + parsed["abstract"])
    for s in parsed["sections"]:
        parts.append((s["heading"] + "\n" if s["heading"] else "") + s["text"])
    return "\n\n".join(p for p in parts if p)


def process_pdf(session: requests.Session, url: str, pdf: Path, out_dir: Path,
                timeout: tuple[int, int]) -> dict:
    """Parse one PDF. Never raises: failures are returned as a record with status 'failed'."""
    record = {"pdf": pdf.name, "sha256": sha256_of(pdf), "status": "failed", "title": "",
              "sections": 0, "chars": 0, "error": None}
    try:
        with pdf.open("rb") as fh:
            r = session.post(f"{url}/api/processFulltextDocument", files={"input": fh},
                             data={"consolidateHeader": "0"}, timeout=timeout)
        r.raise_for_status()
        parsed = parse_tei(r.text)
        text = render_text(parsed)
        if len(text) < 500:                       # near-empty: scanned PDF or a GROBID miss
            raise ValueError(f"extraction too short ({len(text)} chars)")
        (out_dir / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        record.update(status="ok", title=parsed["title"] or pdf.stem,
                      sections=len(parsed["sections"]), chars=len(text))
    except Exception as exc:                      # one bad PDF must not stop the batch
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


# --------------------------------------------------------------------------------------------
# Pre-flight: flag unusually large documents BEFORE parsing
# --------------------------------------------------------------------------------------------
def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def page_count(pdf: Path) -> int | None:
    if PdfReader is None:
        return None
    try:
        return len(PdfReader(str(pdf)).pages)
    except Exception:
        return None


def oversized_report(pdfs: list[Path], manifest: dict, max_pages: int) -> list[str]:
    """Print a report of PDFs whose page count is far above a normal paper. Returns their names.

    Page count is the signal, not file size: a 48 MB PDF can be a normal 21-page paper full of images,
    while a 350+ page proceedings volume is a book and breaks the one-paper-per-document model.
    """
    print("\n== Pre-flight: unusually large documents ==")
    if PdfReader is None:
        print("  skipped (pip install pypdf to enable)")
        return []
    pages = {p.name: page_count(p) for p in pdfs}
    known = sorted(n for n in pages.values() if n)
    if known:
        print(f"  pages per PDF: median={known[len(known) // 2]}  p90={known[int(0.9 * len(known))]}  "
              f"max={known[-1]}  (threshold: > {max_pages})")
    flagged = sorted((p for p in pdfs if (pages[p.name] or 0) > max_pages), key=lambda p: -(pages[p.name] or 0))
    unreadable = [p.name for p in pdfs if pages[p.name] is None]
    for p in flagged:
        rec = manifest.get(p.name, {})
        parsed = f"parsed, {rec['chars']:,} chars" if rec.get("status") == "ok" else \
            (f"FAILED: {rec['error'][:60]}" if rec.get("error") else "not parsed yet")
        print(f"  LARGE  {p.name:<46} {pages[p.name]:>4} pp  {p.stat().st_size / 1e6:6.1f} MB  [{parsed}]")
    if unreadable:
        print(f"  page count unreadable: {', '.join(unreadable)}")
    if flagged:
        print(f"  -> {len(flagged)} document(s) look like volumes/books, not single papers. They may fail in "
              "GROBID (1M-token limit) or, if parsed, dominate retrieval; consider excluding them.")
    else:
        print("  none")
    return [p.name for p in flagged]


# --------------------------------------------------------------------------------------------
# Sanity tests: was the PDF -> text conversion accurate?
# --------------------------------------------------------------------------------------------
def verify_output(pdfs: list[Path], out_dir: Path, manifest: dict) -> int:
    """Run sanity tests over the output. Returns the number of unexpected FAILs.

    Layer 1 (self-consistency, no extra dependency): file/markup/encoding/title/staleness checks.
    Layer 2 (independent cross-check, needs pypdf): re-reads the PDF with a DIFFERENT parser and checks that
    (a) the extracted title's words appear on the first pages, and (b) the start of the extracted text
    is made of words that really occur in the PDF. Layer 1 alone cannot catch a wrong title.
    Limits: it does not prove the whole body was captured, only its start plus a length-per-page check.
    """
    fails: list[tuple[str, str, str]] = []
    warns: list[tuple[str, str, str]] = []
    pdf_by_name = {p.name: p for p in pdfs}
    titles: dict[str, list[str]] = {}
    checked = 0

    fail_files: set[str] = set()
    warn_files: set[str] = set()

    def fail(f, check, detail):
        fails.append((f, check, detail)); fail_files.add(f)

    def warn(f, check, detail):
        warns.append((f, check, detail)); warn_files.update(x.strip() for x in f.split(","))

    for name, rec in sorted(manifest.items()):
        if rec.get("status") != "ok" or name not in pdf_by_name:
            continue
        checked += 1
        pdf = pdf_by_name[name]
        txt_path = out_dir / f"{pdf.stem}.txt"

        # ---- Layer 1: self-consistency
        if not txt_path.exists():
            fail(name, "txt_exists", "manifest says ok but the .txt file is missing")
            continue
        try:
            text = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            fail(name, "utf8", "text file is not valid UTF-8")
            continue
        if sha256_of(pdf) != rec.get("sha256"):
            fail(name, "stale", "PDF changed since it was parsed; rerun ingestion")
        # Only GROBID/TEI-specific markers: papers legitimately contain things like <User> or <answer>
        # (prompt templates), so a generic "looks like a tag" test gives false alarms.
        if re.search(r"</?(?:TEI|teiHeader|biblStruct|figDesc|formula|ref)\b[^>]{0,80}>|xmlns|xml:id", text):
            fail(name, "markup_leak", "TEI/XML markup left in the text")
        if text.count("\ufffd") > 0.005 * max(len(text), 1):
            fail(name, "encoding", f"{text.count(chr(0xfffd))} replacement characters")
        title = rec.get("title", "")
        if not (10 <= len(title) <= 300) or title == pdf.stem or len(_norm_tokens(title)) < 2:
            fail(name, "title_shape", f"implausible title: {title!r}")
        titles.setdefault(" ".join(_norm_tokens(title)), []).append(name)
        nonspace = [c for c in text if not c.isspace()]
        if nonspace and sum(c.isalpha() for c in nonspace) / len(nonspace) < 0.6:
            warn(name, "alpha_ratio", "less than 60% letters: garbled text or very math-heavy")
        if rec.get("sections", 0) < 2:
            warn(name, "structure", f"only {rec.get('sections', 0)} section(s) found")
        if len(text) < 3000:
            warn(name, "short", f"only {len(text):,} characters")

        # ---- Layer 2: independent cross-check with pypdf
        if PdfReader is None:
            continue
        try:
            reader = PdfReader(str(pdf))
            n_pages = len(reader.pages)
            page_texts = [reader.pages[i].extract_text() or "" for i in range(min(6, n_pages))]   # once
            first_pages = " ".join(page_texts)
        except Exception as exc:
            warn(name, "cross_check", f"pypdf could not read the PDF ({type(exc).__name__})")
            continue
        page_words = set(_norm_tokens(first_pages))
        title_words = [w for w in _norm_tokens(title) if len(w) >= 3]
        if title_words and page_words:
            cov = sum(w in page_words for w in title_words) / len(title_words)
            first3 = set(_norm_tokens(" ".join(page_texts[:3])))
            cov3 = sum(w in first3 for w in title_words) / len(title_words)
            if cov3 < 0.6:
                (fail if cov < 0.6 else warn)(name, "title_on_first_pages",
                    f"only {cov3:.0%} of title words on pages 1-3 (title: {title[:60]!r})")
        sample = {w for w in _norm_tokens(text[:4000]) if len(w) >= 5}
        if sample and page_words:
            cov = len(sample & page_words) / len(sample)
            if cov < 0.70:
                fail(name, "text_matches_pdf", f"only {cov:.0%} of extracted words occur in the PDF's first pages")
            elif cov < 0.85:
                warn(name, "text_matches_pdf", f"{cov:.0%} of extracted words occur in the PDF's first pages")
        if n_pages and len(text) / n_pages < 1000:
            warn(name, "chars_per_page", f"{len(text) / n_pages:,.0f} chars/page: parse may be truncated")

    # ---- corpus-level
    for key, names in titles.items():
        if len(names) > 1 and key:
            warn(", ".join(names), "duplicate_title", "same title extracted for several files")
    ok_stems = {Path(n).stem for n, r in manifest.items() if r.get("status") == "ok"}
    orphans = sorted(t.name for t in out_dir.glob("*.txt") if t.stem not in ok_stems)
    if orphans:
        fail("(corpus)", "orphan_txt", f"{len(orphans)} .txt file(s) with no ok manifest entry: {orphans[:3]}")
    n_txt = len(list(out_dir.glob("*.txt")))
    if n_txt != sum(r.get("status") == "ok" for r in manifest.values()):
        fail("(corpus)", "count", f"{n_txt} .txt files vs {sum(r.get('status') == 'ok' for r in manifest.values())} ok records")

    # ---- report
    unexpected = [f for f in fails if f[0] not in KNOWN_ISSUES]
    known = [f for f in fails + warns if f[0] in KNOWN_ISSUES]
    print("\n== Sanity tests: PDF -> text accuracy ==")
    print(f"  papers checked: {checked}   independent cross-check (pypdf): "
          f"{'ON' if PdfReader else 'OFF - pip install pypdf'}")
    for label, items in (("FAIL", unexpected), ("WARN", [w for w in warns if w[0] not in KNOWN_ISSUES])):
        for f, check, detail in items:
            print(f"  {label}  {f if ',' in f else f'{f:<44}'} {check}: {detail}")
    for f, check, detail in known:
        print(f"  KNOWN {f[:44]:<44} {check}: {detail}  [{KNOWN_ISSUES[f]}]")
    known_files = set(KNOWN_ISSUES)
    ok_files = {n for n, r in manifest.items() if r.get("status") == "ok"}
    f_files = (fail_files - known_files) & ok_files
    w_files = (warn_files - known_files - f_files) & ok_files
    clean = ok_files - f_files - w_files - known_files
    print(f"  findings: {len(unexpected)} FAIL, {len([w for w in warns if w[0] not in KNOWN_ISSUES])} WARN, "
          f"{len(known)} known-issue")
    print(f"  files ({len(ok_files)} parsed): {len(clean)} clean, {len(w_files)} with warnings only, "
          f"{len(f_files)} with at least one FAIL, {len(ok_files & known_files)} known-issue")
    return len(unexpected)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--grobid-url", default=DEFAULT_GROBID_URL)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (keep <= GROBID's worker pool)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N PDFs (0 = all)")
    ap.add_argument("--no-docker", action="store_true",
                    help="do not manage a container; expect GROBID to be running already")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to run (default: %(default)s)")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES,
                    help="pre-flight report flags PDFs with more pages than this (default: %(default)s)")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip parsing (and GROBID); only run the pre-flight report and sanity tests")
    ap.add_argument("--skip-verify", action="store_true", help="do not run the sanity tests at the end")
    ap.add_argument("--force", action="store_true", help="re-parse even if unchanged and already ok")
    args = ap.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"No PDFs found in {args.pdf_dir}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    oversized_report(pdfs, manifest, args.max_pages)

    todo = []
    for pdf in pdfs:
        prev = manifest.get(pdf.name)
        unchanged = prev and prev["status"] == "ok" and prev["sha256"] == sha256_of(pdf) \
            and (args.out_dir / f"{pdf.stem}.txt").exists()
        if not unchanged or args.force:
            todo.append(pdf)
    print(f"{len(pdfs)} PDFs, {len(pdfs) - len(todo)} up to date, {len(todo)} to parse "
          f"({args.workers} workers)")

    if todo and not args.verify_only:             # nothing to parse -> don't start GROBID at all
        session = make_session()
        if args.no_docker:
            if not grobid_ready(session, args.grobid_url):
                print(f"GROBID is not reachable/ready at {args.grobid_url}. Start it, or drop --no-docker.")
                return 1
            ctx = nullcontext()
        else:
            ctx = grobid_container(args.grobid_url, args.image)

        with ctx:
            done = 0
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                futures = [pool.submit(process_pdf, session, args.grobid_url, pdf, args.out_dir, (10, 300))
                           for pdf in todo]
                for fut in as_completed(futures):
                    rec = fut.result()
                    manifest[rec["pdf"]] = rec
                    done += 1
                    flag = "ok  " if rec["status"] == "ok" else "FAIL"
                    print(f"[{done}/{len(todo)}] {flag} {rec['pdf']}  {rec['title'][:70] or rec['error']}")
                    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
            except BaseException:                 # Ctrl-C / error: drop queued work so shutdown is quick
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown()

    failed = [n for n, r in manifest.items() if r["status"] != "ok" and (args.pdf_dir / n).exists()]
    print(f"\nDone. ok={sum(r['status'] == 'ok' for r in manifest.values())} failed={len(failed)}")
    if failed:
        print("Failed (rerun to retry):", ", ".join(f"{n}{' [known]' if n in KNOWN_ISSUES else ''}" for n in failed))
    unexpected_failed = [n for n in failed if n not in KNOWN_ISSUES]
    bad = 0 if args.skip_verify else verify_output(pdfs, args.out_dir, manifest)
    return 1 if (unexpected_failed or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
