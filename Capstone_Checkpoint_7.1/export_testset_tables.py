"""Writes the test set and the last run's results as CSV and Markdown tables (with the category column)."""
import csv, json
from pathlib import Path

here = Path(__file__).parent
cases = json.loads((here / "testset_7_1.json").read_text(encoding="utf-8"))
res = json.loads((here / "checkpoint_7_1_testset_results.json").read_text(encoding="utf-8"))
hard = {r["id"]: r for r in res["hardened"]}
base = {r["id"]: r for r in res.get("baseline", [])}
flat = lambda s: " ".join(str(s).split())
md = lambda s: flat(s).replace("|", "\\|")


def question(c):
    return " -> ".join(c["turns"]) if c["turns"] else c["question"]


def write(stem, header, rows):
    with (here / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows([[flat(x) for x in r] for r in rows])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(md(x) for x in r) + " |" for r in rows]
    (here / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


write("testset_7_1", ["id", "category", "expected_behavior", "question (multi-turn: turns joined by ->)", "grading_notes", "evidence_strings"],
      [[c["id"], c["category"], c["behavior"], question(c), c["grading_notes"], "; ".join(c["evidence"]) or "-"] for c in cases])

mark = lambda r: "PASS" if r and r["passed"] else ("FAIL" if r else "-")
write("checkpoint_7_1_testset_results",
      ["id", "category", "hardened", "plain_direct", "hardened_path", "hardened_tokens", "hardened_seconds", "evidence_reached", "hardened_answer (first 300 chars)"],
      [[c["id"], c["category"], mark(hard[c["id"]]), mark(base.get(c["id"])), hard[c["id"]]["path"], hard[c["id"]]["tokens"],
        hard[c["id"]]["latency_s"], hard[c["id"]]["evidence_found"], hard[c["id"]]["answer"][:300]] for c in cases])
print("tables written")
