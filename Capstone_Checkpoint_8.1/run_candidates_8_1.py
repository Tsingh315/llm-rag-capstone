"""Compare the two production candidates on the 35-question test set (reused from Checkpoint 7.1).

    python run_candidates_8_1.py --runs 3                 # A and B, 3 runs each
    python run_candidates_8_1.py --ids SP1,RT5 --runs 1   # a quick subset
    python run_candidates_8_1.py --price-in 0.25 --price-out 2.0   # optional: $ per 1M tokens, to add a cost column

Per run and system it records: pass/fail (LLM judge for answers, refusal format for refusals), evidence reaching the
LLM, real input/output tokens, LLM calls, latency and the path taken. Cost is reported in tokens (the repo's
proxy); dollars only if you pass your model's prices.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics as st
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import capstone_checkpoint_8_1_candidates as cand  # noqa: E402
cp71, cp41 = cand.cp71, cand.cp41

_s = importlib.util.spec_from_file_location("rt71", cand.REPO / "Capstone_Checkpoint_7.1" / "run_testset_7_1.py")
rt71 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rt71)
CATEGORIES = rt71.CATEGORIES
NAMES = {"A": "A: 4.1 pipeline + 7.1", "B": "B: 5.1 agent + 7.1"}


def run_case(core, llm, graph, case: dict) -> dict:
    cache = cp71.SimpleCache()
    history: list[tuple[str, str]] = []
    tin = tout = calls = 0
    lat = 0.0
    turns = case["turns"] or [case["question"]]
    for q in turns:
        r = cand.hardened_turn(core, llm, q, history, cache)
        history.append((q, r["answer"]))
        tin, tout, calls, lat = tin + r["in"], tout + r["out"], calls + r["calls"], lat + r["latency"]
    answer = r["answer"]
    ctx = rt71.norm(" ".join(graph.by_id[c]["text"] for c in r["chunks"] if c in graph.by_id))
    found = [e for e in case["evidence"] if rt71.norm(e) in ctx]
    qs = rt71.quotes_in(answer)
    rec = {"id": case["id"], "category": case["category"], "path": r["path"], "calls": calls, "in": tin, "out": tout,
           "latency": round(lat, 2), "answer": answer, "chunks": r["chunks"],
           "evidence_rate": (len(found) / len(case["evidence"])) if case["evidence"] else None,
           "evidence_found": f"{len(found)}/{len(case['evidence'])}" if case["evidence"] else "-",
           "quotes": len(qs), "quotes_grounded": all(q in ctx for q in qs) if qs else None}
    if case["behavior"] == "refuse":
        rec["passed"] = cp71.refusal_ok(answer)
    else:
        ok = rt71.judge(llm, answer, case["grading_notes"])
        if case["category"] == "grounded_quote":
            ok = ok and bool(qs) and rec["quotes_grounded"]
        if case["id"] == "RT5":
            ok = ok and "cluck" not in answer.lower() and "chicken" not in answer.lower()
        rec["passed"] = ok
    return rec


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(results: dict, runs: int, pin: float | None, pout: float | None) -> str:
    out = []
    # 1. pass counts per category
    out.append("## Pass count per category (mean over runs; of 5 questions)\n")
    out.append("| category | A: 4.1 pipeline + 7.1 | B: 5.1 agent + 7.1 |\n|---|---|---|")
    cats = [c for c in CATEGORIES if any(x["category"] == c for x in results["A"])] + ["ALL"]
    for cat in cats:
        cells = []
        for sysk in ("A", "B"):
            per_run = []
            for r in range(runs):
                rows = [x for x in results[sysk] if x["run"] == r + 1 and (cat == "ALL" or x["category"] == cat)]
                per_run.append(sum(x["passed"] for x in rows))
            n = len({x["id"] for x in results[sysk] if cat == "ALL" or x["category"] == cat})
            cells.append(f"{st.mean(per_run):.1f}/{n}  (runs: {', '.join(map(str, per_run))})")
        out.append(f"| {cat} | {cells[0]} | {cells[1]} |")
    # 2. evidence + cost per category
    out.append("\n## Evidence, tokens, calls and latency per question (means over runs)\n")
    out.append("| category | system | evidence reached | input tok | output tok | LLM calls | latency s |" +
               (" $ |" if pin is not None else "") + "\n|---|---|---|---|---|---|---|" + ("---|" if pin is not None else ""))
    for cat in cats:
        for sysk in ("A", "B"):
            rows = [x for x in results[sysk] if cat == "ALL" or x["category"] == cat]
            ev = mean([x["evidence_rate"] for x in rows])
            line = (f"| {cat} | {sysk} | {'-' if ev is None else f'{100 * ev:.0f}%'} | {mean([x['in'] for x in rows]):.0f} | "
                    f"{mean([x['out'] for x in rows]):.0f} | {mean([x['calls'] for x in rows]):.1f} | "
                    f"{mean([x['latency'] for x in rows]):.1f} |")
            if pin is not None:
                cost = mean([x["in"] * pin / 1e6 + x["out"] * pout / 1e6 for x in rows])
                line += f" {cost:.5f} |"
            out.append(line)
    # 3. paths
    out.append("\n## Path taken (count over all runs)\n")
    for sysk in ("A", "B"):
        c: dict[str, int] = {}
        for x in results[sysk]:
            c[x["path"]] = c.get(x["path"], 0) + 1
        out.append(f"- {NAMES[sysk]}: " + ", ".join(f"{k} x{v}" for k, v in sorted(c.items())))
    return "\n".join(out)


def question_table(cases: list[dict], results: dict, runs: int) -> list[list]:
    rows = []
    for c in cases:
        row = [c["id"], c["category"]]
        for sysk in ("A", "B"):
            rs = [x for x in results[sysk] if x["id"] == c["id"]]
            ev = mean([x["evidence_rate"] for x in rs])
            row += [f"{sum(x['passed'] for x in rs)}/{len(rs)}", "-" if ev is None else f"{100 * ev:.0f}%",
                    round(mean([x["in"] + x["out"] for x in rs])), round(mean([x["latency"] for x in rs]), 1)]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ids", default="")
    ap.add_argument("--price-in", type=float, default=None, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="$ per 1M output tokens")
    ap.add_argument("--out", default="checkpoint_8_1_results")
    ap.add_argument("--resume", action="store_true", help="keep finished (system, run) blocks from the saved JSON")
    a = ap.parse_args()
    cases = json.loads((cand.REPO / "Capstone_Checkpoint_7.1" / "testset_7_1.json").read_text(encoding="utf-8"))
    if a.ids:
        cases = [c for c in cases if c["id"] in set(a.ids.split(","))]
    llm = cp71.make_llm()
    llm.request_timeout, llm.max_retries = 90, 2      # a hung API call must not stall the whole run
    cores, graph = cand.build_candidates(llm)
    results: dict[str, list[dict]] = {"A": [], "B": []}
    if a.resume and (HERE / f"{a.out}.json").exists():
        saved = json.loads((HERE / f"{a.out}.json").read_text(encoding="utf-8"))
        results = {k: [x for x in saved[k]] for k in ("A", "B")}
    for run in range(1, a.runs + 1):
        for sysk in ("A", "B"):
            if sum(1 for x in results[sysk] if x["run"] == run) >= len(cases):
                continue                                   # block already complete
            results[sysk] = [x for x in results[sysk] if x["run"] != run]   # drop a partial block
            for c in cases:
                try:
                    rec = run_case(cores[sysk], llm, graph, c)
                except Exception as e:                       # one bad question must not lose the run
                    print(f"[{sysk} run{run}] {c['id']} ERROR {type(e).__name__}: {e}", flush=True)
                    rec = {"id": c["id"], "category": c["category"], "path": "error", "calls": 0, "in": 0, "out": 0,
                           "latency": 0.0, "answer": f"ERROR {e}", "chunks": [], "evidence_rate": None,
                           "evidence_found": "-", "quotes": 0, "quotes_grounded": None, "passed": False}
                rec.update(run=run, system=sysk)
                results[sysk].append(rec)
                print(f"[{sysk} run{run}] {rec['id']:4} {'PASS' if rec['passed'] else 'FAIL'} {rec['path']:22} "
                      f"in={rec['in']:5} out={rec['out']:4} calls={rec['calls']} {rec['latency']:5.1f}s "
                      f"ev={rec['evidence_found']:4} | {rec['answer'][:70]!r}", flush=True)
            (HERE / f"{a.out}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    runs = a.runs
    md = summarize(results, runs, a.price_in, a.price_out)
    (HERE / f"{a.out}.md").write_text(f"# Checkpoint 8.1 - candidate comparison ({runs} runs, {len(cases)} questions)\n\n{md}\n",
                                      encoding="utf-8")
    with (HERE / f"{a.out}_per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "A_pass", "A_evidence", "A_tokens", "A_seconds", "B_pass", "B_evidence", "B_tokens", "B_seconds"])
        w.writerows(question_table(cases, results, runs))
    print("\n" + md)


if __name__ == "__main__":
    main()
