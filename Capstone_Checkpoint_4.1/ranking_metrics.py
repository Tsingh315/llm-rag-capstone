r"""Rank-aware retrieval metrics for the Checkpoint 4.1 evaluation (offline, no LLM calls).

The 4.1 evaluation reports recall only at the full budget (did the evidence reach the LLM at all?). This script
re-scores the stored runs in checkpoint_4_1_results.json with rank-aware metrics, using the order in which the
chunks were placed in the prompt:

    Recall@k   share of the target found in the first k chunks (k = 1, 3, 5, 8)
    MRR        1 / rank of the first relevant chunk (0 if none)
    nDCG@8     discounted gain of the ranking relative to the best possible ranking

Two relevance levels, both from the existing eval set (my_eval_set in the 4.1 file):
    passage    a chunk is relevant for each answer string ("evidence") it contains that no earlier chunk
               contained, so a duplicate copy of an already-seen chunk earns nothing
    paper      a chunk is relevant if it belongs to an expected paper not seen earlier in the list

Also reports cost: LLM calls and latency (measured) and the prompt size of the answer call (estimated as
characters / 4).

Run (from this folder):  python ranking_metrics.py            # writes checkpoint_4_1_ranking_metrics.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from capstone_checkpoint_4_1_advanced_retrieval_starter import (
    PAPERS_DIR, _flat, _norm_title, load_chunks, my_eval_set,
)

HERE = Path(__file__).resolve().parent
KS = (1, 3, 5, 8)
SYSTEMS = ("baseline_k5", "baseline_k8", "decomposition_only", "graph_only", "advanced")


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ideal_evidence_gains(evidence: list[str], texts: list[str]) -> list[int]:
    """Best possible gains for passage relevance: the fewest corpus chunks that cover all answer strings
    (greedy set cover), most-covering chunk first. Q1's two strings sit in one chunk, so its ideal is [2]."""
    remaining, gains = set(evidence), []
    while remaining:
        best = max(texts, key=lambda t: sum(e in t for e in remaining))
        hit = {e for e in remaining if e in best}
        if not hit:                      # string not in the corpus: it can never be retrieved
            break
        gains.append(len(hit))
        remaining -= hit
    return gains


def score_ranking(chunk_texts: list[str], chunk_titles: list[str], item: dict, ideal_ev: list[int]) -> dict:
    """Rank metrics for one retrieved list (already in prompt order)."""
    out: dict[str, float] = {}

    # Passage level: gain = number of answer strings a chunk adds that no earlier chunk had.
    seen: set[str] = set()
    ev_gains = []
    for t in chunk_texts:
        new = {e for e in item["evidence"] if e in t} - seen
        seen |= new
        ev_gains.append(len(new))
    total = len(item["evidence"])
    for k in KS:
        out[f"ev_recall@{k}"] = sum(ev_gains[:k]) / total
    first = next((i for i, g in enumerate(ev_gains) if g), None)
    out["ev_mrr"] = 0.0 if first is None else 1 / (first + 1)
    out["ev_ndcg"] = _dcg(ev_gains) / _dcg(ideal_ev) if ideal_ev else 0.0
    out["ev_first_rank"] = 0 if first is None else first + 1

    # Paper level: gain 1 for the first chunk of each expected paper.
    expected = {_norm_title(p) for p in item["papers"]}
    seen_p: set[str] = set()
    p_gains = []
    for title in chunk_titles:
        t = _norm_title(title)
        p_gains.append(1 if t in expected and t not in seen_p else 0)
        if t in expected:
            seen_p.add(t)
    for k in KS:
        out[f"paper_recall@{k}"] = sum(p_gains[:k]) / len(expected)
    first = next((i for i, g in enumerate(p_gains) if g), None)
    out["paper_mrr"] = 0.0 if first is None else 1 / (first + 1)
    out["paper_ndcg"] = _dcg(p_gains) / _dcg([1] * min(len(expected), len(p_gains)))
    return out


def evaluate(records: list[dict], chunks: list[dict]) -> list[dict]:
    by_id = {c["chunk_id"]: c for c in chunks}
    flat = {cid: _flat(c["text"]) for cid, c in by_id.items()}
    corpus = list(flat.values())
    items = {q["id"]: q for q in my_eval_set()}
    ideal = {qid: ideal_evidence_gains(q["evidence"], corpus) for qid, q in items.items()}
    rows = []
    for r in records:
        item = items[r["id"]]
        texts = [flat[c] for c in r["chunks"]]
        m = score_ranking(texts, [by_id[c]["title"] for c in r["chunks"]], item, ideal[r["id"]])
        prompt_chars = sum(len(by_id[c]["text"]) for c in r["chunks"]) + len(item["question"])
        m.update(system=r["system"], id=r["id"], run=r["run"], latency=r["t_total"], llm_calls=r["llm_calls"],
                 est_prompt_tokens=prompt_chars / 4)
        rows.append(m)
    return rows


def table(rows: list[dict], ids: list[str], title: str) -> str:
    cols = [*(f"ev_recall@{k}" for k in KS), "ev_mrr", "ev_ndcg",
            *(f"paper_recall@{k}" for k in KS), "paper_mrr", "paper_ndcg"]
    heads = ["R@1", "R@3", "R@5", "R@8", "MRR", "nDCG",
             "R@1", "R@3", "R@5", "R@8", "MRR", "nDCG"]
    lines = [f"### {title}", "",
             "| system | " + " | ".join(f"ev {h}" for h in heads[:6]) + " | "
             + " | ".join(f"paper {h}" for h in heads[6:]) + " | LLM calls | latency s | est. prompt tokens |",
             "|---" * (len(cols) + 4) + "|"]
    for s in SYSTEMS:
        rs = [r for r in rows if r["system"] == s and r["id"] in ids]
        if not rs:
            continue
        mean = {c: sum(r[c] for r in rs) / len(rs) for c in cols + ["llm_calls", "latency", "est_prompt_tokens"]}
        lines.append(f"| {s} | " + " | ".join(f"{mean[c]:.2f}" for c in cols)
                     + f" | {mean['llm_calls']:.1f} | {mean['latency']:.1f} | {mean['est_prompt_tokens']:,.0f} |")
    return "\n".join(lines)


def first_hit_table(rows: list[dict]) -> str:
    """Rank of the first chunk holding an answer string, per question and run (0 = never retrieved)."""
    ids = sorted({r["id"] for r in rows})
    lines = ["### Rank of the first answer-bearing chunk (per run; 0 = not retrieved)", "",
             "| system | " + " | ".join(ids) + " |", "|---" * (len(ids) + 1) + "|"]
    for s in SYSTEMS:
        cells = []
        for qid in ids:
            ranks = [r["ev_first_rank"] for r in sorted(rows, key=lambda r: r["run"])
                     if r["system"] == s and r["id"] == qid]
            if not ranks:
                break
            cells.append(" / ".join(str(int(x)) for x in ranks))
        if cells:
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rank-aware metrics for the stored 4.1 evaluation runs.")
    ap.add_argument("--results", type=Path, default=HERE / "checkpoint_4_1_results.json")
    ap.add_argument("--out", type=Path, default=HERE / "checkpoint_4_1_ranking_metrics.md")
    args = ap.parse_args(argv)

    records = json.loads(args.results.read_text(encoding="utf-8"))
    rows = evaluate(records, load_chunks(PAPERS_DIR))
    n_runs = len({r["run"] for r in rows})
    report = "\n\n".join([
        f"# Checkpoint 4.1 — rank-aware retrieval metrics\n\nRe-scored offline from `{args.results.name}` "
        f"({len(records)} records, {n_runs} runs; means over runs and questions). `ev` = answer strings "
        "(passage level), `paper` = expected papers. Ranks follow the order chunks were placed in the prompt "
        "(advanced: PRIMARY 1-5, then CONTEXT 1-3). Prompt tokens are estimated as characters / 4. Means are per "
        "question (macro), so R@8 differs from the pooled percentages in `checkpoint_4_1_run_output.txt` (e.g. "
        "advanced evidence 0.83 macro vs 78% pooled, because Q2's three answer strings weigh more when pooled).",
        table(rows, ["Q1", "Q2", "Q3", "Q4"], "3.1 evaluation set (Q1-Q4)"),
        table(rows, sorted({r["id"] for r in rows}), "Extended set (Q1-Q6)"),
        first_hit_table(rows),
    ])
    args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWritten to {args.out.name}")


if __name__ == "__main__":
    main()
