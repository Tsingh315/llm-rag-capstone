# Checkpoint 4.1 — rank-aware retrieval metrics

Re-scored offline from `checkpoint_4_1_results.json` (90 records, 3 runs; means over runs and questions). `ev` = answer strings (passage level), `paper` = expected papers. Ranks follow the order chunks were placed in the prompt (advanced: PRIMARY 1-5, then CONTEXT 1-3). Prompt tokens are estimated as characters / 4. Means are per question (macro), so R@8 differs from the pooled percentages in `checkpoint_4_1_run_output.txt` (e.g. advanced evidence 0.83 macro vs 78% pooled, because Q2's three answer strings weigh more when pooled).

### 3.1 evaluation set (Q1-Q4)

| system | ev R@1 | ev R@3 | ev R@5 | ev R@8 | ev MRR | ev nDCG | paper R@1 | paper R@3 | paper R@5 | paper R@8 | paper MRR | paper nDCG | LLM calls | latency s | est. prompt tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_k5 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.50 | 0.75 | 0.75 | 0.75 | 0.62 | 0.66 | 1.0 | 2.0 | 1,618 |
| baseline_k8 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.50 | 0.75 | 0.75 | 0.75 | 0.62 | 0.66 | 1.0 | 1.7 | 2,533 |
| decomposition_only | 0.17 | 0.33 | 0.33 | 0.33 | 0.22 | 0.25 | 0.67 | 0.75 | 0.75 | 0.75 | 0.71 | 0.72 | 2.0 | 4.0 | 2,128 |
| graph_only | 0.00 | 0.33 | 0.33 | 0.33 | 0.21 | 0.22 | 0.25 | 0.83 | 0.83 | 0.83 | 0.58 | 0.62 | 1.0 | 2.8 | 2,186 |
| advanced | 0.25 | 0.33 | 0.58 | 0.83 | 0.41 | 0.48 | 0.75 | 0.83 | 0.83 | 0.86 | 0.83 | 0.82 | 2.0 | 4.9 | 2,599 |

### Extended set (Q1-Q6)

| system | ev R@1 | ev R@3 | ev R@5 | ev R@8 | ev MRR | ev nDCG | paper R@1 | paper R@3 | paper R@5 | paper R@8 | paper MRR | paper nDCG | LLM calls | latency s | est. prompt tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_k5 | 0.17 | 0.25 | 0.33 | 0.33 | 0.22 | 0.24 | 0.50 | 0.67 | 0.67 | 0.67 | 0.58 | 0.61 | 1.0 | 1.9 | 1,624 |
| baseline_k8 | 0.17 | 0.25 | 0.33 | 0.33 | 0.22 | 0.24 | 0.50 | 0.67 | 0.67 | 0.67 | 0.58 | 0.61 | 1.0 | 1.8 | 2,506 |
| decomposition_only | 0.17 | 0.33 | 0.36 | 0.39 | 0.28 | 0.27 | 0.61 | 0.67 | 0.67 | 0.67 | 0.64 | 0.65 | 2.0 | 3.9 | 2,234 |
| graph_only | 0.00 | 0.22 | 0.22 | 0.22 | 0.14 | 0.14 | 0.33 | 0.72 | 0.72 | 0.72 | 0.56 | 0.58 | 1.0 | 2.9 | 2,326 |
| advanced | 0.25 | 0.33 | 0.56 | 0.72 | 0.44 | 0.44 | 0.67 | 0.72 | 0.72 | 0.74 | 0.72 | 0.71 | 2.0 | 5.1 | 2,483 |

### Rank of the first answer-bearing chunk (per run; 0 = not retrieved)

| system | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
|---|---|---|---|---|---|---|
| baseline_k5 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 / 1 | 3 / 3 / 3 | 0 / 0 / 0 |
| baseline_k8 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 / 1 | 3 / 3 / 3 | 0 / 0 / 0 |
| decomposition_only | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 3 / 1 | 0 / 3 / 0 | 1 / 3 / 1 | 0 / 0 / 0 |
| graph_only | 0 / 0 / 0 | 3 / 3 / 3 | 0 / 0 / 0 | 2 / 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 |
| advanced | 8 / 8 / 8 | 3 / 3 / 3 | 5 / 5 / 5 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
