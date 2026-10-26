# Checkpoint 8.1 - candidate comparison (3 runs, 35 questions)

## Pass count per category (mean over runs; of 5 questions)

| category | A: 4.1 pipeline + 7.1 | B: 5.1 agent + 7.1 |
|---|---|---|
| single_paper_fact | 5.0/5  (runs: 5, 5, 5) | 4.7/5  (runs: 4, 5, 5) |
| cross_paper_comparison | 1.0/5  (runs: 1, 1, 1) | 2.7/5  (runs: 2, 3, 3) |
| trend_over_time | 0.0/5  (runs: 0, 0, 0) | 0.7/5  (runs: 1, 0, 1) |
| multi_turn_follow_up | 3.3/5  (runs: 3, 4, 3) | 2.3/5  (runs: 2, 2, 3) |
| grounded_quote | 2.7/5  (runs: 3, 2, 3) | 3.7/5  (runs: 4, 4, 3) |
| not_in_collection | 5.0/5  (runs: 5, 5, 5) | 5.0/5  (runs: 5, 5, 5) |
| red_team | 5.0/5  (runs: 5, 5, 5) | 5.0/5  (runs: 5, 5, 5) |
| ALL | 22.0/35  (runs: 22, 22, 22) | 24.0/35  (runs: 23, 24, 25) |

## Evidence, tokens, calls and latency per question (means over runs)

| category | system | evidence reached | input tok | output tok | LLM calls | latency s |
|---|---|---|---|---|---|---|
| single_paper_fact | A | 100% | 3163 | 105 | 2.0 | 4.1 |
| single_paper_fact | B | 93% | 3623 | 182 | 3.0 | 5.5 |
| cross_paper_comparison | A | 23% | 3085 | 133 | 2.0 | 6.2 |
| cross_paper_comparison | B | 63% | 3596 | 234 | 3.0 | 6.8 |
| trend_over_time | A | 17% | 2956 | 174 | 2.0 | 7.4 |
| trend_over_time | B | 17% | 3484 | 243 | 3.0 | 11.6 |
| multi_turn_follow_up | A | 67% | 5531 | 302 | 5.0 | 9.4 |
| multi_turn_follow_up | B | 67% | 6702 | 429 | 7.0 | 12.7 |
| grounded_quote | A | 100% | 2749 | 92 | 2.0 | 3.6 |
| grounded_quote | B | 100% | 3278 | 155 | 3.0 | 7.9 |
| not_in_collection | A | - | 2821 | 93 | 2.0 | 4.4 |
| not_in_collection | B | - | 2811 | 136 | 2.8 | 5.2 |
| red_team | A | 100% | 2548 | 95 | 2.0 | 3.9 |
| red_team | B | 100% | 1342 | 105 | 2.3 | 2.9 |
| ALL | A | 63% | 3265 | 142 | 2.4 | 5.6 |
| ALL | B | 69% | 3548 | 212 | 3.4 | 7.5 |

## Path taken (count over all runs)

- A: 4.1 pipeline + 7.1: pipeline:multi x33, pipeline:single x72
- B: 5.1 agent + 7.1: agent:answer x14, agent:clarify x1, agent:graph_retrieve x90
