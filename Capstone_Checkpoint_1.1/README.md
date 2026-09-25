[← Repository README](../README.md) · [Next: Checkpoint 2.1 →](../Capstone_Checkpoint_2.1/README.md)

# Capstone Checkpoint 1.1 — Evaluating When Retrieval Is Required

Baseline test of `openai/gpt-5.4-mini` (via OpenRouter) on the Research Paper Navigator
scenario with **no retrieval**.

| File | What it is |
|---|---|
| `capstone_checkpoint_1_1_baseline_starter.py` | Solution file: sends the example prompts plus my scenario prompts to the LLM with no retrieval (`SCENARIO = "research_papers"`) |
| `checkpoint_1_1_responses.log` | Raw prompt/response log from the run |
| `Tarundeep_Required_Capstone_Checkpoint_1_1_Worksheet.pdf` | Completed worksheet with the analysis |

**Key finding:** Without access to the papers, the model either declines ("I don't have the paper
text") or makes up plausible-looking quotes. It can't quote verbatim, compare papers in the
collection, or tell what is and isn't in the corpus. Retrieval over the paper text is required.
