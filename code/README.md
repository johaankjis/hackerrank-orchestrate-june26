# Multi-Modal Evidence Review – Code

Damage-claim verification pipeline for the HackerRank Orchestrate hackathon.

## Quick start

```bash
cd code/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then fill in your API key(s)

# Run on the test set
python main.py --input ../dataset/claims.csv --output ../output.csv

# Quick smoke-test (5 rows, sample data)
python main.py --input ../dataset/sample_claims.csv --output sample_out.csv --limit 5

# Run both sample strategies and write evaluation/evaluation_report.md
python evaluation/main.py
```

## Project layout

| File | Purpose |
|---|---|
| `main.py` | CLI entry-point: reads claims CSV → runs pipeline → writes output.csv |
| `config.py` | Loads `.env`, validates required API keys, exports constants & paths |
| `context_loader.py` | Loads all 4 dataset CSVs; resolves images, user history, evidence requirements |
| `llm_client.py` | Async LLM wrapper with concurrency, retry, disk cache, and call logging |
| `pipeline_stages.py` | Pipeline stages: `extract_claim`, `verify_images`, `single_pass`, `decide` |
| `evaluation/main.py` | Runs/scores sample strategies and writes `evaluation_report.md` |
| `.env.example` | Template for required environment variables |
| `requirements.txt` | Python dependencies |

## Pipeline strategies

- **`two_stage`** (default): Extract claim → Verify images → Decide
- **`single_pass`**: One LLM call per row combining all stages

## LLM provider

`llm_client.call_llm()` is the only pipeline entry point for model calls:

```python
await call_llm(stage_name, model, system_prompt, user_content, images=None)
```

The provider adapter is selected with environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `LLM_PROVIDER` | `openai` | Provider adapter to use |
| `LLM_MODEL` | `gpt-4o-mini` | Model name passed to the adapter |
| `OPENAI_API_KEY` | unset | Required when `LLM_PROVIDER=openai` |
| `LLM_CONCURRENCY` | `8` | Async model-call concurrency |
| `LLM_INPUT_COST_PER_1M` | `0.15` | Evaluation cost assumption |
| `LLM_OUTPUT_COST_PER_1M` | `0.60` | Evaluation cost assumption |

Only the OpenAI adapter is implemented currently. The public `call_llm()`
signature is provider-agnostic, so another provider can be added by extending
`llm_client._call_provider()` without changing pipeline stages.

## Validation and coercion

The final `decide()` stage is pure Python. It validates all enum outputs before
returning an output row. Coercion rule:

1. lowercase and normalize spaces/punctuation to underscores,
2. apply explicit aliases such as `scrape -> scratch`,
3. use a conservative closest-match fallback for misspellings,
4. otherwise fall back to `unknown`, `none`, or `not_enough_information` as
   appropriate for the column.

History-derived flags are additive only. They can add `user_history_risk` or
`manual_review_required`, but never change claim status, issue type, object
part, severity, or evidence visibility.
