# Evaluation Report

Generated: 2026-06-19T16:41:25.795860+00:00
Gold file: `/Users/jkathila/Desktop/work/hackerrank-orchestrate-june26/dataset/sample_claims.csv`
Limit: full sample set

## Exact-Match Results

| strategy | rows | claim_status | issue_type | object_part | evidence_standard_met | valid_image | severity | supporting_image_ids | mean_exact | runtime | llm_calls | billable_calls | cache_hits | prompt_tokens | completion_tokens | images | est_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_stage | 20 | 70.0% | 65.0% | 85.0% | 85.0% | 90.0% | 45.0% | 80.0% | 74.3% | 129.2s | 40 | 40 | 0 | 89253 | 5912 | 29 | $0.016935 |
| single_pass | 20 | 65.0% | 45.0% | 85.0% | 85.0% | 90.0% | 50.0% | 65.0% | 69.3% | 143.2s | 20 | 20 | 0 | 50182 | 10625 | 29 | $0.013902 |

## Set-Overlap Diagnostics

| strategy | risk_flags_f1 | supporting_image_ids_f1 | predictions |
| --- | --- | --- | --- |
| two_stage | 59.7% | 86.7% | /Users/jkathila/Desktop/work/hackerrank-orchestrate-june26/code/evaluation/sample_predictions_two_stage.csv |
| single_pass | 63.2% | 78.3% | /Users/jkathila/Desktop/work/hackerrank-orchestrate-june26/code/evaluation/sample_predictions_single_pass.csv |

## Strategy Comparison

Final strategy decision: use `two_stage` for `dataset/claims.csv`.

The current v5 sample comparison favors `two_stage` on the fields that most directly drive final grading and claim interpretation. `two_stage` has higher overall mean exact-match accuracy (`74.3%` vs `69.3%`), better `claim_status` (`70.0%` vs `65.0%`), much better `issue_type` (`65.0%` vs `45.0%`), and better exact `supporting_image_ids` (`80.0%` vs `65.0%`). `single_pass` is slightly cheaper on the sample run (`$0.013902` vs `$0.016935`) and has better risk-flag set overlap (`63.2%` vs `59.7%`) and severity accuracy (`50.0%` vs `45.0%`), but its weaker issue classification is a larger risk for the final claims run.

The small glass taxonomy clarification was applied to both strategies. In the remaining `glass_shatter` disagreements, the model is engaging with the physical-observation requirement rather than ignoring it. For `user_007`, it describes the side mirror glass as having broken pieces visible while gold labels `broken_part`; this is a close label-boundary case. For `user_018`, it still reads the laptop screen as having visible separation/shattered glass while gold labels `crack`; this remains a vision-perception/taxonomy-boundary miss, not a schema-validation failure.

## Known Residual Disagreement

`user_034` remains a residual disagreement with the sample gold labels. This is not currently classified as a prompt-injection or safety failure: the image-level guard fired (`text_instruction_present` was present), and the model did not follow the visible "approve this claim" instruction as an instruction. The remaining mismatch is a genuine vision-perception edge case. The model's `physical_observations` engaged with the physical image evidence and read `img_1` as showing a torn seal with material discontinuity, while the gold label treats the seal as intact (`issue_type=none`, `claim_status=contradicted`).

We are stopping prompt iteration on this specific row. The unresolved issue is the model's visual interpretation of the seal condition, not the deterministic decision layer or the adversarial-text safety guard.

## Operational Analysis

| strategy | stage | model | calls | billable_calls | cache_hits | prompt_tokens | completion_tokens | images | avg_latency_ms | est_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_stage | claim_extraction | gpt-4.1-mini | 20 | 20 | 0 | 11964 | 685 | 0 | 1357.6 | $0.002206 |
| two_stage | vision_verification | gpt-4.1-mini | 20 | 20 | 0 | 77289 | 5227 | 29 | 5086.4 | $0.014730 |
| single_pass | single_pass | gpt-4.1-mini | 20 | 20 | 0 | 50182 | 10625 | 29 | 7146.9 | $0.013902 |

Cost estimate uses $0.1500/1M input tokens and $0.6000/1M output tokens. Override LLM_INPUT_COST_PER_1M and LLM_OUTPUT_COST_PER_1M if the selected provider/model uses different pricing.

TPM/RPM notes: the two_stage strategy makes one text call and one vision call per claim; single_pass makes one vision call per claim. llm_client uses an asyncio semaphore, exponential retry for 429/5xx style failures, SHA-256 disk caching keyed by prompt and image content, and logs cache hits so repeated evaluation runs avoid unnecessary billable model calls.
