# Experiment Summary

This document groups experiments by research question rather than by internal week or version numbers.

## Low-rank resource optimization

The resource study compared LoRA ranks under the same TinyLlama, SAMSum, NF4 QLoRA, answer-only loss, data split, and training budget. Rank 8 was selected for deployment.

- Rank-8 adapter: 13.23 MB
- Rank-32 adapter: 52.71 MB
- Storage reduction: 74.9%
- Mean validation ROUGE-L change across three seeds: -0.00101
- Locked test ROUGE-L change: -0.00206

The evidence supports a resource/quality trade-off: rank 8 is much smaller and preserves comparable summary quality, although strict numerical equivalence is not claimed.

## Hierarchical long-dialogue summarization

The direct baseline preserves the beginning and end of an oversized prompt, which can discard middle turns. The hierarchical pipeline instead:

1. splits the input at complete-turn boundaries;
2. generates a local summary for each chunk;
3. extracts a source fact table containing speakers, people, numbers, times, and amounts;
4. recursively merges local summaries when needed; and
5. checks the final output against source evidence.

On the locked test set, all 45 real long dialogues achieved 100% input coverage. Relative to direct truncation, hierarchical processing improved ROUGE-L by 0.03032 and ROUGE-1 recall by 0.06349. Latency increased because the method requires several sequential generation calls.

## Output reliability

The reliability layer identifies unsupported people, unsupported numbers, possible number omissions, duplicated events, attribution conflicts, incomplete sentences, and input truncation. High-confidence attribution conflicts can be repaired only when the original dialogue contains stronger action evidence for another speaker. Every warning remains visible for human review.

The checks are deterministic heuristics. They are not probabilities and are not used as the sole evidence for model-quality claims.

## Negative results

Two additional approaches were evaluated and retained as negative results:

- Differential LoRA learning rates produced similar ROUGE but worsened the critical-error proxy.
- Factual contrastive QLoRA improved strong-margin pair discrimination but did not improve the primary factual-quality gate.

No extra seeds were run after the preregistered gates failed. These methods are excluded from the deployed model and are not presented as successful innovations.

## Reproducibility entry points

- Application: `app.py`
- Hierarchical pipeline: `scripts/long_dialogue/hierarchical.py`
- Reliability checks: `scripts/core/reliability.py`
- Evaluation utilities: `scripts/long_dialogue/`
- UI regression: `scripts/ui_validation/smoke_ui_example.py`
- Unit tests: `tests/test_long_dialogue.py`
- Aggregate reports: `reports/long_dialogue`, `reports/output_reliability`, and `reports/project_summary`
