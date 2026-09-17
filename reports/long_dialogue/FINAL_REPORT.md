# Hierarchical Long-Dialogue Evaluation

## Method

The method keeps the deployed rank-8 model unchanged. Inputs that exceed the 412-token formatted-prompt budget are split into complete-turn chunks of at most 280 tokens. Each chunk is summarized locally, then merged with a fact table extracted from the full source dialogue. The direct baseline uses the same model with head-and-tail truncation.

## Validation set

The development evaluation contained 38 real long dialogues and 30 deterministic synthetic stress cases.

| Metric | Hierarchical minus direct |
|---|---:|
| ROUGE-L | +0.03683 |
| ROUGE-L 95% paired-bootstrap CI | [0.00573, 0.06653] |
| ROUGE-1 recall | +0.09011 |
| Critical-error proxy | -1.88% |

The hierarchical pipeline achieved 100% input coverage, used at most 2.14 GB peak VRAM, and increased mean latency from 2.96 to 10.21 seconds. It passed the locked engineering and quality gates before the test split was evaluated.

## Locked test set

The test evaluation contained 45 real long dialogues and 30 synthetic stress cases.

| Group | ROUGE-L change | ROUGE-1 recall change | Critical-error proxy change |
|---|---:|---:|---:|
| All 75 samples | +0.04936 | +0.08941 | -11.75% |
| 45 real long dialogues | +0.03032 | +0.06349 | -14.96% |
| 30 synthetic stress cases | +0.07791 | +0.12829 | -8.86% |

For all test samples, input coverage was 100% and peak VRAM was 2.14 GB. Mean latency increased from 2.46 to 7.73 seconds. The result supports the claim that hierarchical processing prevents middle-turn deletion and improves automatic long-dialogue metrics, with a clear latency cost.

## Human evaluation status

A fixed 30-item blinded human-AI collaborative review sheet was prepared with 15 real and 15 synthetic cases. Manual labels were not completed, so this report does not claim that the predefined human-evaluation gate was passed.
