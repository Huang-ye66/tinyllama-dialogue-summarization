# Final Project Results

## Deployed model

The deployed resource-optimized model uses rank-8 QLoRA. Under the same TinyLlama, SAMSum, NF4, answer-only-loss, and training-budget conditions, reducing LoRA rank from 32 to 8 decreased adapter size from 52.71 MB to 13.23 MB, a 74.9% reduction.

Across three validation seeds, mean ROUGE-L decreased by 0.00101 and the critical-error proxy worsened by approximately 0.32%. This supports comparable quality with substantially lower adapter storage.

## Locked test comparison

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Critical-error proxy | Adapter size |
|---|---:|---:|---:|---:|---:|
| Rank 8 | 0.48385 | 0.23114 | 0.40223 | 1.04136 | 13.23 MB |
| Rank 32 | 0.48433 | 0.23417 | 0.40428 | 1.03180 | 52.71 MB |

The paired ROUGE-L difference was -0.00206 with a 10,000-sample bootstrap 95% confidence interval of [-0.00717, 0.00284]. The project therefore claims comparable quality rather than strict equivalence.

## Ablations

- Differential LoRA learning rates achieved similar ROUGE but worsened the critical-error proxy by 3.16% and did not pass the gate.
- Factual contrastive QLoRA improved strong-margin pair discrimination from 28.46% to 59.84%, but ordinary discrimination remained 93.88% and the primary gate was not met.

Both are reported as negative results and are excluded from the deployed system.

## Application acceptance

- Streamlit rendered without application-test exceptions.
- Rank-8 and rank-32 models generated sequentially without CUDA out-of-memory errors.
- The interface reports latency, generated tokens, peak VRAM, input coverage, truncation, and explainable reliability warnings.
- Model switching releases the previous model before loading the next one.
- The local health endpoint returned HTTP 200 during validation.
