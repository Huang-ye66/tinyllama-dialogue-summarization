# TinyLlama Dialogue Summarization

A local dialogue summarization system built with TinyLlama 1.1B, NF4 QLoRA, and Streamlit. The project targets consumer GPUs with 6 GB of VRAM and adds a speaker-aware hierarchical pipeline for long conversations that would otherwise be truncated.

The repository contains the application, inference pipeline, evaluation code, tests, and aggregate results. Model weights, datasets, caches, raw conversations, and machine-specific runtime files are intentionally excluded.

## Highlights

- Fine-tuned TinyLlama on SAMSum with answer-only loss and NF4 QLoRA.
- Compared rank-8 and rank-32 adapters under the same data and training budget.
- Added automatic routing between direct and hierarchical summarization.
- Preserved complete dialogue turns through chunking, local summaries, and recursive merging.
- Extracted speakers, people, numbers, times, and amounts to support factual review.
- Added evidence-based checks for unsupported entities, attribution conflicts, duplicated events, numeric omissions, incomplete sentences, and input truncation.
- Built a Streamlit interface with sequential model comparison, latency, generated-token, coverage, and peak-VRAM reporting.

## Verified results

| Experiment | Result |
|---|---:|
| Rank-8 adapter size | 13.23 MB |
| Rank-32 adapter size | 52.71 MB |
| Adapter-size reduction | 74.9% |
| Mean validation ROUGE-L change across three seeds, R8 vs. R32 | -0.00101 |
| Locked test ROUGE-L change, R8 vs. R32 | -0.00206 |
| Long-dialogue test coverage | 100% |
| Real long-dialogue ROUGE-L gain, hierarchical vs. truncation | +0.03032 |
| Real long-dialogue ROUGE-1 recall gain | +0.06349 |

The rank-8 model substantially reduced adapter storage while maintaining comparable quality. The long-dialogue module improved content recall by retaining middle turns, at the cost of additional inference time. These are applied engineering and evaluation contributions; the project does not claim a new foundation-model algorithm.

## System design

```text
Dialogue
   |
   +-- short input --> direct summarization
   |
   +-- long input --> complete-turn chunking
                         --> local summaries
                         --> speaker/fact table
                         --> recursive merge
                         --> reliability checks
```

Only one model is loaded at a time. This allows rank-8/rank-32 comparisons without exceeding the intended 6 GB VRAM budget.

## Repository structure

```text
app.py                                  Streamlit application
scripts/core/reliability.py             entity, number, attribution, and completeness checks
scripts/long_dialogue/hierarchical.py   chunking, local generation, merge, and attribution repair
scripts/long_dialogue/                  evaluation and reproducibility utilities
scripts/ui_validation/                  end-to-end UI smoke test
tests/test_long_dialogue.py             unit tests
docs/EXPERIMENTS.md                     experiment design and interpretation
reports/long_dialogue/FINAL_REPORT.md   aggregate long-dialogue results
reports/output_reliability/              reliability validation summary
reports/project_summary/                 final resource/quality comparison
```

## Requirements

- Windows 10/11 or Linux
- Python 3.11
- CUDA-capable GPU; the reference system used 6 GB VRAM
- A LitGPT-compatible TinyLlama checkpoint and LoRA adapter

Install the Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Large model artifacts are not stored in Git. Place compatible checkpoints at:

```text
outputs/models/resource_optimized_r8/final
outputs/models/quality_baseline_r32/final
```

Alternatively, set `TINYLLAMA_PROJECT_ROOT` to a directory containing the same `outputs/models/...` layout.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://127.0.0.1:8501/>, enter an English dialogue, choose a model and processing mode, and generate a summary. `Auto` uses direct summarization for short inputs and hierarchical processing for inputs that exceed the prompt budget.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_long_dialogue -v
.\.venv\Scripts\python.exe scripts\ui_validation\smoke_ui_example.py direct
.\.venv\Scripts\python.exe scripts\ui_validation\smoke_ui_example.py hierarchical
```

The unit suite checks empty input, Unicode, complete-turn coverage, oversized turns, deterministic chunking, repeated-block handling, attribution repair, and reliability flags.

## Experiment notes

The repository reports successful and negative experiments separately. LoRA+ differential learning rates and factual contrastive training did not pass their preregistered quality gates, so they are not presented as successful improvements. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for details.

## Limitations

- The model and evaluation are primarily designed for English conversations.
- Reliability checks are explainable heuristics, not calibrated probabilities.
- A missing number may be harmless when it is not summary-critical and still requires human review.
- Hierarchical summarization improves coverage but increases latency because it performs multiple generation passes.
- Model weights and SAMSum examples are excluded from this public repository; users must obtain them under their original licenses.

## Upstream projects and licenses

- Training and inference framework: [Lightning-AI/LitGPT](https://github.com/Lightning-AI/litgpt)
- Base model: [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- Dataset: [SAMSum](https://huggingface.co/datasets/Samsung/samsum)

Follow the licenses and terms of the upstream framework, model, and dataset. The repository intentionally contains no personal contact details, local absolute paths, credentials, or raw private documents.
