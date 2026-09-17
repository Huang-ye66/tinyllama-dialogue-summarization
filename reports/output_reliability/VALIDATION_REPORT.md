# Output Reliability Validation

## Problems addressed

The original demonstration could repeat local summaries, reassign an action to the wrong speaker, end with an incomplete sentence, and retain stale model-cache state. Numeric omissions were detected, but attribution conflicts and duplicated events were not.

## Changes

- Collapse only a complete repeated dialogue block of at least four turns while preserving source-index provenance.
- Instruct the merge stage to preserve attribution, remove duplicated events, and finish every sentence.
- Retry incomplete output once and keep only complete sentences if the retry remains incomplete.
- Compare near-duplicate events with original speaker-turn evidence before removing a high-confidence unsupported attribution.
- Detect duplicated events, possible attribution conflicts, incomplete sentences, unsupported entities, unsupported numbers, and potential numeric omissions.
- Release LitGPT KV-cache state after generation and synchronize the application cache state.

## Validation result

The built-in regression case now preserves complete source coverage, removes an unsupported duplicated responsibility, produces a complete summary, and reports only the remaining numeric-omission risk. The rank-8 run completed at approximately 2.13 GB peak VRAM.

Ten unit tests passed, Streamlit static application testing completed without exceptions, and local CUDA generation succeeded in direct and hierarchical modes.

## Limitation

Attribution repair is intentionally conservative. It operates only when two people are linked to highly similar actions and the original dialogue provides clearly stronger evidence for one speaker. Medium-risk numeric omissions still require human judgment because a concise summary need not reproduce every number.
