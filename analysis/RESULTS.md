# ReasoningForge result summary

| Method | Final val. | Best val. | Last-5 val. | Last-20 train | Final format |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO | 49.51% | 50.49% | 49.06% | 49.86% | 94.92% |
| Dr.GRPO | 45.90% | 47.95% | 45.88% | 47.38% | 93.95% |
| Kimi-OPMD | 50.88% | 51.37% | 49.88% | 51.72% | 96.78% |

## Descriptive findings

- Kimi-OPMD finishes 1.37 percentage points above GRPO; its last-five-validation mean is 0.82 points higher.
- This is not compute matched: Kimi uses 400 optimizer updates and 1.71 recorded train hours, versus 200 updates and 1.12 hours for GRPO.
- Across active Kimi updates, mirror/PG force has median 0.534; its last-50 mean is 0.576.
- The last-50 mean policy movement is 3.419, with zero non-finite updates.

## Interpretation guardrails

- Single seed only: differences are descriptive, not inferential.
- Tau-sweep intervals are within-run IQRs, not confidence intervals.
- The outer-iteration comparison is rollout/sample aligned, while the optimizer-update and time plots provide compute-aware views.
- Gradient norm is pre-clip; all active Kimi updates are clipped to the configured threshold.
- Training-side response lengths are retokenized text lengths, not vLLM token-id counts.
