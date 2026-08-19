# Figure captions

**Figure 1 — Optimization dynamics and held-out performance.** Training exact-answer reward (raw values and trailing nine-step mean) and held-out GSM8K accuracy for GRPO, Dr.GRPO, and Kimi-OPMD with `tau=0.03`. All methods use 200 outer iterations and seed 0. Kimi performs two optimizer updates per outer iteration; therefore, the outer-iteration axis is rollout/sample aligned rather than optimizer-update matched.

**Figure 2 — Kimi-OPMD mechanism diagnostics.** Sequence-level policy-gradient force, mirror force, their absolute-force ratio, within-outer policy movement, and pre-clip global gradient norm for Kimi-OPMD with `tau=0.03` and two inner updates. The mirror force stabilizes at roughly half the PG force while policy movement remains nonzero. The gradient norm is measured before clipping at 1.0; all active updates enter the clipping regime.

**Figure 3 — Diagnostic tau sweep.** Within-run medians and interquartile ranges over active inner updates for 20-outer-iteration sweeps. Increasing `tau` monotonically reduces policy movement and increases the mirror-to-PG loss ratio; `tau=0.1` also produces the largest gradient dispersion. Intervals summarize steps within one run and are not confidence intervals across seeds.

**Figure 4 — Compute-aware held-out performance.** Held-out GSM8K accuracy plotted against cumulative optimizer updates and cumulative recorded training time. Kimi uses 400 optimizer updates versus 200 for each baseline and 1.71 recorded training hours versus 1.12 for GRPO. Recorded step time excludes unlogged startup and evaluation overhead.

## Evidence limitations

- Only seed 0 is available, so differences are descriptive and do not establish statistical significance.
- The tau sweep is a short mechanism diagnostic, not a final-accuracy comparison.
- Training-side response length is measured after text retokenization and is not identical to the vLLM token-id count.
