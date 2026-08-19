# Render QA ledger

| Issue | Artifact | Severity | Fix | Status |
| --- | --- | --- | --- | --- |
| Single seed cannot support confidence intervals | All figures | High | Show raw/rolling trajectories and label within-run IQR explicitly | Passed |
| Precomputed tau-sweep ratio averaged unstable microbatch ratios | Figure 3 | High | Recompute ratio from aggregated mirror loss and PG loss | Passed |
| Gradient norm is pre-clip, not post-clip | Figure 2 | Medium | Label axis and caption as pre-clip; show clipping threshold | Passed |
| Retokenized response length can exceed vLLM max token count | Table/caption | Medium | Report as a measurement caveat; do not imply generation exceeded its API limit | Passed |
| Panel labels collided with long titles | Figures 2–3 | Medium | Move labels outward and shorten the mechanism title | Passed after rerender |
| Color-only force distinction | Figure 2a | Medium | Add a dashed mirror-force line style | Passed |
| Raster/vector export integrity | All figures | Medium | Export PNG/SVG/PDF and inspect rendered PNGs | Passed |
| Outer iterations are not optimizer-update matched | Figures 1 and 4 | High | Keep Figure 1 as rollout/sample view and add optimizer-update/time views | Passed |
