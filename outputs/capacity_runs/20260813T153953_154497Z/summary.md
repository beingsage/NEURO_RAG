# Capacity Scaling

- Run ID: `20260813T153953_154497Z`
- Timestamp UTC: `2026-08-13T15:39:53.154528+00:00`
- Seeds: `[0, 1, 2, 3, 4]`
- Cue fraction: `0.4`
- Exact cutoff: `0`

## Summary

| N | Mode | Mean retention | False retrieval | Mean margin | Memory footprint MB |
| --- | --- | --- | --- | --- | --- |
| `100` | `proxy` | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.978 ± 0.013 | 0.0 ± 0.0 |
| `100000` | `proxy` | 0.998 ± 0.001 | 0.002 ± 0.001 | 0.283 ± 0.003 | 11.3 ± 0.0 |

## Paired Tests vs N=100

| N | Metric | Mean delta | Paired t p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| `100000` | `mean_retention` | -0.002 | 0.03468 | [-0.003, -0.001] |
| `100000` | `false_retrieval_rate` | 0.002 | 0.03411 | [0.001, 0.003] |
| `100000` | `mean_margin` | -0.695 | 1.752e-08 | [-0.703, -0.686] |
