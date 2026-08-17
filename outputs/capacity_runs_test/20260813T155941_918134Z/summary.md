# Capacity Scaling

- Run ID: `20260813T155941_918134Z`
- Timestamp UTC: `2026-08-13T15:59:41.918163+00:00`
- Seeds: `[0]`
- Cue fraction: `0.4`
- Exact cutoff: `0`

## Summary

| N | Mode | Mean retention | False retrieval | Mean margin | Memory footprint MB |
| --- | --- | --- | --- | --- | --- |
| `100` | `proxy` | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.982 ± 0.000 | 0.0 ± 0.0 |
| `100000` | `proxy` | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.284 ± 0.000 | 11.3 ± 0.0 |

## Paired Tests vs N=100

| N | Metric | Mean delta | Paired t p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| `100000` | `mean_retention` | 0.000 | 1 | [0.000, 0.000] |
| `100000` | `false_retrieval_rate` | 0.000 | 1 | [0.000, 0.000] |
| `100000` | `mean_margin` | -0.698 | 1 | [-0.698, -0.698] |
