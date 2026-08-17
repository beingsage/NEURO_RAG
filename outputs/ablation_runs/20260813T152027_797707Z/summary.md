# Ablation Matrix

- Run ID: `20260813T152027_797707Z`
- Timestamp UTC: `2026-08-13T15:20:27.797749+00:00`
- Seeds: `[0, 1, 2, 3, 4]`
- Partial SQL cue fraction: `0.4`
- Retrieval duration: `50.0 ms`

## Summary

| Condition | Relation acc (mean ± std) | Mean CERA (mean ± std) | SQL→Graph CERA (mean ± std) |
| --- | --- | --- | --- |
| `no_delay` | 1.000 ± 0.000 | 0.116 ± 0.028 | 0.050 ± 0.112 |
| `random_delay` | 1.000 ± 0.000 | 0.106 ± 0.013 | 0.000 ± 0.000 |
| `no_stdp` | 1.000 ± 0.000 | 0.103 ± 0.019 | 0.050 ± 0.112 |
| `no_dg` | 0.250 ± 0.000 | 0.267 ± 0.048 | 0.100 ± 0.137 |
| `no_ca3_recurrence` | 1.000 ± 0.000 | 0.087 ± 0.007 | 0.000 ± 0.000 |
| `full` | 1.000 ± 0.000 | 0.109 ± 0.015 | 0.000 ± 0.000 |

## Paired Tests vs Full

| Condition | Metric | Mean delta | Paired t p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| `no_delay` | relation accuracy | 0.000 | 1 | [0.000, 0.000] |
| `no_delay` | mean CERA | 0.007 | 0.4354 | [-0.003, 0.023] |
| `random_delay` | relation accuracy | 0.000 | 1 | [0.000, 0.000] |
| `random_delay` | mean CERA | -0.003 | 0.3002 | [-0.008, 0.001] |
| `no_stdp` | relation accuracy | 0.000 | 1 | [0.000, 0.000] |
| `no_stdp` | mean CERA | -0.006 | 0.4681 | [-0.020, 0.007] |
| `no_dg` | relation accuracy | -0.750 | 0 | [-0.750, -0.750] |
| `no_dg` | mean CERA | 0.158 | 0.00206 | [0.121, 0.197] |
| `no_ca3_recurrence` | relation accuracy | 0.000 | 1 | [0.000, 0.000] |
| `no_ca3_recurrence` | mean CERA | -0.023 | 0.02576 | [-0.034, -0.011] |
