# Ablation Matrix

- Run ID: `20260813T174601_179963Z`
- Timestamp UTC: `2026-08-13T17:46:01.180022+00:00`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Partial SQL cue fraction: `0.4`
- Retrieval duration: `50.0 ms`

## Summary

| Condition | Relation acc (mean ± std) | Mean CERA (mean ± std) | SQL→Graph CERA (mean ± std) |
| --- | --- | --- | --- |
| `no_delay` | 1.000 ± 0.000 | 0.092 ± 0.025 | 0.025 ± 0.079 |
| `random_delay` | 1.000 ± 0.000 | 0.094 ± 0.027 | 0.025 ± 0.079 |
| `no_stdp` | 0.925 ± 0.169 | 0.112 ± 0.040 | 0.025 ± 0.079 |
| `no_dg` | 0.250 ± 0.000 | 0.244 ± 0.049 | 0.050 ± 0.105 |
| `no_ca3_recurrence` | 1.000 ± 0.000 | 0.098 ± 0.017 | 0.025 ± 0.079 |
| `full` | 1.000 ± 0.000 | 0.099 ± 0.034 | 0.050 ± 0.105 |

## Paired Tests vs Full

| Condition | Metric | Mean delta | Paired t p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| `no_delay` | relation accuracy | 0.000 | 1 | [0.000, 0.000] |
| `no_delay` | mean CERA | -0.008 | 0.1815 | [-0.018, 0.001] |
| `random_delay` | relation accuracy | 0.000 | 1 | [0.000, 0.000] |
| `random_delay` | mean CERA | -0.006 | 0.2142 | [-0.015, -0.000] |
| `no_stdp` | relation accuracy | -0.075 | 0.1934 | [-0.200, 0.000] |
| `no_stdp` | mean CERA | 0.013 | 0.05731 | [0.002, 0.023] |
| `no_dg` | relation accuracy | -0.750 | 0 | [-0.750, -0.750] |
| `no_dg` | mean CERA | 0.144 | 3.026e-05 | [0.111, 0.180] |
| `no_ca3_recurrence` | relation accuracy | 0.000 | 1 | [0.000, 0.000] |
| `no_ca3_recurrence` | mean CERA | -0.002 | 0.8699 | [-0.020, 0.016] |
