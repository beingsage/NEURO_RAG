# Ablation Matrix

- Run ID: `20260813T102449_913896Z`
- Timestamp UTC: `2026-08-13T10:24:49.913933+00:00`
- Seeds: `[0, 1, 2, 3, 4]`
- Partial SQL cue fraction: `0.4`
- Retrieval duration: `50.0 ms`

## Summary

| Condition | Relation acc (mean ± std) | Mean CERA (mean ± std) | SQL→Graph CERA (mean ± std) |
| --- | --- | --- | --- |
| `no_delay` | 0.350 ± 0.137 | 0.093 ± 0.023 | 0.050 ± 0.112 |
| `random_delay` | 0.250 ± 0.177 | 0.102 ± 0.032 | 0.100 ± 0.137 |
| `no_stdp` | 0.200 ± 0.112 | 0.118 ± 0.039 | 0.150 ± 0.137 |
| `no_dg` | 0.250 ± 0.000 | 0.237 ± 0.070 | 0.050 ± 0.112 |
| `no_ca3_recurrence` | 0.400 ± 0.137 | 0.097 ± 0.029 | 0.050 ± 0.112 |
| `full` | 0.350 ± 0.224 | 0.091 ± 0.026 | 0.050 ± 0.112 |

## Paired Tests vs Full

| Condition | Metric | Mean delta | Paired t p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| `no_delay` | relation accuracy | 0.000 | 1 | [-0.150, 0.150] |
| `no_delay` | mean CERA | 0.002 | 0.4759 | [-0.002, 0.008] |
| `random_delay` | relation accuracy | -0.100 | 0.4766 | [-0.301, 0.100] |
| `random_delay` | mean CERA | 0.011 | 0.37 | [-0.001, 0.032] |
| `no_stdp` | relation accuracy | -0.150 | 0.07048 | [-0.250, -0.050] |
| `no_stdp` | mean CERA | 0.027 | 0.236 | [-0.001, 0.066] |
| `no_dg` | relation accuracy | -0.100 | 0.3739 | [-0.250, 0.100] |
| `no_dg` | mean CERA | 0.146 | 0.008173 | [0.096, 0.199] |
| `no_ca3_recurrence` | relation accuracy | 0.050 | 0.6213 | [-0.100, 0.200] |
| `no_ca3_recurrence` | mean CERA | 0.006 | 0.746 | [-0.023, 0.038] |
