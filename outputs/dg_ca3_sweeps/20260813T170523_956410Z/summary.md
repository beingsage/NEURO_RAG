# DG / CA3 Sweep

- Run ID: `20260813T170523_956410Z`
- Timestamp UTC: `2026-08-13T17:05:23.956440+00:00`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Fixed bridge_lr: `0.01`
- Fixed DG output_dim: `2000`
- Fixed CA1 epochs/lr: `12 / 0.03`

## dg_sparsity

- Parameter: `dg_target_sparsity`
- Baseline: `0.01`

| Value | Relation acc | False retrieval | Separation margin | Mean retention | Best impostor | Pairwise overlap |
| --- | --- | --- | --- | --- | --- | --- |
| `0.005` | 1.000 ± 0.000 | 0.125 ± 0.177 | 0.243 ± 0.138 | 0.829 ± 0.085 | 0.587 ± 0.144 | 0.433 ± 0.156 |
| `0.01` | 1.000 ± 0.000 | 0.325 ± 0.169 | 0.054 ± 0.049 | 0.878 ± 0.031 | 0.824 ± 0.049 | 0.712 ± 0.066 |
| `0.015` | 1.000 ± 0.000 | 0.250 ± 0.289 | 0.006 ± 0.043 | 0.931 ± 0.049 | 0.925 ± 0.048 | 0.861 ± 0.080 |
| `0.02` | 1.000 ± 0.000 | 0.450 ± 0.258 | -0.018 ± 0.024 | 0.908 ± 0.054 | 0.926 ± 0.048 | 0.916 ± 0.043 |
| `0.03` | 1.000 ± 0.000 | 0.725 ± 0.079 | -0.063 ± 0.023 | 0.683 ± 0.161 | 0.746 ± 0.142 | 0.875 ± 0.079 |

### Paired Tests vs Baseline

| Value | Metric | Mean delta | Paired t p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| `0.005` | `false_retrieval_rate` | -0.200 | 0.02237 | [-0.325, -0.050] |
| `0.005` | `separation_margin_mean` | 0.189 | 0.0007386 | [0.119, 0.260] |
| `0.005` | `mean_retention` | -0.049 | 0.1119 | [-0.098, 0.003] |
| `0.005` | `best_impostor_overlap_mean` | -0.238 | 0.0004507 | [-0.319, -0.153] |
| `0.015` | `false_retrieval_rate` | -0.075 | 0.4961 | [-0.250, 0.125] |
| `0.015` | `separation_margin_mean` | -0.048 | 0.02603 | [-0.080, -0.014] |
| `0.015` | `mean_retention` | 0.053 | 0.006069 | [0.027, 0.081] |
| `0.015` | `best_impostor_overlap_mean` | 0.101 | 0.003867 | [0.052, 0.150] |
| `0.02` | `false_retrieval_rate` | 0.125 | 0.1382 | [-0.025, 0.275] |
| `0.02` | `separation_margin_mean` | -0.073 | 0.0009488 | [-0.102, -0.045] |
| `0.02` | `mean_retention` | 0.030 | 0.1283 | [-0.002, 0.063] |
| `0.02` | `best_impostor_overlap_mean` | 0.102 | 0.002928 | [0.056, 0.150] |
| `0.03` | `false_retrieval_rate` | 0.400 | 4.24e-06 | [0.325, 0.475] |
| `0.03` | `separation_margin_mean` | -0.117 | 8.945e-05 | [-0.151, -0.086] |
| `0.03` | `mean_retention` | -0.195 | 0.002108 | [-0.290, -0.127] |
| `0.03` | `best_impostor_overlap_mean` | -0.078 | 0.1364 | [-0.176, -0.000] |

## ca3_fanout

- Parameter: `dg_bridge_fanout`
- Baseline: `12`

| Value | Relation acc | False retrieval | Separation margin | Mean retention | Best impostor | Pairwise overlap |
| --- | --- | --- | --- | --- | --- | --- |
| `4` | 1.000 ± 0.000 | 0.275 ± 0.249 | 0.119 ± 0.078 | 0.829 ± 0.037 | 0.709 ± 0.078 | 0.563 ± 0.100 |
| `8` | 1.000 ± 0.000 | 0.275 ± 0.249 | 0.119 ± 0.078 | 0.829 ± 0.037 | 0.709 ± 0.078 | 0.563 ± 0.100 |
| `12` | 1.000 ± 0.000 | 0.325 ± 0.169 | 0.054 ± 0.049 | 0.878 ± 0.031 | 0.824 ± 0.049 | 0.712 ± 0.066 |
| `16` | 1.000 ± 0.000 | 0.350 ± 0.211 | 0.011 ± 0.027 | 0.924 ± 0.035 | 0.913 ± 0.050 | 0.829 ± 0.056 |
| `24` | 1.000 ± 0.000 | 0.400 ± 0.211 | -0.013 ± 0.021 | 0.939 ± 0.043 | 0.952 ± 0.044 | 0.946 ± 0.031 |

### Paired Tests vs Baseline

| Value | Metric | Mean delta | Paired t p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| `4` | `false_retrieval_rate` | -0.050 | 0.6193 | [-0.225, 0.125] |
| `4` | `separation_margin_mean` | 0.065 | 0.0168 | [0.024, 0.107] |
| `4` | `mean_retention` | -0.050 | 0.01856 | [-0.079, -0.015] |
| `4` | `best_impostor_overlap_mean` | -0.115 | 0.001665 | [-0.164, -0.067] |
| `8` | `false_retrieval_rate` | -0.050 | 0.6193 | [-0.225, 0.125] |
| `8` | `separation_margin_mean` | 0.065 | 0.0168 | [0.024, 0.107] |
| `8` | `mean_retention` | -0.050 | 0.01856 | [-0.079, -0.015] |
| `8` | `best_impostor_overlap_mean` | -0.115 | 0.001665 | [-0.164, -0.067] |
| `16` | `false_retrieval_rate` | 0.025 | 0.8321 | [-0.200, 0.225] |
| `16` | `separation_margin_mean` | -0.043 | 0.04136 | [-0.077, -0.010] |
| `16` | `mean_retention` | 0.046 | 0.0001065 | [0.032, 0.058] |
| `16` | `best_impostor_overlap_mean` | 0.089 | 0.001339 | [0.053, 0.125] |
| `24` | `false_retrieval_rate` | 0.075 | 0.2789 | [-0.025, 0.200] |
| `24` | `separation_margin_mean` | -0.068 | 0.0009974 | [-0.096, -0.043] |
| `24` | `mean_retention` | 0.060 | 0.0001629 | [0.042, 0.078] |
| `24` | `best_impostor_overlap_mean` | 0.128 | 8.392e-05 | [0.094, 0.164] |
