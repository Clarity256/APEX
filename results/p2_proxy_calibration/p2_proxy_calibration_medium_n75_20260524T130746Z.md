# P2 Proxy Calibration

| Metric | Value |
|---|---:|
| `cell_samples` | 900 |
| `oracle_solves` | 75 |
| `proxy_xi_mean` | 0.999661 |
| `oracle_xi_mean` | 1 |
| `signed_error_mean` | -0.000338774 |
| `abs_error_mean` | 0.000338794 |
| `abs_error_median` | 1.16125e-10 |
| `abs_error_p95` | 4.21018e-08 |
| `abs_error_max` | 0.101635 |
| `rel_error_median` | 1.16125e-10 |
| `rel_error_p95` | 4.21018e-08 |
| `rel_error_max` | 0.101635 |
| `overestimate_rate` | 0.976667 |
| `underestimate_rate` | 0.00333333 |
| `pearson_corr` | 0.0119387 |
| `demand_bits_mean` | 2.07641e+06 |
| `oracle_time_total_s` | 3.77545 |

## Oracle Status Counts

| Status | Solves |
|---|---:|
| `optimal_inaccurate` | 29 |
| `optimal` | 46 |

## Demand Quantiles

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q1` | 225 | 1 | 1 | 7.40387e-09 | 2.0606e-10 | 3.15026e-08 | 0.964444 |
| `q2` | 225 | 1 | 1 | 3.09771e-09 | 6.87382e-11 | 2.07995e-08 | 1 |
| `q3` | 225 | 1 | 0.998645 | -0.00135513 | 1.20875e-10 | 3.81566e-08 | 0.946667 |
| `q4` | 225 | 1 | 1 | 2.26279e-08 | 1.89694e-10 | 1.34267e-07 | 0.995556 |

## Oracle Xi Bins

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0.75,1]` | 900 | 1 | 0.999661 | -0.000338774 | 1.16125e-10 | 4.21018e-08 | 0.976667 |

## Visible Satellite Groups

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1` | 18 | 1 | 0.983061 | -0.0169392 | 4.23121e-09 | 0.101635 | 0.777778 |
| `2` | 177 | 1 | 1 | 9.00584e-09 | 1.09626e-10 | 3.9985e-08 | 0.977401 |
| `3` | 324 | 1 | 1 | 1.06286e-08 | 1.12497e-10 | 3.89624e-08 | 0.984568 |
| `4` | 381 | 1 | 1 | 9.69553e-09 | 1.19927e-10 | 3.941e-08 | 0.979003 |
