# P2 Proxy Calibration

| Metric | Value |
|---|---:|
| `cell_samples` | 900 |
| `oracle_solves` | 75 |
| `proxy_xi_mean` | 0.046342 |
| `oracle_xi_mean` | 0.0694542 |
| `signed_error_mean` | -0.0231122 |
| `abs_error_mean` | 0.0396275 |
| `abs_error_median` | 0.0305461 |
| `abs_error_p95` | 0.103211 |
| `abs_error_max` | 0.365598 |
| `rel_error_median` | 0.529453 |
| `rel_error_p95` | 3.23085 |
| `rel_error_max` | 8.48933 |
| `overestimate_rate` | 0.31 |
| `underestimate_rate` | 0.69 |
| `pearson_corr` | 0.269043 |
| `demand_bits_mean` | 2.07641e+08 |
| `oracle_time_total_s` | 6.08092 |

## Oracle Status Counts

| Status | Solves |
|---|---:|
| `optimal_inaccurate` | 26 |
| `optimal` | 49 |

## Demand Quantiles

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q1` | 225 | 0.109313 | 0.0612543 | -0.0480586 | 0.0326516 | 0.154486 | 0.128889 |
| `q2` | 225 | 0.0824727 | 0.0423041 | -0.0401686 | 0.0329656 | 0.103852 | 0.0933333 |
| `q3` | 225 | 0.0681183 | 0.0345323 | -0.033586 | 0.0286828 | 0.0812572 | 0.0533333 |
| `q4` | 225 | 0.0179128 | 0.0472774 | 0.0293647 | 0.0274775 | 0.0633276 | 0.964444 |

## Oracle Xi Bins

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0,0.25)` | 891 | 0.0667167 | 0.0460306 | -0.0206861 | 0.0302384 | 0.0984361 | 0.313131 |
| `[0.25,0.5)` | 9 | 0.340468 | 0.0771775 | -0.26329 | 0.255265 | 0.361038 | 0 |

## Visible Satellite Groups

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1` | 18 | 0.0583987 | 0.012764 | -0.0456347 | 0.044276 | 0.103987 | 0.111111 |
| `2` | 177 | 0.0677152 | 0.0311578 | -0.0365574 | 0.0331093 | 0.107604 | 0.112994 |
| `3` | 324 | 0.069285 | 0.044082 | -0.0252031 | 0.0296829 | 0.0990116 | 0.324074 |
| `4` | 381 | 0.0709282 | 0.0569044 | -0.0140238 | 0.0282238 | 0.103599 | 0.39895 |
