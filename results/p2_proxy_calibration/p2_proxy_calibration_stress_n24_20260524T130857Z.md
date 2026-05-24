# P2 Proxy Calibration

| Metric | Value |
|---|---:|
| `cell_samples` | 720 |
| `oracle_solves` | 24 |
| `proxy_xi_mean` | 0.317665 |
| `oracle_xi_mean` | 0.464084 |
| `signed_error_mean` | -0.146419 |
| `abs_error_mean` | 0.255905 |
| `abs_error_median` | 0.210762 |
| `abs_error_p95` | 0.64647 |
| `abs_error_max` | 0.885311 |
| `rel_error_median` | 0.531981 |
| `rel_error_p95` | 2.63722 |
| `rel_error_max` | 6.00212 |
| `overestimate_rate` | 0.326389 |
| `underestimate_rate` | 0.673611 |
| `pearson_corr` | 0.276491 |
| `demand_bits_mean` | 2.34323e+07 |
| `oracle_time_total_s` | 10.7978 |

## Oracle Status Counts

| Status | Solves |
|---|---:|
| `optimal_inaccurate` | 23 |
| `optimal` | 1 |

## Demand Quantiles

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q1` | 180 | 0.696858 | 0.403234 | -0.293624 | 0.274548 | 0.64686 | 0.0888889 |
| `q2` | 180 | 0.583993 | 0.299238 | -0.284755 | 0.246719 | 0.718052 | 0.0666667 |
| `q3` | 180 | 0.445358 | 0.278387 | -0.166971 | 0.218965 | 0.636604 | 0.188889 |
| `q4` | 180 | 0.130126 | 0.289799 | 0.159673 | 0.168602 | 0.331811 | 0.961111 |

## Oracle Xi Bins

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0,0.25)` | 200 | 0.122955 | 0.301712 | 0.178756 | 0.172979 | 0.370302 | 0.965 |
| `[0.25,0.5)` | 233 | 0.396164 | 0.273848 | -0.122316 | 0.140361 | 0.298925 | 0.154506 |
| `[0.5,0.75)` | 161 | 0.62332 | 0.345246 | -0.278074 | 0.287234 | 0.478737 | 0.0372671 |
| `[0.75,1]` | 126 | 0.927687 | 0.38877 | -0.538917 | 0.539384 | 0.805822 | 0 |

## Visible Satellite Groups

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2` | 2 | 0.764898 | 0.159806 | -0.605092 | 0.605092 | 0.788405 | 0 |
| `3` | 62 | 0.466207 | 0.20069 | -0.265518 | 0.243812 | 0.765289 | 0.241935 |
| `4` | 164 | 0.439951 | 0.232241 | -0.20771 | 0.203711 | 0.7045 | 0.256098 |
| `5` | 240 | 0.464657 | 0.305258 | -0.159399 | 0.210653 | 0.630749 | 0.308333 |
| `6` | 152 | 0.469509 | 0.371249 | -0.0982603 | 0.19952 | 0.568757 | 0.355263 |
| `7` | 66 | 0.549588 | 0.501187 | -0.0484014 | 0.228655 | 0.520853 | 0.454545 |
| `8` | 34 | 0.364644 | 0.444075 | 0.0794306 | 0.260996 | 0.435141 | 0.588235 |
