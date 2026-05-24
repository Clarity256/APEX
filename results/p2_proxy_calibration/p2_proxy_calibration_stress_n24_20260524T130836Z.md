# P2 Proxy Calibration

| Metric | Value |
|---|---:|
| `cell_samples` | 720 |
| `oracle_solves` | 24 |
| `proxy_xi_mean` | 0.0317665 |
| `oracle_xi_mean` | 0.0505935 |
| `signed_error_mean` | -0.0188271 |
| `abs_error_mean` | 0.0299771 |
| `abs_error_median` | 0.0211856 |
| `abs_error_p95` | 0.0866834 |
| `abs_error_max` | 0.345072 |
| `rel_error_median` | 0.545081 |
| `rel_error_p95` | 2.6496 |
| `rel_error_max` | 6.07169 |
| `overestimate_rate` | 0.323611 |
| `underestimate_rate` | 0.676389 |
| `pearson_corr` | 0.266898 |
| `demand_bits_mean` | 2.34323e+08 |
| `oracle_time_total_s` | 11.2397 |

## Oracle Status Counts

| Status | Solves |
|---|---:|
| `optimal_inaccurate` | 18 |
| `optimal` | 6 |

## Demand Quantiles

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q1` | 180 | 0.0814921 | 0.0403234 | -0.0411687 | 0.027815 | 0.12435 | 0.0888889 |
| `q2` | 180 | 0.0625821 | 0.0299238 | -0.0326583 | 0.0248474 | 0.0884116 | 0.0611111 |
| `q3` | 180 | 0.0455672 | 0.0278387 | -0.0177284 | 0.0220354 | 0.0636539 | 0.183333 |
| `q4` | 180 | 0.0127328 | 0.0289799 | 0.0162471 | 0.0168642 | 0.0332969 | 0.961111 |

## Oracle Xi Bins

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0,0.25)` | 715 | 0.0486959 | 0.031652 | -0.0170439 | 0.0211268 | 0.0818398 | 0.325874 |
| `[0.25,0.5)` | 5 | 0.321958 | 0.0481344 | -0.273823 | 0.261043 | 0.328266 | 0 |

## Visible Satellite Groups

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2` | 2 | 0.105229 | 0.0159806 | -0.0892487 | 0.0892487 | 0.133736 | 0 |
| `3` | 62 | 0.0490039 | 0.020069 | -0.028935 | 0.0245901 | 0.0940883 | 0.241935 |
| `4` | 164 | 0.0456566 | 0.0232241 | -0.0224325 | 0.0204022 | 0.0819954 | 0.256098 |
| `5` | 240 | 0.0536284 | 0.0305258 | -0.0231026 | 0.0212592 | 0.107532 | 0.304167 |
| `6` | 152 | 0.0506458 | 0.0371249 | -0.0135209 | 0.0199834 | 0.0688462 | 0.355263 |
| `7` | 66 | 0.0579343 | 0.0501187 | -0.00781564 | 0.0240136 | 0.052297 | 0.439394 |
| `8` | 34 | 0.038186 | 0.0444075 | 0.00622151 | 0.026152 | 0.0437856 | 0.588235 |
