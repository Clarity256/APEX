# P2 Proxy Calibration

| Metric | Value |
|---|---:|
| `cell_samples` | 900 |
| `oracle_solves` | 75 |
| `proxy_xi_mean` | 0.460855 |
| `oracle_xi_mean` | 0.603049 |
| `signed_error_mean` | -0.142194 |
| `abs_error_mean` | 0.299934 |
| `abs_error_median` | 0.280365 |
| `abs_error_p95` | 0.640425 |
| `abs_error_max` | 0.864578 |
| `rel_error_median` | 0.496241 |
| `rel_error_p95` | 3.21398 |
| `rel_error_max` | 8.41275 |
| `overestimate_rate` | 0.324444 |
| `underestimate_rate` | 0.675556 |
| `pearson_corr` | 0.194428 |
| `demand_bits_mean` | 2.07641e+07 |
| `oracle_time_total_s` | 6.87621 |

## Oracle Status Counts

| Status | Solves |
|---|---:|
| `optimal_inaccurate` | 42 |
| `optimal` | 33 |

## Demand Quantiles

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q1` | 225 | 0.842849 | 0.605339 | -0.237509 | 0.259125 | 0.627532 | 0.177778 |
| `q2` | 225 | 0.739261 | 0.423041 | -0.31622 | 0.329344 | 0.656437 | 0.106667 |
| `q3` | 225 | 0.638505 | 0.343027 | -0.295478 | 0.284293 | 0.648581 | 0.0577778 |
| `q4` | 225 | 0.19158 | 0.472011 | 0.280431 | 0.261029 | 0.632847 | 0.955556 |

## Oracle Xi Bins

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0,0.25)` | 180 | 0.141679 | 0.458104 | 0.316425 | 0.282749 | 0.658028 | 0.994444 |
| `[0.25,0.5)` | 155 | 0.411246 | 0.397645 | -0.0136006 | 0.124164 | 0.357373 | 0.419355 |
| `[0.5,0.75)` | 235 | 0.624903 | 0.412253 | -0.21265 | 0.23006 | 0.445287 | 0.140426 |
| `[0.75,1]` | 330 | 0.929231 | 0.526655 | -0.402576 | 0.444885 | 0.690606 | 0.0454545 |

## Visible Satellite Groups

| Group | Count | Oracle xi mean | Proxy xi mean | Mean signed error | Median abs error | P95 abs error | Overestimate rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1` | 18 | 0.560884 | 0.12764 | -0.433243 | 0.44245 | 0.836624 | 0.111111 |
| `2` | 177 | 0.612211 | 0.309359 | -0.302851 | 0.319685 | 0.672941 | 0.124294 |
| `3` | 324 | 0.608067 | 0.437158 | -0.17091 | 0.275771 | 0.651411 | 0.333333 |
| `4` | 381 | 0.596516 | 0.567129 | -0.0293876 | 0.255665 | 0.599564 | 0.419948 |
