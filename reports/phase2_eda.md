# Phase 2 - EDA & Leakage Audit (train)

- Rows: **590,540**, Columns: **434**
- Fraud rate (class imbalance): **3.50%** (20,663 of 590,540)  ->  ~28:1 negative:positive
- TransactionDT span: **182.0 days**, monotonic increasing: **True**
  -> use TransactionDT ordering for time-respecting CV, NOT random splits.

## Transaction amount by class
|   isFraud |   mean |   median |     max |
|----------:|-------:|---------:|--------:|
|         0 | 134.51 |     68.5 | 31937.4 |
|         1 | 149.24 |     75   |  5191   |

## Missingness by column group
| group                |   n_cols |   mean_missing_% |   min_missing_% |   max_missing_% |
|:---------------------|---------:|-----------------:|----------------:|----------------:|
| C (counting)         |       14 |              0   |             0   |             0   |
| D (timedelta)        |       17 |             60.5 |             0.2 |            93.4 |
| M (match flags)      |        9 |             49.9 |            28.7 |            59.3 |
| V (Vesta engineered) |      339 |             43   |             0   |            86.1 |
| id (identity)        |       38 |             84.8 |            75.6 |            99.2 |
| card                 |        6 |              0.5 |             0   |             1.5 |
| addr                 |        2 |             11.1 |            11.1 |            11.1 |
| dist                 |        2 |             76.6 |            59.7 |            93.6 |

## Cardinality of entity-key / categorical candidates
| column        |   nunique |   missing_% |
|:--------------|----------:|------------:|
| card1         |     13553 |         0   |
| card2         |       500 |         1.5 |
| card3         |       114 |         0.3 |
| card4         |         4 |         0.3 |
| card5         |       119 |         0.7 |
| card6         |         4 |         0.3 |
| addr1         |       332 |        11.1 |
| addr2         |        74 |        11.1 |
| P_emaildomain |        59 |        16   |
| R_emaildomain |        60 |        76.8 |
| ProductCD     |         5 |         0   |
| DeviceType    |         2 |        76.2 |
| DeviceInfo    |      1786 |        79.9 |

## Leakage smell test (baseline fraud = 3.499%)
Columns where *presence vs absence of a value* most separates fraud. Large gaps are candidates to inspect before trusting them as features.

| column   |   fraud_if_present_% |   fraud_if_absent_% |   gap_pp |
|:---------|---------------------:|--------------------:|---------:|
| V309     |                  3.5 |               16.67 |    13.17 |
| V299     |                  3.5 |               16.67 |    13.17 |
| V318     |                  3.5 |               16.67 |    13.17 |
| V317     |                  3.5 |               16.67 |    13.17 |
| V316     |                  3.5 |               16.67 |    13.17 |
| V312     |                  3.5 |               16.67 |    13.17 |
| V311     |                  3.5 |               16.67 |    13.17 |
| V310     |                  3.5 |               16.67 |    13.17 |
| V308     |                  3.5 |               16.67 |    13.17 |
| V307     |                  3.5 |               16.67 |    13.17 |
| V306     |                  3.5 |               16.67 |    13.17 |
| V305     |                  3.5 |               16.67 |    13.17 |
| V304     |                  3.5 |               16.67 |    13.17 |
| V303     |                  3.5 |               16.67 |    13.17 |
| V302     |                  3.5 |               16.67 |    13.17 |
