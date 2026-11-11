## MobileNetV2, alpha=1, seeds 42/43/44: round 200 (mean ± sample std [min, max]; per seed)

### p = 10%

| policy | round 200 | per seed | last 10 rounds | per seed |
|---|---|---|---|---|
| Round Robin | 65.3 ± 1.4 [64.1, 66.8] | 42: 66.8, 43: 64.1, 44: 65.1 | 64.5 ± 1.0 [63.5, 65.4] | 42: 65.4, 43: 63.5, 44: 64.8 |
| Sliding Window | 60.9 ± 1.6 [59.5, 62.5] | 42: 60.6, 43: 59.5, 44: 62.5 | 62.3 ± 1.5 [60.7, 63.6] | 42: 62.5, 43: 60.7, 44: 63.6 |
| Top-$k$ (FedPHE) | 27.6 ± 5.5 [24.1, 33.9] | 42: 24.1, 43: 33.9, 44: 24.8 | 27.7 ± 5.5 [24.2, 34.1] | 42: 24.2, 43: 34.1, 44: 24.8 |
| Random | 60.0 [60.0, 60.0] | 42: 60.0 | 60.2 [60.2, 60.2] | 42: 60.2 |

- RR - SW at round 200: +4.5 points in the mean, ranges disjoint (RR [64.1, 66.8], SW [59.5, 62.5])
- RR - SW at mean of last 10 rounds: +2.3 points in the mean, ranges overlap (RR [63.5, 65.4], SW [60.7, 63.6])

### p = 70%

| policy | round 200 | per seed | last 10 rounds | per seed |
|---|---|---|---|---|
| Round Robin | 76.7 ± 0.6 [76.1, 77.1] | 42: 76.9, 43: 76.1, 44: 77.1 | 75.2 ± 0.4 [74.8, 75.6] | 42: 75.2, 43: 74.8, 44: 75.6 |
| Sliding Window | 75.9 ± 1.0 [74.8, 76.6] | 42: 76.6, 43: 74.8, 44: 76.3 | 75.7 ± 0.9 [74.6, 76.4] | 42: 76.4, 43: 74.6, 44: 76.1 |
| Top-$k$ (FedPHE) | 75.4 ± 0.7 [74.7, 76.1] | 42: 74.7, 43: 75.5, 44: 76.1 | 75.3 ± 0.5 [75.0, 75.8] | 42: 75.0, 43: 75.0, 44: 75.8 |
| Random | 75.5 [75.5, 75.5] | 42: 75.5 | 75.7 [75.7, 75.7] | 42: 75.7 |

- RR - SW at round 200: +0.8 points in the mean, ranges overlap (RR [76.1, 77.1], SW [74.8, 76.6])
- RR - SW at mean of last 10 rounds: -0.5 points in the mean, ranges overlap (RR [74.8, 75.6], SW [74.6, 76.4])

Plaintext FedAvg, seed 42, round 200: 76.4
