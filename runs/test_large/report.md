# Results: test_large

## Retrieval

| metric | value |
|---|---|
| instances | 1000 |
| gold_recall@k | 79.1 |
| conflict_recall@k (conflict scenarios) | 77.2 |
| gold+conflict both retrieved (conflict scenarios) | 74.8 |

## Overall (evidence-complete instances)

| system | accuracy (answerable) | misled | handling (micro) | handling (macro) | conflict P | conflict R | conflict F1 |
|---|---|---|---|---|---|---|---|
| vanilla | 61.9 | 15.3 | 46.8 | 45.6 | 50.0 | 0.2 | 0.4 |
| prompted | 75.3 | 6.3 | 42.6 | 42.3 | 81.0 | 61.5 | 69.9 |
| conflict_aware | 76.9 | 2.0 | 67.0 | 67.9 | 90.0 | 65.6 | 75.9 |
| arbitration_only | 76.3 | 1.8 | 66.4 | 67.3 | 90.0 | 65.6 | 75.9 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 0.0 (n=188) | 46.0 (n=189) | 63.0 (n=184) | 73.5 (n=196) |
| prompted | 51.6 (n=188) | 30.7 (n=189) | 37.0 (n=184) | 50.0 (n=196) |
| conflict_aware | 54.8 (n=188) | 75.7 (n=189) | 78.3 (n=184) | 62.8 (n=196) |
| arbitration_only | 54.3 (n=188) | 75.7 (n=189) | 77.7 (n=184) | 61.7 (n=196) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 72.9 (n=188) | 46.6 (n=189) | 64.1 (n=184) | 73.5 (n=196) |
| prompted | 84.0 (n=188) | 72.0 (n=189) | 82.6 (n=184) | 73.5 (n=196) |
| conflict_aware | 70.2 (n=188) | 76.2 (n=189) | 78.3 (n=184) | 79.1 (n=196) |
| arbitration_only | 69.7 (n=188) | 76.2 (n=189) | 77.7 (n=184) | 78.1 (n=196) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 11.2 (n=188) | 32.3 (n=189) | 16.8 (n=184) | 0.0 (n=196) |
| prompted | 4.3 (n=188) | 14.3 (n=189) | 4.9 (n=184) | 1.0 (n=196) |
| conflict_aware | 8.5 (n=188) | 3.7 (n=189) | 2.7 (n=184) | 0.0 (n=196) |
| arbitration_only | 8.5 (n=188) | 3.2 (n=189) | 2.7 (n=184) | 0.0 (n=196) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 0.0 (n=188) | 0.5 (n=189) | 0.0 (n=184) | 0.5 (n=196) |
| prompted | 63.3 (n=188) | 57.7 (n=189) | 63.6 (n=184) | 34.7 (n=196) |
| conflict_aware | 64.9 (n=188) | 63.0 (n=189) | 69.0 (n=184) | 17.9 (n=196) |
| arbitration_only | 64.9 (n=188) | 63.0 (n=189) | 69.0 (n=184) | 17.9 (n=196) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 4.3 (n=188) | 2.6 (n=189) | 4.9 (n=184) | 1.5 (n=196) |
| prompted | 53.7 (n=188) | 50.3 (n=189) | 54.9 (n=184) | 2.0 (n=196) |
| conflict_aware | 54.8 (n=188) | 53.4 (n=189) | 51.6 (n=184) | 2.0 (n=196) |
| arbitration_only | 54.3 (n=188) | 52.4 (n=189) | 51.1 (n=184) | 1.5 (n=196) |
