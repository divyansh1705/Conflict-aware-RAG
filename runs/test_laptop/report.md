# Results: test_laptop

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
| vanilla | 54.9 | 14.9 | 41.5 | 40.6 | 100.0 | 1.4 | 2.8 |
| prompted | 51.9 | 8.6 | 33.5 | 33.5 | 76.2 | 70.8 | 73.4 |
| conflict_aware | 68.2 | 2.3 | 55.9 | 56.9 | 86.8 | 63.3 | 73.2 |
| arbitration_only | 68.0 | 2.3 | 56.0 | 57.0 | 86.8 | 63.3 | 73.2 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 0.0 (n=188) | 44.4 (n=189) | 53.8 (n=184) | 64.3 (n=196) |
| prompted | 51.1 (n=188) | 19.0 (n=189) | 27.2 (n=184) | 36.7 (n=196) |
| conflict_aware | 40.4 (n=188) | 69.3 (n=189) | 64.1 (n=184) | 53.6 (n=196) |
| arbitration_only | 41.5 (n=188) | 69.3 (n=189) | 64.1 (n=184) | 53.1 (n=196) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 60.1 (n=188) | 44.4 (n=189) | 55.4 (n=184) | 64.3 (n=196) |
| prompted | 55.3 (n=188) | 50.3 (n=189) | 57.1 (n=184) | 50.5 (n=196) |
| conflict_aware | 59.6 (n=188) | 72.0 (n=189) | 64.1 (n=184) | 72.4 (n=196) |
| arbitration_only | 59.0 (n=188) | 72.0 (n=189) | 64.1 (n=184) | 71.9 (n=196) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 13.3 (n=188) | 34.4 (n=189) | 13.6 (n=184) | 0.0 (n=196) |
| prompted | 6.9 (n=188) | 18.0 (n=189) | 9.8 (n=184) | 0.0 (n=196) |
| conflict_aware | 12.2 (n=188) | 3.2 (n=189) | 4.3 (n=184) | 0.0 (n=196) |
| arbitration_only | 12.8 (n=188) | 3.2 (n=189) | 4.3 (n=184) | 0.0 (n=196) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 0.0 (n=188) | 3.2 (n=189) | 1.1 (n=184) | 0.0 (n=196) |
| prompted | 72.3 (n=188) | 69.8 (n=189) | 70.1 (n=184) | 54.1 (n=196) |
| conflict_aware | 58.5 (n=188) | 66.1 (n=189) | 65.2 (n=184) | 22.4 (n=196) |
| arbitration_only | 58.5 (n=188) | 66.1 (n=189) | 65.2 (n=184) | 22.4 (n=196) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 16.0 (n=188) | 9.5 (n=189) | 11.4 (n=184) | 0.0 (n=196) |
| prompted | 53.7 (n=188) | 49.2 (n=189) | 51.6 (n=184) | 1.0 (n=196) |
| conflict_aware | 42.0 (n=188) | 46.0 (n=189) | 40.8 (n=184) | 1.5 (n=196) |
| arbitration_only | 41.5 (n=188) | 44.4 (n=189) | 39.7 (n=184) | 1.5 (n=196) |
