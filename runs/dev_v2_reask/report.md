# Results: dev_v2_reask

## Retrieval

| metric | value |
|---|---|
| instances | 300 |
| gold_recall@k | 79.7 |
| conflict_recall@k (conflict scenarios) | 80.9 |
| gold+conflict both retrieved (conflict scenarios) | 77.3 |

## Overall (evidence-complete instances)

| system | accuracy (answerable) | misled | handling (micro) | handling (macro) | conflict P | conflict R | conflict F1 |
|---|---|---|---|---|---|---|---|
| arbitration_only | 75.3 | 1.7 | 70.7 | 72.1 | 91.7 | 75.9 | 83.0 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 73.8 (n=61) | 85.5 (n=55) | 70.7 (n=58) | 58.3 (n=60) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 73.8 (n=61) | 90.9 (n=55) | 70.7 (n=58) | 70.0 (n=60) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 11.5 (n=61) | 1.8 (n=55) | 3.4 (n=58) | 0.0 (n=60) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 83.6 (n=61) | 74.5 (n=55) | 69.0 (n=58) | 18.3 (n=60) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 73.8 (n=61) | 56.4 (n=55) | 50.0 (n=58) | 0.0 (n=60) |
