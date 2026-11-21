# Results: dev_v3_soft

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
| arbitration_only | 73.0 | 2.2 | 68.2 | 69.5 | 91.4 | 73.6 | 81.5 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 68.9 (n=61) | 83.6 (n=55) | 69.0 (n=58) | 56.7 (n=60) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 70.5 (n=61) | 87.3 (n=55) | 69.0 (n=58) | 68.3 (n=60) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 8.2 (n=61) | 1.8 (n=55) | 5.2 (n=58) | 0.0 (n=60) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 80.3 (n=61) | 74.5 (n=55) | 65.5 (n=58) | 18.3 (n=60) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 68.9 (n=61) | 49.1 (n=55) | 41.4 (n=58) | 0.0 (n=60) |
