# Results: dev_v3_pair

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
| arbitration_only | 71.9 | 3.4 | 63.6 | 65.0 | 94.8 | 42.0 | 58.2 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 44.3 (n=61) | 85.5 (n=55) | 63.8 (n=58) | 66.7 (n=60) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 65.6 (n=61) | 87.3 (n=55) | 63.8 (n=58) | 70.0 (n=60) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 18.0 (n=61) | 1.8 (n=55) | 8.6 (n=58) | 0.0 (n=60) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 50.8 (n=61) | 34.5 (n=55) | 39.7 (n=58) | 6.7 (n=60) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 44.3 (n=61) | 29.1 (n=55) | 29.3 (n=58) | 0.0 (n=60) |
