# Results: dev_v3_laptop

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
| arbitration_only | 65.7 | 1.7 | 60.7 | 62.3 | 90.6 | 72.4 | 80.5 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 59.0 (n=61) | 74.5 (n=55) | 67.2 (n=58) | 48.3 (n=60) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 63.9 (n=61) | 74.5 (n=55) | 67.2 (n=58) | 60.0 (n=60) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 11.5 (n=61) | 0.0 (n=55) | 5.2 (n=58) | 0.0 (n=60) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 75.4 (n=61) | 69.1 (n=55) | 72.4 (n=58) | 20.0 (n=60) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 59.0 (n=61) | 47.3 (n=55) | 48.3 (n=58) | 0.0 (n=60) |
