# Results: dev_v2_base

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
| arbitration_only | 71.9 | 1.7 | 69.0 | 70.2 | 92.7 | 73.0 | 81.7 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 73.8 (n=61) | 80.0 (n=55) | 65.5 (n=58) | 61.7 (n=60) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 72.1 (n=61) | 85.5 (n=55) | 65.5 (n=58) | 70.0 (n=60) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 11.5 (n=61) | 1.8 (n=55) | 3.4 (n=58) | 0.0 (n=60) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 85.2 (n=61) | 69.1 (n=55) | 63.8 (n=58) | 15.0 (n=60) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 73.8 (n=61) | 56.4 (n=55) | 50.0 (n=58) | 0.0 (n=60) |
