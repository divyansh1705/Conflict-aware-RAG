# Results: dev_v3_k8

## Retrieval

| metric | value |
|---|---|
| instances | 300 |
| gold_recall@k | 84.7 |
| conflict_recall@k (conflict scenarios) | 84.9 |
| gold+conflict both retrieved (conflict scenarios) | 83.6 |

## Overall (evidence-complete instances)

| system | accuracy (answerable) | misled | handling (micro) | handling (macro) | conflict P | conflict R | conflict F1 |
|---|---|---|---|---|---|---|---|
| arbitration_only | 75.7 | 1.6 | 69.3 | 69.8 | 92.6 | 73.4 | 81.9 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 69.2 (n=65) | 83.9 (n=62) | 67.2 (n=61) | 58.7 (n=63) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 72.3 (n=65) | 88.7 (n=62) | 70.5 (n=61) | 69.8 (n=63) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 10.8 (n=65) | 1.6 (n=62) | 3.3 (n=61) | 0.0 (n=63) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 80.0 (n=65) | 72.6 (n=62) | 67.2 (n=61) | 17.5 (n=63) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| arbitration_only | 69.2 (n=65) | 51.6 (n=62) | 49.2 (n=61) | 0.0 (n=63) |
