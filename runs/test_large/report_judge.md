# LLM-judge results: test_large (judge: hf:Qwen/Qwen2.5-7B-Instruct)

## Overall (evidence-complete instances)

| system | accuracy (answerable) | misled | handling (micro) | handling (macro) | conflict P | conflict R | conflict F1 |
|---|---|---|---|---|---|---|---|
| vanilla | 55.6 | 14.8 | 41.8 | 41.0 | 81.5 | 7.8 | 14.3 |
| prompted | 85.1 | 3.0 | 42.1 | 42.1 | 80.4 | 65.8 | 72.4 |
| conflict_aware | 78.6 | 1.8 | 59.2 | 59.8 | 86.2 | 71.3 | 78.0 |
| arbitration_only | 75.8 | 1.7 | 57.6 | 58.5 | 87.0 | 70.2 | 77.7 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 3.7 (n=188) | 36.5 (n=189) | 56.0 (n=184) | 67.9 (n=196) |
| prompted | 61.2 (n=188) | 27.5 (n=189) | 33.7 (n=184) | 45.9 (n=196) |
| conflict_aware | 41.0 (n=188) | 73.5 (n=189) | 70.1 (n=184) | 54.6 (n=196) |
| arbitration_only | 39.4 (n=188) | 72.5 (n=189) | 68.5 (n=184) | 53.6 (n=196) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 67.6 (n=188) | 38.6 (n=189) | 58.7 (n=184) | 68.4 (n=196) |
| prompted | 91.0 (n=188) | 82.0 (n=189) | 93.5 (n=184) | 81.6 (n=196) |
| conflict_aware | 76.1 (n=188) | 82.0 (n=189) | 80.4 (n=184) | 76.0 (n=196) |
| arbitration_only | 70.7 (n=188) | 78.3 (n=189) | 78.3 (n=184) | 73.5 (n=196) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 11.2 (n=188) | 30.2 (n=189) | 17.4 (n=184) | 0.0 (n=196) |
| prompted | 1.6 (n=188) | 7.4 (n=189) | 2.2 (n=184) | 0.0 (n=196) |
| conflict_aware | 3.2 (n=188) | 3.2 (n=189) | 2.7 (n=184) | 0.0 (n=196) |
| arbitration_only | 2.7 (n=188) | 2.6 (n=189) | 2.7 (n=184) | 0.0 (n=196) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 9.0 (n=188) | 7.9 (n=189) | 6.5 (n=184) | 4.6 (n=196) |
| prompted | 67.0 (n=188) | 61.9 (n=189) | 68.5 (n=184) | 39.3 (n=196) |
| conflict_aware | 70.7 (n=188) | 68.8 (n=189) | 74.5 (n=184) | 27.6 (n=196) |
| arbitration_only | 69.7 (n=188) | 67.7 (n=189) | 73.4 (n=184) | 25.0 (n=196) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 3.7 (n=188) | 2.1 (n=189) | 2.7 (n=184) | 0.5 (n=196) |
| prompted | 61.2 (n=188) | 54.5 (n=189) | 59.8 (n=184) | 34.7 (n=196) |
| conflict_aware | 41.0 (n=188) | 8.5 (n=189) | 10.3 (n=184) | 15.8 (n=196) |
| arbitration_only | 39.4 (n=188) | 5.8 (n=189) | 9.8 (n=184) | 16.8 (n=196) |

## Agreement with string-matching metrics (evidence-complete instances)

| signal | agreement |
|---|---|
| correct | 85.9 |
| misled | 97.8 |
| flagged | 93.9 |
| handled_ok | 85.5 |
