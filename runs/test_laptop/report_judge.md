# LLM-judge results: test_laptop (judge: hf:Qwen/Qwen2.5-7B-Instruct)

## Overall (evidence-complete instances)

| system | accuracy (answerable) | misled | handling (micro) | handling (macro) | conflict P | conflict R | conflict F1 |
|---|---|---|---|---|---|---|---|
| vanilla | 54.7 | 14.9 | 39.2 | 38.5 | 77.8 | 20.0 | 31.8 |
| prompted | 86.7 | 4.6 | 35.1 | 35.4 | 75.1 | 78.6 | 76.8 |
| conflict_aware | 74.8 | 2.2 | 53.5 | 54.8 | 83.1 | 75.4 | 79.1 |
| arbitration_only | 71.3 | 2.2 | 51.8 | 52.8 | 84.5 | 67.2 | 74.9 |

## Conflict-handling success by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 6.9 (n=188) | 39.7 (n=189) | 47.3 (n=184) | 60.2 (n=196) |
| prompted | 71.3 (n=188) | 17.5 (n=189) | 20.7 (n=184) | 32.1 (n=196) |
| conflict_aware | 43.1 (n=188) | 66.7 (n=189) | 62.0 (n=184) | 47.4 (n=196) |
| arbitration_only | 38.3 (n=188) | 64.0 (n=189) | 59.8 (n=184) | 49.0 (n=196) |

## Accuracy (gold in ANSWER) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 56.9 (n=188) | 43.4 (n=189) | 54.3 (n=184) | 65.3 (n=196) |
| prompted | 89.4 (n=188) | 85.2 (n=189) | 87.0 (n=184) | 88.3 (n=196) |
| conflict_aware | 72.3 (n=188) | 81.0 (n=189) | 76.6 (n=184) | 70.4 (n=196) |
| arbitration_only | 66.0 (n=188) | 75.7 (n=189) | 71.2 (n=184) | 70.9 (n=196) |

## Misled (wrong substituted answer given) by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 11.2 (n=188) | 32.8 (n=189) | 15.2 (n=184) | 0.0 (n=196) |
| prompted | 4.3 (n=188) | 10.1 (n=189) | 4.3 (n=184) | 0.5 (n=196) |
| conflict_aware | 6.9 (n=188) | 2.1 (n=189) | 4.9 (n=184) | 0.0 (n=196) |
| arbitration_only | 5.9 (n=188) | 2.1 (n=189) | 4.9 (n=184) | 0.0 (n=196) |

## Conflict flagged by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 25.5 (n=188) | 15.9 (n=189) | 18.5 (n=184) | 14.3 (n=196) |
| prompted | 78.2 (n=188) | 79.9 (n=189) | 77.7 (n=184) | 63.3 (n=196) |
| conflict_aware | 69.7 (n=188) | 79.9 (n=189) | 76.6 (n=184) | 35.2 (n=196) |
| arbitration_only | 62.8 (n=188) | 70.4 (n=189) | 68.5 (n=184) | 28.1 (n=196) |

## Both answers mentioned by scenario

| system | balanced_conflict | misinfo_majority | misinfo_single | no_conflict |
|---|---|---|---|---|
| vanilla | 6.9 (n=188) | 3.7 (n=189) | 7.1 (n=184) | 0.5 (n=196) |
| prompted | 71.3 (n=188) | 67.7 (n=189) | 66.3 (n=184) | 50.0 (n=196) |
| conflict_aware | 43.1 (n=188) | 14.3 (n=189) | 14.7 (n=184) | 15.3 (n=196) |
| arbitration_only | 38.3 (n=188) | 11.6 (n=189) | 11.4 (n=184) | 15.3 (n=196) |

## Agreement with string-matching metrics (evidence-complete instances)

| signal | agreement |
|---|---|
| correct | 80.6 |
| misled | 97.3 |
| flagged | 89.0 |
| handled_ok | 84.4 |
