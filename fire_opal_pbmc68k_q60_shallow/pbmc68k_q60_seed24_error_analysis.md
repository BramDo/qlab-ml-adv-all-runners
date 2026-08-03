# PBMC68k q60 frozen error analysis

## Reproduction contract

- Confirmation seed: 24
- Recorded split hashes reproduced: yes
- Recorded selected observables reproduced: yes
- Provider calls: 0

## Paired held-out result

| Metric | Value |
|---|---:|
| Frozen q60 balanced accuracy | 0.6042 |
| Classical balanced accuracy | 0.7083 |
| q60 minus classical | -0.1042 |
| Paired bootstrap 95% interval | [-0.2708, +0.0625] |
| Exact McNemar p-value | 0.3323 |

## Paired outcomes

- Both correct: 23
- q60 only correct: 6
- Classical only correct: 11
- Both wrong: 8

## Representation audit

- Selected measurement bases: {'Y': 1, 'Z': 23}
- Multiqubit observables selected: 0
- Features numerically sensitive to pair scale: 1 / 24
- `pair_scale=0` post-hoc balanced accuracy: 0.6250
- Ablation delta from frozen q60: +0.0208

## Decision

The error analysis is diagnostic only. A future model should make interaction-sensitive X/Y or multiqubit observables part of the pre-registered representation before another fresh split is opened. No hardware step is justified by this artifact.

> This is post-hoc error analysis on an already evaluated local test split. It can diagnose representation failure and guide a future pre-registered model, but it is not new held-out performance evidence, hardware evidence, biological feature attribution, or evidence of quantum advantage.
