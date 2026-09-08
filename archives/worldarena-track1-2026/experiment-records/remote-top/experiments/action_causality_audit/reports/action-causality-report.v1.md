# Wan-Action-Lite+ Action Causality Audit

## Outcome

- S1A125 classification: `cross_arm_gradient_routing`
- q-only50 classification: `cross_arm_gradient_routing`
- q-only/S1 robot sensitivity ratio: `1.015402`
- Adapter tensors: byte-identical
- Scope: eight paired discovery episodes; this is diagnostic, not a promotion result.
- Both checkpoints preserve distinct action features, non-negligible residuals, and robot-focused output sensitivity.
- Correct action beats reverse, swap, and random-valid robot-region FM loss on all 8/8 paired episodes.
- Blocking defect: same-arm gradients do not dominate cross-arm gradients.
- Decision: `qonly_does_not_repair_cross_arm_routing`.

## Left/right gradient routing (same-arm / cross-arm)

| Arm | S1A125 | q-only50 | Required |
|---|---:|---:|---:|
| left | 1.052396 | 1.051376 | >=1.5 |
| right | 0.953824 | 0.954351 | >=1.5 |

## Robot-loss margins (wrong minus correct)

| Perturbation | S1A125 median | q-only50 median | q-only - S1 |
|---|---:|---:|---:|
| reverse | 0.00050049 | 0.00050885 | 0.00000836 |
| swap | 0.00047716 | 0.00049108 | 0.00001393 |
| random-valid | 0.00025376 | 0.00027065 | 0.00001689 |

## Decision

Do not extend the q-only hyperparameter search: it raises median robot sensitivity only marginally and does not repair arm identity routing.
The next bounded experiment should repair left/right spatial routing in the Adapter (support-gated per-arm raster residuals and removal of dense empty-token leakage), then rerun this same frozen discovery-8 audit before any training-scale expansion.
No new training is authorized by this report alone.
