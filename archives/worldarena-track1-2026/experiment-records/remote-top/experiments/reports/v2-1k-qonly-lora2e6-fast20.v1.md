# q-only LoRA bounded ablation

## Decision

Stop Adapter/LoRA hyperparameter search. Audit action representation, spatial-temporal alignment, and trajectory-focused supervision before further training.

## Configuration

- Adapter frozen exactly; blocks 8-29 self_attn.q LoRA only; rank/alpha 16/16; LR 2e-6; warmup 10; 50 steps; 7 GPUs.
- 60/60 matched videos generated at 81 frames, 480x640, 50 sampling steps; 0 generation failures.

## Calibrated proxy versus q/v step75

| step | trajectory delta | mean DTW | win rate | black rate | detector failure |
|---:|---:|---:|---:|---:|---:|
| 10 | -2.19% | 27.480 | 35% | 0% | 40% |
| 25 | -3.12% | 27.691 | 40% | 0% | 35% |
| 50 | +6.40% | 25.414 | 65% | 0% | 35% |

Step50 was the only proxy-positive checkpoint: +6.40%, DTW 25.41 vs 27.28, win rate 65%, black 0%, detector failure 35% vs 40%.

## Pinned official SAM3 versus formal S1A125

- Evaluator commit: `7b3feee108427bee3380064bb5154970ed7468b5`
- Valid episodes: baseline 6/20; candidate 8/20; paired finite 4/20.
- Paired win rate: 0%.
- Raw mean DTW: 0.1710 -> 0.2379 (worse).
- Action-adherence distance: 0.1838 -> 0.2522 (worse).
- Strict gate: FAIL; reasons: invalid episodes present, official mean inverse gain is below 5%, raw mean distance is not strictly lower, paired win rate is not strictly above 55%, action-adherence distance did not improve by 5%.
- The candidate all20 mean(1/d) is outlier dominated and is not used to override the strict paired gate.

## Next engineering question

Verify that action raster/SE(3) timing, camera projection, arm identity, and weighted flow-matching create a direct, correctly signed trajectory gradient. Do not run more LR/rank/block sweeps until this audit produces a falsifiable fix.
