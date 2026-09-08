# Wan-Action-Lite+ v6 Gripper Trajectory Supervision Design

## Objective

Directly optimize the final predicted clean latent for left/right gripper image-space trajectory while freezing Wan, the clean support-gated parent Adapter, and the trajectory probe.

## Non-negotiable boundaries

- Formal artifacts live under `/data/di/worldarena2_track1_20260815`.
- Source lives at `/home/huazhi/nlh/baseline`; Wan source lives at `/home/huazhi/nlh/Wan2.2`.
- Training uses GPU0-6 only. GPU7 is excluded.
- Training data is clean-1000. Probe fitting uses a deterministic train/heldout split of clean-1785.
- Both splits must retain the existing zero-leakage receipt against dev-fast20 and official test manifests.
- The contaminated historical gated-step10 checkpoint is never a formal parent.
- T5 and VAE remain outside the training hot path.
- Wan, the parent Adapter, and the probe remain frozen during v6 training.
- No LoRA, Pose, PRoPE, Depth, V-JEPA, DMD, GAN, temporal probe layer, or temporal correction branch is introduced.

## Clean parent

Reproduce support-gated step10 from the clean S1A125 checkpoint with the clean-1000 manifest, deterministic seven-rank replay, and complete source/dataset/checkpoint provenance. The parent gate requires zero evaluation leakage, correct action better than swap/reverse/random, routing improvement, healthy loss/gradients, and a three-step production smoke with no black/broken sample. It does not consume a separate matched fast20 generation budget.

## Frozen gripper probe

The probe consumes one clean latent frame at a time with shape `48x30x40`. A shared frame-local convolutional network produces two `60x80` heatmaps. Soft-argmax returns left/right positions in normalized raster coordinates. The architecture must contain no temporal operator and must produce the current output independently of all future latent frames.

Labels come from raster channels 1 and 6 under the exact `81 -> 21` causal grouping. Latent0 is invalid. A token is valid only when the source provides reliable image-space observability; unknown occlusion is invalid rather than inferred from URDF geometry. Velocity is valid only when both adjacent position tokens are valid.

The clean-1785 probe split is deterministic, task-stratified, hash-bound, disjoint, and checked against evaluation manifests. Heldout gates are median position error <= 1.5 raster pixels, P90 <= 4 pixels, correct-arm error below swapped-arm error, per-arm and bimanual/crossing gates, and exact frame-local future-independence.

## V6 correction

For each arm and block 8/16/24, reuse the frozen parent Adapter's `256`-dimensional encoded raster tokens and add an independent rank-8 delta:

`256 -> A(8) -> B(3072) -> arm condition support -> Wan block`

`A` is small random initialization and `B` is exact zero initialization. Therefore the complete v6 model is bitwise equal to the parent at step0 while `B` receives a nonzero first-step gradient. Padding, null action, action scale, and support application occur after projection and remain fail closed.

## Objective and calibration

The Wan flow parameterization is:

`z_t = (1 - sigma) * z0 + sigma * noise`, `v_target = noise - z0`.

Therefore `z0_hat = z_t - sigma * v_prediction`. The frozen probe receives `z0_hat` without detach, so gradients flow only into the v6 delta.

The objective is:

`L = L_weighted_FM + lambda_pos * L_position + lambda_vel * L_velocity`.

Position and velocity use masked Huber loss in normalized raster coordinates. Latent0, unobservable tokens, and velocity edges adjacent to any invalid token are excluded with per-sample normalization.

Before training, the clean parent is evaluated in fixed sigma buckets. The trajectory loss is enabled only in buckets whose parent `z0_hat` probe error passes the frozen reliability rule. A fixed `w_traj(sigma)` is stored in the run contract. Lambda calibration uses gradient norms on the zero-initialized `B` weights only, targeting position/FM = 0.4 and velocity/FM = 0.2. Calibration fails on zero/non-finite gradients or out-of-contract lambdas and is immutable after step0.

## Anti-exploitation gate

At discovery step25, 4-8 heldout episodes are evaluated without gradient through an independent RGB-space trajectory proxy. Latent probe error and RGB-space trajectory error must improve in the same direction. Any latent-only improvement stops the experiment.

## Bounded run

- Seven ranks, micro-batch one, 81 frames, 480x640.
- Production smoke: three complete forward/backward/step/zero-grad iterations; every rank allocated and reserved peak <22 GiB.
- Training checkpoints: 10/25/50/100.
- Step25 requires position and velocity error each improve >=15%, correct better than time-shift/reverse/swap, routing retention >=90%, and no material FM regression.
- Step50 runs matched fast20 only after mechanism and RGB anti-exploitation gates pass.
- Step100 runs only when step50 remains healthy and improving.
- Matched video comparison contains clean S1A125, clean-gated-step10, and at most two v6 candidates.
- Promotion requires DTW improvement >=5%, paired wins >=12/20, detector coverage loss <=5 percentage points, and zero black videos.

