# v13-ScoreBoost Design

Date: 2026-08-19  
Status: Approved architecture; implementation pending written-spec review.

## 1. Objective

v13 stops action-controller research and optimizes WorldArena Track-1 aggregate quality while preserving the current trajectory incumbent.

The experiment asks one bounded question:

> Can low-rank adaptation of Wan late self-attention improve robot/object integrity, temporal object persistence, and scene geometry without materially regressing the trajectory behavior of `clean-gated-step10`?

No phase model, SE(3) controller, action Jacobian, PRA, top-k routing, GRU/TCN, V-JEPA, GAN, DMD, or new action branch is permitted.

## 2. Fixed architecture

```text
clean-gated-step10 parent                 frozen
Wan blocks 0-17                          frozen
Wan blocks 18-25 self-attn Q/V/O LoRA    trainable, rank/alpha 16/16
Wan blocks 26-29                         frozen
training-only depth head                 trainable
T5 / VAE / SAM3 / depth teacher          frozen and absent from training hot path
```

The existing frozen raster parent remains the only action-conditioning path. v13 does not modify its raster, support, arm routing, or injection points.

LoRA is injected only into `self_attn.q`, `self_attn.v`, and `self_attn.o` in blocks 18 through 25 inclusive. The exact trainable whitelist contains only LoRA down/up weights and the depth head.

Learning rates are fixed, not swept:

```text
LoRA Q/V/O: 5e-6
depth head: 1e-4
```

## 3. Training data and leakage contract

The source pool is the existing clean optimizer-2060 manifest. The v13 training manifest contains exactly 150 unique episodes selected deterministically and task-balanced from that pool.

It must have zero overlap with:

- audit20;
- dev-fast20;
- dev-clean50;
- any frozen official-test manifest if/when it becomes available.

Selection, source manifest, exclusion manifests, cache files, and the final 150-row manifest are bound by SHA256 receipts. Missing, duplicate, unreadable, or overlapping rows fail before CUDA initialization.

The fixed RGB8 gate is also held out from the 150 training rows. It contains four single-arm-dominant and four bimanual/crossing episodes where SAM3 can observe the robot and manipulated object.

## 4. Offline supervision cache

Only the 21 RGB frames corresponding to the 21 Wan latent time slots are processed. Full 81-frame labels are not generated for this bounded experiment.

### 4.1 SAM3 masks

The existing frozen SAM3 checkpoint at the formal artifact root is used offline.

For each aligned frame, cache:

- robot union mask from the reviewed robot-arm/end-effector/gripper prompts;
- manipulated-object mask from a source-controlled task-to-object prompt map;
- validity, confidence, connected-component, area, and temporal-continuity metadata.

Masks are resized conservatively to the `21×30×40` latent grid. Robot and object masks may overlap. A failed or temporally implausible object track invalidates the sample; the sampler deterministically refills it from the clean optimizer pool. It never writes a guessed empty mask as a valid label.

SAM3 is never loaded by the trainer.

### 4.2 Relative depth

Depth Anything V2 Small is downloaded once under the formal `/data/di/worldarena2_track1_20260815` artifact root and pinned by model-file SHA256.

It runs offline on the same 21 aligned RGB frames. Each clip stores finite relative inverse depth normalized by robust per-clip median and MAD, then resized to `21×30×40`. The cache includes model hash, source RGB/HDF5 identity, frame indices, normalization statistics, schema version, and payload hash.

The depth teacher is never loaded by the trainer or inference path. Native HDF5 does not contain depth and is not treated as though it did.

## 5. Training losses

Let `z0` be the cached clean latent and `z0_hat` the clean latent reconstructed from the current flow prediction.

### 5.1 Base flow matching

Standard valid-latent flow matching remains the main video-distribution objective:

```text
L_fm = valid-latent normalized flow-matching loss
```

### 5.2 Robot/object masked clean-latent loss

The per-token mask is:

```text
w = 1 + 2 * robot_mask + 4 * object_mask
```

Weights are normalized per sample over valid latent tokens. The loss compares `z0_hat` and `z0`; it does not decode RGB in the training loop.

```text
L_mask = normalized weighted SmoothL1(z0_hat, z0)
```

### 5.3 Object temporal loss

For adjacent valid object-mask pairs, match predicted and ground-truth latent changes inside the union object support:

```text
L_temporal = SmoothL1(
    z0_hat[t+1] - z0_hat[t],
    z0[t+1]     - z0[t]
) within valid object union masks
```

Pairs with missing/invalid object masks are excluded, not zero-filled. This objective penalizes disappearance, teleportation, and temporal deformation without adding a new temporal network.

### 5.4 Training-only depth head

The head reads block-25 hidden tokens, applies LayerNorm and a small MLP to one depth value per `15×20` visual token, then bilinearly resizes to `30×40`.

It predicts normalized relative inverse depth and uses a robust scale/shift-invariant SmoothL1 loss. Gradients flow through the head and the selected LoRA parameters; the frozen native Wan weights remain frozen.

### 5.5 Fixed objective

```text
L = L_fm + 0.5 * L_mask + 0.1 * L_temporal + 0.1 * L_depth
```

No counterfactual, phase, trajectory, JEPA, adversarial, or additional auxiliary loss is added in v13.

## 6. Training schedule

Topology is selected once before launch. Under current ownership constraints, physical GPU6 is the only permitted GPU. GPU0-5 and GPU7 are not touched.

```text
micro-batch: 1
optimizer exposures: 150 maximum
checkpoint: 100 and 150
warmup: 10 exposures
LoRA LR: fixed after warmup
depth-head LR: fixed after warmup
```

Before formal training, a production-shape smoke executes three complete forward/backward/step/zero-grad cycles and requires:

- peak allocated and reserved memory below 22 GiB;
- nonzero finite Q/V/O LoRA and depth-head gradients;
- exact frozen-parameter whitelist;
- no VAE, T5, SAM3, or depth-teacher module loaded;
- all persistent outputs under the formal `/data/di` root.

Exposure100 is a health checkpoint. Training stops there only for NaN/OOM, frozen-parameter gradients, loss explosion, invalid cache, or source/lineage drift. Otherwise it completes exposure150.

## 7. Evaluation order

### 7.1 Incumbent profile first

Generate the missing `clean-gated-step10` dev-clean50 videos under frozen inference parameters and run the complete development metric set:

- WorldArena base metrics;
- WorldArena_VLM;
- WorldArena_JEPA;
- trajectory and coverage diagnostics.

These are development results, not official test-1000 claims.

### 7.2 Matched RGB8 gate

Generate exactly the same held-out RGB8 for `clean-gated-step10` and v13 exposure150, with identical seed, prompt, resolution, denoise steps, CFG, and action scale.

v13 passes only if all conditions hold:

- trajectory is no worse than the parent beyond a 2% failure-aware tolerance;
- SAM3 coverage is not lower;
- black/broken videos equal zero;
- object-disappearance count decreases;
- penetration/deformation count decreases;
- neither object-integrity failure count increases.

The last two items use fixed per-episode SAM3 track statistics plus a blinded paired review record. At least one of disappearance or penetration/deformation must improve.

If RGB8 fails, v13 stops and fast20 is not run.

### 7.3 Fast20 and inference search

After RGB8 passes, run matched fast20 against `clean-gated-step10`.

Required:

```text
paired trajectory win rate >= 55%
trajectory mean improvement > 0
coverage not lower
black = 0
```

Only after that gate, evaluate inference parameters. Avoid a blind Cartesian sweep:

1. action scale `1.0 / 1.25 / 1.5` at text CFG 5;
2. text CFG `4 / 5 / 6` at the selected action scale.

Jobs use atomic video-level claiming. Additional GPUs may join only after a fresh ownership check; no unrelated process is stopped.

## 8. Failure handling and terminal decision

- Missing or invalid SAM3/depth cache: fail before training.
- No depth teacher hash: fail before cache publication.
- Training regression/OOM: stop at the last atomic checkpoint.
- RGB8 failure: reject v13; retain `clean-gated-step10`.
- Fast20 failure: reject v13; no wider generation or official-test use.
- Fast20 pass: run dev-clean50 complete profile for the selected v13 checkpoint, then compare total-score dimensions before any final freeze.

This experiment does not reopen action architecture research regardless of outcome.
