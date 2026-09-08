# Wan-Action-Lite+ v3 adaptive-compression design

## Goal

Meet the 2026-08-30 submission deadline on exactly 8 x RTX 4090 24GB without
changing the 81-frame, 480x640 target distribution, action-conditioning
contract, or official evaluation semantics. Reduce only low-information
optimizer steps and redundant full evaluation rounds.

## Production-shape memory gate

Before Stage-1, run the exact frozen Wan2.2-TI2V-5B backbone, v3 pose-enabled
action adapter, BF16/FSDP configuration, micro-batch one, 81 frames, and
480x640 resolution on eight ranks. Run three consecutive
`forward -> backward -> optimizer.step -> zero_grad` iterations and record
per-rank peak allocated bytes, peak reserved bytes, and each step duration.
Every rank must remain below 22 GiB allocated and reserved; any OOM,
non-finite result, increasing leak signal, or rank failure blocks training.

The Stage-1/2 processes consume only cached latent, cached text context, and
cached action conditioning. Loading Wan T5 or VAE modules in a training/smoke
process is a hard failure. Aggregate GPU memory is never treated as pooled
memory.

## Adaptive optimizer budget

Stage-1 has a 200-step aggregate ceiling: a shared raster-only prefix from
step 0 through 50, Branch A from 50 through 125, and Branch B from 50 through
125. Both branches use the paired replay contract. At step 125 both run the
dev-fast-20 trajectory/action-adherence gate; only the winner advances to
dev-clean-50. Extra branch steps require an explicit inconclusive or
quality/trajectory trade-off result and are not part of the default launcher.

Stage-2 defaults to 400 optimizer steps, with fast trajectory evaluation at
100/200/300/400 and complete WorldArena + WorldArena_VLM + WorldArena_JEPA
development evaluation only at 200 and 400. An extension to 600 is allowed
only when trajectory is still clearly improving at 400, all no-regression
gates pass, and training has not plateaued. Step 800 is removed from the
default contract.

## Evaluation and deadline

Trajectory Accuracy is the primary objective. JEPA Similarity, Subject
Consistency, Image Quality, Motion Smoothness, and the remaining official
metrics are no-regression gates against the calibrated native Wan2.2 baseline.
If Stage-2 extends, step 600 receives one complete development evaluation.

Checkpoint and CFG are frozen by 2026-08-24. From 2026-08-25 through
2026-08-29 the only default work is test-1000 generation, WorldArena,
WorldArena_VLM, WorldArena_JEPA, submission packaging, validation, and failure
recovery. Post-freeze model changes require a concrete blocker.

## Explicit exclusions

Do not reduce frame count or spatial resolution. Do not add V-JEPA to
training, Depth, DMD, GAN, object-mask loss, a trajectory auxiliary head,
sequence parallelism, or randomized-500 expansion in this iteration.
