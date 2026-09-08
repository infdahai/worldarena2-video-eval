# v14 FlowWAM WA2 Pivot — Execution Plan

## Objective

Replace the failed small action-controller line with the released FlowWAM
WorldArena checkpoint and test the paper's native action representation:
robot-only optical-flow video.  The incumbent `clean-gated-step10` remains the
fallback and comparison baseline; it is not the v14 parent.

## Evidence-bound architecture

Pinned upstream: `YixiangChen515/FlowWAM@68abaa2`.  Pinned checkpoint:
`flowwam_worldarena_stage1.safetensors`, 10,137,267,208 bytes, SHA-256
`e211e32b...96c4`.

The paper's world-model mode is stricter than the public policy server:

```text
RGB first-frame latent + RGB noise --t>  shared dual-stream Wan  --> RGB velocity
clean desired flow latent -------0>  (joint RGB/flow attention) --> never stepped
```

All flow tokens use timestep zero; RGB frame-zero tokens use zero; future RGB
tokens use the scheduler timestep.  Only RGB is updated after each denoising
call.  The public policy server's joint RGB+flow denoising path is not used.

## Phase 1 — Action-to-flow gate

Use the released paired standard/robot-only HDF5 files as canonical outputs of
the official SAPIEN renderer.  Select 20 task-balanced training episodes with
task-level exclusion against dev-fast20 and dev-clean50.  Decode exactly 81
frames, run torchvision RAFT-large, mask to robot pixels, resize vector fields
with component scaling, and use the released 8-bit HSV codec with magnitude 25.

Every episode must pass:

- 81 frames and exact-white frame zero;
- codec support IoU at least 0.90;
- direction cosine at least 0.95;
- median codec EPE at most 2 px;
- codec round-trip max EPE at most 0.5 px;
- left/right commanded-motion identity accuracy 100%.

Identity is measured with a renderer-level counterfactual, not an image-half
heuristic: render the named left/right joint streams, render the same episode
with arm and gripper streams exchanged, run the identical RAFT path, and
require the correct render's median EPE against the released robot-only flow to
be strictly lower for every audit episode.

The cache is immutable, hash-bound and stored only under
`/data/di/worldarena2_track1_20260815/runs/v14-flowwam-wa2`.

## Phase 2 — Zero-shot only

Load the exact released checkpoint with a strict key/shape inventory.  Run a
single-GPU production smoke on GPU6 at 81 frames, 480x640, 50 steps.  Peak
allocated and reserved memory must both remain below 22 GiB.  T5 and VAE may be
offloaded; desired-flow VAE latents should be precomputed before reducing
resolution.

The strict load must preserve the released fp32 modulation, timestep-MLP and
LayerNorm runtime path used by the official loader; silently casting these
checkpoint tensors into the bf16 DiT is a failed load even when keys match.

Then decode four fixed unseen-task candidates (two single-arm, two bimanual)
with one seed and frozen generation parameters.  Compare with the incumbent.
Zero-shot advances only if at least two of four improve trajectory, black count
is zero, robot coverage falls by no more than ten points, no wrong-arm failure
appears, and object behavior is not catastrophic.

## Phase 3 — Conditional follow-up

Only after RGB4 passes, run matched dev-fast20.  Only after zero-shot fast20
passes may a fine-tune be built.  The first fine-tune is limited to Q/K/V/O
LoRA rank 16 on all shared blocks plus flow patch embedding and stream embedding,
uses RGB flow-matching with light motion weighting, and stops at 0.125/0.25
effective epoch gates.  No auxiliary action architecture or loss is permitted.

## Hard exclusions

- no official-test use during model selection;
- no dev-fast20/dev-clean50 task in training or audit20;
- no joint stepping of the clean flow condition;
- no checkpoint partial-load acceptance;
- no write outside the formal artifact root;
- no GPU0–5 or GPU7 use;
- no fine-tuning before zero-shot gates pass.
