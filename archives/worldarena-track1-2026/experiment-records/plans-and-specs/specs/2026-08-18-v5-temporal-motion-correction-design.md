# Wan-Action-Lite+ v5 Temporal Motion Correction Design

## Decision

Build one bounded architecture experiment named `v5-temporal-motion-correction`.
The frozen incumbent is the support-gated step-10 checkpoint. The Wan backbone,
incumbent Adapter, raster schema, 81-to-21 packing, spatial support, weighted
flow-matching objective, and inference contract remain unchanged.

The experiment adds a removable, zero-initialized per-arm temporal motion
correction at Wan blocks 8, 16, and 24. It must prove that temporal routing
improves without losing the incumbent's spatial routing or video quality within
24 hours. Otherwise it is stopped.

## Evidence and Scope

- Input alignment passes: spatial median 0.70 raster pixels, P90 1.73, best lag
  zero, correct arm mapping win rate 100%, flow cosine 0.743.
- Support gating on the frozen S1A125 checkpoint raises same/cross routing from
  approximately 1.0 to left 3.14 and right 2.85.
- The trained gated-step10 incumbent improves matched-fast20 DTW from 18.09 to
  16.27 and coverage from 65.1% to 87.5%.
- The v4 dual-stream Pose-FiLM and coordination branch lowers static arm-swap
  rate but worsens DTW by 110.7% and is retired.

This experiment does not change data scale, add LoRA, Pose, PRoPE, V-JEPA,
Depth, GAN, trajectory loss, coordination loss, or a new support definition.

## Architecture

### Frozen parent

The exact gated-step10 checkpoint is loaded and hash-pinned. Its Adapter and
Wan backbone are frozen. Parent residuals continue to be support-gated and are
injected at blocks 0, 8, 16, and 24.

At step zero, the v5 wrapper must be bit-identical to the frozen parent.

### Motion input

Each arm consumes only its existing raster `flow-x`, `flow-y`, and the temporal
difference of its opening-map channel. Framewise spatial encoding maps the
three channels from 60x80 to 15x20. Canonical causal grouping then maps 81
frames to 21 latent timesteps.

No EEF delta, speed, Pose, opposite-arm features, or learned arm embedding is
added in this experiment.

### Local causal router

At each 15x20 spatial token, a shared two-block causal depthwise temporal TCN
processes the per-arm motion features. Both blocks use kernel size 3; dilation
is 1 then 2, giving a seven-token receptive field. Each block contains a
depthwise causal Conv1D, pointwise projection, SiLU, and residual connection.

The left and right arms run through the shared router independently. No tensor
containing both arms enters the router.

### Arm-level temporal context

For each arm and timestep, locally encoded features are spatially pooled using
the unchanged arm support. A small shared causal TCN processes the resulting
per-arm `[T,C]` sequence and predicts a bounded context gate. Left and right
pooling remain separate.

This path represents phase context such as approach, grasp, transport, and
pause without global left/right mixing.

### Explicit activity gate

Each arm gets an independent continuous activity value derived only from its
own flow magnitude and opening-map delta, using the already established quiet
and active thresholds. No winner-take-all comparison is allowed.

- both arms active: both gates remain active;
- one arm active: only the quiet arm is attenuated;
- both quiet: both corrections are attenuated;
- ambiguous: continuous values are preserved;
- spatial support overlap and arm crossing remain legal.

### Correction and injection

For each arm:

```text
local causal feature
  * learned arm-level context gate
  * explicit activity gate
  * unchanged spatial support
  * action_present
  -> arm-specific zero-init projection
```

Left and right projections are independent. Their corrections are summed only
after final masking. Corrections are injected at blocks 8, 16, and 24. Block 0
is owned exclusively by the frozen parent.

## Training Contract

- GPUs: 0-6 only; GPU7 is excluded.
- Production shape: 81 frames, 480x640 video latent, micro-batch 1.
- Parent and backbone: frozen.
- Trainable parameters: v5 router, context gate, and projections only.
- Objective: existing weighted flow matching only.
- Maximum steps: 50.
- Checkpoints: 5, 10, 25, 50; step 5 is smoke-only.
- Warmup: 5 optimizer steps.
- Projection LR: `1e-4`.
- Router/context LR: `5e-5`.
- No LR sweep.
- Persistent outputs: `/data/di/worldarena2_track1_20260815` only.

## Hard Gates

### Before training

- step-zero wrapper output is `torch.equal` to gated-step10;
- future action impulse has exactly zero effect on current router correction;
- left-only input produces exactly zero right correction and conversely;
- BOTH keeps both corrections;
- support exterior is exactly zero after projection;
- crossing and overlap do not impose mutual exclusion;
- all trainable parameters match an exact whitelist;
- three production forward/backward/step cycles stay below 22 GiB allocated
  and reserved on every training rank.

### Checkpoint audit

At steps 10, 25, and 50:

- correct action wins reverse, swap, and random-valid on at least 8/8 discovery
  episodes;
- spatial same/cross routing does not materially regress from the incumbent
  values of left 3.14 and right 2.85; values near the 1.5 floor do not pass;
- router future-to-current leakage is numerical zero;
- model-level future-to-current leakage falls at least 30% relative to the
  frozen gated-step10 incumbent;
- inactive-arm leakage decreases.

If router leakage falls but model-level leakage is unchanged, training stops
without waiting for step 50: Wan temporal mixing is the identified bottleneck.

### Video promotion

At most two audit-passing checkpoints enter the exact matched dev-fast20.
Relative to gated-step10, promotion requires all of:

- mean DTW at or below 15.46 (at least 5% improvement from 16.27);
- paired wins at least 12/20;
- coverage drop no more than 5 percentage points;
- detector failures do not increase by more than one episode;
- black-video count is zero;
- bimanual temporal leakage does not regress on human review.

## Stop and Fallback

The architecture experiment ends within 24 hours.

- Mechanism gate fails by step 50: retire v5.
- Router leakage improves but model leakage does not: move to arm-aware causal
  masked attention; do not train the TCN longer.
- Model leakage improves but matched-fast20 trajectory does not: temporal
  leakage is not the principal trajectory bottleneck; evaluate lightweight
  gripper-trajectory supervision instead of enlarging the router.
- Any safety or video-quality regression: retain gated-step10 as incumbent.

