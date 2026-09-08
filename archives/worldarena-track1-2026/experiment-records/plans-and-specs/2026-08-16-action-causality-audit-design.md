# Action Causality Audit — approved design

## Objective

Determine whether Wan-Action-Lite+ uses the commanded action causally, and
where the signal fails when it does not.  This audit compares the frozen
`S1A125` parent with its `q-only50` child.  It does not train either model and
does not promote a checkpoint.

## Scope

- Eight deterministic discovery episodes: two left-arm dominant, two
  right-arm dominant, two dual-arm, and two large-displacement/contact clips.
- Checkpoints:
  - `S1A125`: raster-only Action Adapter parent.
  - `q-only50`: the same frozen Adapter plus q-only LoRA.
- Production geometry: 81 frames, 480x640 video, 21x30x40 latent grid, and
  21x15x20 action-token grid.
- Fixed latent, text context, noise, and timesteps 250/500/750 within every
  paired comparison.
- Persistent outputs are restricted to
  `/data/di/worldarena2_track1_20260815/experiments/action_causality_audit`.
- GPU7 is excluded.  No training job is started by the audit.

## Conditions

The ranked action-present conditions are `correct`, `hold`, `reverse`,
`swap`, and `random-valid`.  `null` is a separate wiring control with a zero
raster, no pose, zero support, and `action_present=0`.

Every action-present counterfactual remains anchored to the sample's first
frame.  Reverse, swap, and random-valid must be reconstructed as coherent
joint14/full-arm trajectories, pass independent FK, joint-range, projection,
and finite-value gates, or fail closed.  Direct raster-channel swapping is not
accepted as a valid counterfactual.

## Measurements

1. Per-arm encoder-feature deltas.
2. Per-arm raster and pose residual RMS at injection points 0/8/16/24, plus
   residual-to-hidden RMS.
3. Action-sensitivity ratios at blocks 0/8/16/24/29 and the final velocity
   prediction, reported globally and on left-arm, right-arm, gripper, robot,
   and background regions.
4. Unweighted flow-matching squared error partitioned into left arm, right
   arm, gripper, robot, and background regions.
5. Cross-arm gradient routing from left/right regional losses to left/right
   action features and residual components.

Geometry-derived regions are the primary contract.  SAM3 masks may be used as
an optional read-only cross-check but are not a dependency and object masks do
not enter the first audit.

## Discovery gates

- Input and checkpoint provenance is complete and SHA-256 verified.
- Repeated identical forwards establish a numerical sensitivity floor.
- Null and hold remain distinct.
- All ranked action conditions have distinct encoded action features.
- Median robot-loss margins for reverse, swap, and random-valid are positive,
  with correct winning at least six of eight episodes for each perturbation.
- Same-arm gradient routing is at least 1.5 times cross-arm routing for both
  arms.
- Output sensitivity exceeds the repeated-forward numerical floor by at least
  10 times.
- Background response is subordinate to the robot-region response.

These are diagnostic gates, not checkpoint-promotion gates.  A positive
discovery result advances to matched20.  A negative result is classified as:

- weak residual: repair Adapter scaling/normalization/injection;
- presence-only residual: repair action geometry or arm identity;
- strong residual but insensitive robot loss: repair regional supervision;
- cross-arm gradients: repair left/right routing;
- fully causal signal: resume bounded optimization.

## Artifacts

The audit writes an immutable manifest, per-episode NPZ/JSON records, a
checkpoint comparison JSON, a Markdown report, hashes for every input and
output, and an experiment-registry event.  Partial artifacts are atomic and
never count as completion.
