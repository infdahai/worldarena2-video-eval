# FlowWAM Object-Response Bridge Campaign Design

Date: 2026-08-30  
Status: architecture approved; implementation plan frozen  
Deadline: 2026-09-04

## 1. Decision

This design replaces the proposal/3 targeted-LoRA execution DAG as the main
WorldArena 2.0 Track 1 improvement strategy. Proposal/3 remains immutable
historical evidence, but its V1/V2/I1/I2/C1/C2/R1 arms must not be registered,
trained, or used to fill idle GPUs.

The new main mechanism is an **Object-Response Bridge (ORB)** built on the
Official FlowWAM parent. It learns how robot-only desired flow causes contact,
non-robot object motion, and RGB state changes. It does not repeat the old Wan
action-controller, ordinary q/v LoRA, generic flow-matching, or global video
refinement experiments.

The frozen fallback remains the completed P0 selector and its release
artifacts. No experiment may overwrite P0, final-p0-r3, historical Hugging Face
revisions, or submission files.

## 2. Evidence and optimization target

The frozen development references are:

- fresh40 exact P0 replay: corrected15 `0.671070`; frozen selector chose seed1
  for 20 episodes and seed4 for 20 episodes;
- expanded Official FlowWAM seed1 clean50: EWMScore_P `65.4107`, nonmotion12
  `0.707392`;
- expanded clean50 bottlenecks: Photometric `0.1621`, Aesthetic `0.3876`,
  Image `0.5211`, Trajectory `0.6291`, Interaction `0.664`, Instruction
  `0.692`;
- Motion Smoothness GT-cap hit rate `84%`; the combined remaining mean headroom
  of Dynamic, Flow, and Motion Smoothness is too small to justify a motion-only
  branch.

All promotion decisions use:

`EWMScore_P = 100 * mean(corrected 15 metrics)`

Dynamic Degree, Flow Score, and Motion Smoothness are capped per episode at the
matched-GT value. Generated-only nine metrics may select deployable candidates,
but never replace corrected15 promotion evidence.

clean50 and fresh40 are development evidence, not official leaderboard scores
or final unbiased holdouts.

## 3. Historical do-not-repeat boundary

The following mechanisms were already tested in Wan v1-v15, FlowWAM v16, or
the adaptive candidate system and are closed:

- action raster, pose, support, SE(3), dual-arm geometry, TCN, EEF probes and
  persistent controllers;
- generic self/cross-attention q/k/v/o LoRA, relation attention, native-band
  modulation, phase locking, and ordinary RGB flow-matching continuation;
- counterfactual losses that improve internal representations without decoded
  RGB direction, magnitude, locality, coverage, and DTW evidence;
- late q/v/o plus generic mask, temporal, depth, high-frequency, or global
  smoothing losses;
- additional seed1/seed4, flow16/20/24, ordinary text-CFG, Winner-SFT,
  Preference-LoRA, and standard-FM sweeps;
- a fourth unmodified SeedVR2 attempt on 24 GiB GPUs;
- full15-oracle or matched-GT per-episode selection at deployment.

Existing proposal/3 R1 is specifically closed. It can modify action support,
overlaps the failed v13/global-refiner family, and lacks a decoded-RGB
trajectory/contact gate.

## 4. Causal hypothesis

Official FlowWAM conditions RGB generation on a clean robot-only flow stream.
The released training data provides the robot motion, but not an explicit
representation of which object was contacted, how that object should move, or
how its post-contact state should persist.

The main hypothesis is therefore:

> The largest actionable non-motion deficit is caused by a missing causal
> bridge from robot flow to non-robot object response and RGB state change, not
> by insufficient LoRA rank or insufficient optimization of robot motion.

The hypothesis is rejected unless decoded videos, rather than only internal
losses or probes, show correct response direction, locality, contact timing,
and improved corrected15.

## 5. ORB architecture

### 5.1 Training-only response targets

For released training episodes, construct deterministic pseudo supervision
from legal training data only:

1. compute full-scene forward/backward optical flow from adjacent RGB frames;
2. read the codec-input robot-only raw flow before video encoding;
3. align both flows in the same pixel coordinate system;
4. derive robot support, occlusion/confidence, and non-robot residual motion;
5. combine gripper state, end-effector proximity, temporal flow convergence,
   and RGB residuals to estimate contact transition support;
6. propagate high-confidence non-robot residual motion after contact to form
   object-response support;
7. define hard static support only where robot, contact, object response,
   occlusion, and full-scene motion are absent.

Where front/head/wrist views and camera calibration are available, multi-view
agreement is a training-only confidence filter. It is not a mandatory inference
input and cannot be replaced by evaluator masks, hidden GT metrics, fresh40, or
official-test information.

Every pseudo-label artifact is bound to repository revision, archive SHA,
episode manifest SHA, camera selection, flow code SHA, thresholds, and output
SHA. Low-confidence pixels are ignored rather than relabeled as static.

### 5.2 Object-response predictor

A small predictor consumes first-frame RGB features, instruction/task-family
tokens, robot-only flow tokens, gripper state, and end-effector trajectory. It
predicts:

- contact probability over time;
- non-robot object-response flow/residual;
- occlusion/confidence;
- a persistent post-contact state mask.

The first mechanism smoke uses frozen parent features and a zero-initialized
predictor. It must not update the Official FlowWAM backbone.

### 5.3 Asymmetric Flow-to-RGB bridge

The bridge is a zero-initialized, low-rank residual path in pre-registered
middle and late shared DiT blocks:

- robot-flow and predicted object-response tokens may write residual
  information into RGB tokens;
- RGB tokens cannot modify the frozen clean-flow condition through this new
  path;
- the Official parent, VAE, text encoder, and original flow stream remain
  frozen in the first round;
- trainable parameters are limited to the response predictor, bridge
  projections, and explicit gates.

The Official 5B backbone has 37,200 tokens per stream at 121 frames and
640x480. ORB therefore must not introduce full-length, full-width
cross-attention. Round one uses four pre-registered sparse bridges with a
256-wide, eight-head response path. The immutable pure-flow source is pooled
per time step to at most 6x8 spatial tokens before it becomes bridge K/V.
Production smoke is batch 1 at the final 121-frame 640x480 contract and must
remain below 21.5 GiB peak allocated memory; reducing resolution or frame count
cannot be used to claim that the production memory gate passed.

The pure-flow source is captured immediately after flow patchification and
detached. Later flow tokens have already participated in the Official joint
RGB/flow attention and are not a valid pure source for the new branch. Each ORB
residual is applied after the complete frozen Official block and before that
block returns. Only the new ORB branch is strictly one-way; the frozen Official
main path remains bidirectional.

This is not a generic q/v LoRA sweep. The trainable interface is defined by
direction and information type, and its zero gate must reproduce the parent
bit-for-bit within the frozen numerical tolerance.

Parent and ORB weights are stored separately. Inference first loads the
original parent through the Official pipeline, verifies its SHA, then strictly
loads the ORB-only `orb_bridges.*`/predictor/gate state. ORB keys must never be
mixed into a full parent checkpoint because the Official permissive loader can
silently discard unknown non-flow keys.

### 5.4 Prior preservation

The frozen Official parent acts as a teacher:

- outside robot/contact/object-response support, candidate predicted-x0 and
  decoded RGB must remain close to the parent;
- inside object-response support, the candidate learns released RGB state
  change and flow-warp consistency;
- frame 0 is immutable;
- task semantics are protected by correct-instruction versus mismatched-task
  contrasts using training tasks only.

The teacher prevents low-resolution or limited-domain RoboTwin data from
destroying Image, Aesthetic, JEPA, Background, Depth, Perspective, or Semantic
quality.

## 6. Data-resolution gate

Before any visual-quality claim, audit the native supervision resolution of
each randomized500 task and the exact resize path used by the Official recipe.

- Native or genuinely high-resolution RGB may support visual-detail losses.
- Inputs whose supervision is only 320x240 or bicubic-upsampled may train
  contact/object response but cannot be presented as evidence for learning new
  640x480 detail.
- If no adequate high-resolution supervision exists, Image/Aesthetic work is
  restricted to parent-preserving inference methods and must not use synthetic
  sharpness as a training target.

## 7. Fast inference branch

A separate no-training branch may test a flow-aware static anchor compositor:

- build deployable robot, contact, object-response, generated full-scene-flow,
  and occlusion masks;
- copy or warp parent/first-frame detail only inside high-confidence hard-static
  support;
- apply feathering only at the static boundary;
- make the RGB delta exactly zero on frame 0 and throughout robot, contact, and
  object-response support.

This branch is an unfinished historical prerequisite, not a novel model claim.
It is allowed because the old background-projection experiment was closed for
missing assets rather than for negative video evidence. It must pass decoded
RGB trajectory, interaction, instruction, and structure gates before full15.

No full SeedVR2 retry is allowed. A lightweight refiner is permitted only if its
residual is hard-zero, not attenuated, on robot/contact/object-response support.

## 8. Task-family specialization

Global adapters are not the default. After the shared ORB mechanism passes its
causal smoke, specialists may be trained for:

- contact/press;
- grasp/transport;
- articulated open/close;
- placement/stacking;
- bimanual coordination.

Routing uses instruction and known task metadata only. A global frozen-parent
fallback is mandatory. Specialists are unlocked only after a shared ORB result
demonstrates that task-family gradients are materially conflicting.

## 9. Gates and evaluation sequence

### 9.1 Mechanism gate

Before n8 generation, the bridge must pass:

- zero-gate parent equivalence;
- correct versus reversed, swapped, time-shifted, and zero robot flow;
- decoded-RGB object displacement direction and contact-timing checks;
- static-region leakage and robot/object localization checks;
- no frame-0, frame-count, resolution, black-frame, or SHA contract failure.

Internal loss reduction, attention magnitude, or probe accuracy alone cannot
promote a candidate.

### 9.2 Video gates

Candidates proceed asynchronously:

- n8 feasibility: structure pass, decoded causal response above random, no
  catastrophic protected-metric regression;
- n24: corrected15 at least P0 `+0.002`, primary target positive, no protected
  metric below `-0.015` point delta;
- n64: corrected15 at least `+0.003`, nonmotion12 at least `+0.004`, primary
  target at least `+0.02`, LCB90 above zero, and Instruction, Interaction,
  Trajectory, JEPA, Perspective each at least `-0.005`;
- n128: corrected15 at least `+0.004` with LCB95 above zero, nonmotion12 at
  least `+0.005` with LCB95 above zero, paired wins at least `76/128`, and all
  protected-metric LCB95 values above `-0.005`.

Final promotion requires the frozen task/sample-disjoint holdout. Official
test1000 is never used for training, threshold tuning, task routing, candidate
selection, or per-episode winners.

## 10. Eight-GPU asynchronous schedule

There is no campaign-wide barrier. Each physical UUID owns an independent
claim and pulls only hash-complete, non-duplicate work.

Initial allocation after code/data gates pass:

- GPU0-1: full-scene flow, confidence, contact, and object-response pseudo-label
  production;
- GPU2-3: two pre-registered ORB bridge variants;
- GPU4: prior-preservation/distillation ablation;
- GPU5: flow-aware static compositor candidates and parent generation;
- GPU6: generated9, trajectory, and corrected15 scoring backlog;
- GPU7: task-family/lattice control or additional gate-critical scoring.

The allocation is elastic. A completed pseudo-label shard immediately releases
its UUID to training or scoring. A completed checkpoint immediately enters n8;
a completed video immediately enters structural and cheap-score validation.

Existing legal downloads, immutable-data validation, and already-running
non-conflicting official scoring jobs are preserved. Old proposal/3 training,
R1, Q0, ordinary LoRA, and duplicate generation are not valid work for filling
idle GPUs.

## 11. Budget and automatic iteration

Hard limits:

- at most three architecture rounds;
- at most six ORB or protected-inference candidates;
- at most 1600 new raw videos;
- at most 800 full15 video evaluations, with 256 reserved for final evidence;
- at most 384 GPU-hours before explicit user expansion.

A mechanism closes after two independent failures of the same causal
hypothesis. Three rounds without corrected15 uplift of at least `0.003` return
the campaign to frozen P0. Failures produce receipts and a diagnosis; they do
not trigger unregistered rank, LR, seed, block, or threshold sweeps.

## 12. Safety and current integration boundary

The pinned Official FlowWAM source commit is
`f06fa46042e97738c6619c868f1097be6749d48d`. The immutable dataset-root binding implementation has completed independent
review and is available as a clean integration commit based on the current
targeted15 worktree. Integrating it does not authorize old proposal/3 training.

Before any remote launch, verify physical GPU index and UUID, compute PIDs,
user, PPID, PGID, full command line, cwd, memory, utilization, project process
tree, claim, and flock. Never stop, signal, modify, or interfere with
`/data/fjy`, `/data/whn`, or any external task. Project process termination is
allowed only after a same-turn exact ownership audit and only when required by
the approved ORB migration.

The final ZIP and email remain separate, user-confirmed release actions.
