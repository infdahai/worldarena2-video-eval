# FlowWAM Targeted 15-Metric Campaign Design

Date: 2026-08-30  
Status: architecture approved; executable proposal manifests not frozen  
Deadline: 2026-09-04

## 1. Goal

Improve the current Official FlowWAM Stage-1 P0 under the latest WorldArena 2.0
Track 1 15-metric protocol without using official-test feedback, hidden GT labels,
or test-derived per-episode winners.

The campaign optimizes the corrected mean

`EWMScore_P = 100 * mean(corrected 15 metrics)`,

where Dynamic Degree, Flow Score, and Motion Smoothness are capped per episode by
their matched GT values. The other 12 metrics remain uncapped.

The current frozen development reference is the fresh40 P0 replay:

- parent SHA256:
  `e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`;
- selector inputs: generated-only nine metrics;
- evaluation outcomes: full corrected 15 metrics after frozen selection, never
  training labels for the selector or generator;
- corrected15 mean: `0.671070`;
- receipt SHA256:
  `3f561be07b9d6871ced57505aca6496d6c23990e1c07611d924411ac5efda7c4`.

The fresh40 score is development evidence, not an official leaderboard score.

## 2. Why the old scope-only LoRA sweep is insufficient

Changing only the q/v LoRA attachment scope while preserving the same ordinary
RGB flow-matching objective changes the gradient path but does not create a
metric-specific training signal. Historical q/v evidence moved Image and JEPA by
only roughly `0.001-0.002` while Instruction regressed. Therefore the previous
`3 self + 3 cross + 2 all` standard-FM sweep is closed as a main strategy.

One standard-FM q/v arm may remain as a control. It cannot consume the majority
of the GPU budget or be described as targeted Image, JEPA, Instruction, or
Interaction optimization.

## 3. Metric priorities

The design uses two evidence views without conflating them:

1. current Stage-1-only P0 development results, which expose low visual quality;
2. the complete public FlowWAM pipeline, which shows that FlowWAM remains a
   useful action-conditioned backbone but loses relative ground in Trajectory,
   Background, Subject, JEPA, and Interaction after the GT-cap update.

The campaign therefore targets:

1. RGB and temporal consistency: Image, JEPA, Subject, Background, Aesthetic;
2. instruction and contact: Instruction, Interaction;
3. action-to-state geometry: Trajectory and Interaction;
4. protection: Perspectivity, Depth, Semantic, Photometric, and the three capped
   motion metrics.

It does not spend a dedicated branch on raising already-capped motion scores.

## 4. Data and leakage contract

Training data comes only from the released
`YixiangChen/FlowWAM_RoboTwin` randomized-500 files at immutable revision
`506c4e014f7dbd291e7d7683c79fc685dd3e2714`.

The existing q/v trainer is frozen to the different
`YixiangChen/FlowWAM_WorldArena@a3c1...` contract and therefore cannot be reused
by changing CLI arguments. A new contract-v2 must bind the RoboTwin repository,
revision, archive SHA, extracted-file manifest, and split manifests.

The first12 task allocation is fixed before scores are opened:

- model-train6: `click_bell`, `beat_block_hammer`, `grab_roller`, `lift_pot`,
  `move_playingcard_away`, and `pick_dual_bottles`, episode indices `0-399`;
- selector-fit6: the same six tasks, episode indices `400-499`, sample-disjoint
  from model training;
- dev-gate2: `press_stapler` and `turn_switch`;
- final-holdout4: `adjust_bottle`, `click_alarmclock`, `place_object_stand`, and
  `stamp_seal`, opened once after model and selector lock.

Within each dev-gate task, episode indices `0-499` are ordered by
`SHA256("wa2-dev-v2:" + task + ":" + episode_index)`. n24 uses the first 12
per task, n64 the first 32, and n128 the first 64. The final holdout uses the
first 32 episodes per task under the independent salt `wa2-final-v2:`.

The final holdout is task-disjoint from model-train6, selector-fit6, and
dev-gate2. It is not claimed to be unseen by the Official FlowWAM parent.
Before queue registration, acquisition must write the literal episode lists,
archive/file SHAs, four manifest SHAs, and a zero-collision receipt. Until those
SHAs exist, targeted training proposals remain unregistered.

fresh40 is limited to frozen-P0 replay, bottleneck diagnosis, and non-promoting
smoke diagnostics. It does not supply n64/n128 evidence, selector fitting,
threshold calibration, tie-policy calibration, or training supervision.

Training may use released RGB, action, instruction, robot-only flow, and
deterministically derived training-only masks or frozen-teacher features. It may
not use:

- fresh40 GT, fresh40 15-metric values, or fresh40 seed winners;
- official-test data, scores, or per-episode selection outcomes;
- leaderboard outputs as sample weights or labels;
- hidden evaluation embeddings, depth, trajectory, or matched-GT metrics.

Every cache and manifest is bound to source, revision, task, episode, code, and
file SHA256. A missing or mismatched hash fails closed.

## 5. Trainable mechanisms and objectives

### 5.1 Visual and temporal consistency arms V1 and V2

These arms preserve the flow/action path at a low learning rate or frozen state
and add RGB-specific supervision.

V1 uses rank-8 LoRA with learning rate `1e-6` on RGB-stream
`self_attn.q`, `self_attn.v`, `ffn.0`, and `ffn.2`. FlowStream, T5, VAE, and
native weights are frozen. Its common objective is:

`L_V = L_RGB-FM + wx0*L_x0 + whf*L_highfreq + wtemp*L_temporal`.

- `L_x0`: SmoothL1 with beta `0.01` between predicted-x0 and released GT latent;
- `L_highfreq`: SmoothL1 after the fixed spatial kernel
  `[[0,1,0],[1,-4,1],[0,1,0]]` on every latent frame;
- `L_temporal`: adjacent-frame feature consistency in static regions and
  first-frame background anchoring.

For V1, `wx0=0.25`, `whf=0.05`, and `wtemp=0`. Timestep sampling is 75% from
actual scheduler sigma `[0.15,0.55]` and 25% from the unchanged official range.
With the current shift-5 scheduler, the implementation must verify that this
maps to the audited high timestep IDs (approximately `803-966`); it must not
mistake low IDs for low noise.

V2 adds frozen DINO feature loss with weight `0.05` on decoded output RGB frame
indices `40` and `80`. Candidate and same-frame released-GT RGB are resized to
`224x224`, normalized by ImageNet mean/std, and passed through frozen
`dino_vitb16`. The compared feature is the final transformer-block patch-token
matrix with CLS excluded. Each patch token is L2-normalized and the loss is mean
`1-cosine(candidate_patch,stopgrad(GT_patch))` over both frames and all patches.
CLS features, patch averaging, intermediate layers, L1, and L2 distances are
forbidden. V2 performs one differentiable decode of the complete predicted-x0
latent, then selects the two RGB frames; it never indexes latent time with 40 or
80. The teacher is at remote path
`official_track1_eval/weights/dino_model/dino_vitbase16_pretrain.pth`, SHA256
`bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`.
LPIPS is excluded from V2 because no dependency and weight contract is frozen.
The teacher is frozen and receives no evaluation data. V2 fails closed if
decoder gradients do not reach the adapter, peak memory is at least `21.5 GiB`,
or step time exceeds twice V1; it never silently degrades to V1.

The trainer subclass must expose `rgb_z`, `rgb_noisy`, `rgb_pred`, timestep,
sigma, and predicted-x0 from the official forward. The current q/v trainer does
not expose these values and is not an implementation substitute.

### 5.2 Instruction and contact arms I1 and I2

These arms train RGB cross-attention plus a narrowly scoped FFN adapter, not q/v
alone.

Both instruction arms use rank-8 LoRA, learning rate `1e-6`, on RGB-stream
`cross_attn.q`, `cross_attn.v`, `ffn.0`, and `ffn.2`. I1 adds a task-mismatched
instruction contrast:

`L_instruction = max(0, 0.05 + L_correct/(sg(L_correct)+1e-6)
                     - L_wrong/(sg(L_correct)+1e-6))`.

The mismatched task is the next task in sorted model-train6 order, wrapping at
the end; it uses the same episode index. Correct and mismatched branches use
identical RGB target, action, timestep, and noise. The wrong branch is evaluated
sequentially. The contrast weight is `0.1`; activation outside `[25%,95%]` for
three consecutive train quanta closes I1.

I2 adds contact-weighted RGB supervision. Its mask is derived only from released
training RGB, codec-input robot-only raw float flow, and deterministic motion
residuals. For RGB transition `t -> t+1`, the motion residual at the source
pixel is `mean_c(abs(rgb[t+1]-rgb[t]))`; support is strictly greater than
`0.08`. Robot magnitude is
`m=clip(sqrt(dx^2+dy^2)/20,0,1)` and support is `m>0.05`.
Contact support is a 15-pixel RGB-space dilation of source robot support
intersected with the source-aligned motion-residual support; there is no
target-frame union. With temporal VAE stride four, latent index zero uses the
RGB-frame-zero masks; latent index `j>0` area-averages transition masks whose
source indices are `4j-4` through `4j-1`, clipped to `0..119`. Spatial masks use
area interpolation to the exact latent height and width. Token weights are
`1 + 2*robot + 4*contact`, normalized to sample mean one. No SAM, fresh40, or
evaluator detector output is used. Coverage outside `[1%,35%]` or empty contact
masks above 5% closes I2.

I1 uses `L_I1=L_RGB-FM+0.1*L_instruction`. I2 replaces unweighted RGB-FM with
`L_I2=sum(w*per_token_FM_error)/sum(w)`; it does not add a second contact loss.

### 5.3 Combined action-to-state arms C1 and C2

C1 uses rank-8 LoRA, learning rate `1e-6`, on RGB-stream self- and
cross-attention q/v plus `ffn.0` and `ffn.2`. Its objective is normalized
contact-weighted RGB-FM plus `0.25*L_x0 + 0.05*L_highfreq +
0.1*L_instruction`. C2 uses the V1 trainable whitelist and learning rate,
plus predicted-x0 flow-warp and static-background losses. C1 and C2 reference
the component proposal SHAs; no coefficient may be inherited by name alone.

For C2, adjacent latent indices `j -> j+1` use the four released robot-only RGB
forward flows whose source indices are `4j` through `4j+3`. They are composed
in pixel coordinates in chronological order by
`F = F + bilinear(next_flow, grid + F)`, using zero padding and
`align_corners=false`; out-of-bounds samples are removed from valid support.
The composed flow is used to sample the target predicted-x0 at
`grid + (2*dx/W_rgb, 2*dy/H_rgb)` with bilinear
`grid_sample`, zero padding, and `align_corners=false`, then compare it by
SmoothL1 to source predicted-x0 inside valid robot/contact support. This is
target-backward sampling under source forward flow, not forward splatting. A
fixed one-pixel translation test and a four-step constant-flow composition test
must each have max error below `1e-4`. Robot/contact masks use the exact I2
temporal mapping and mask-builder SHA. Outside a
15-pixel dilation of robot/contact support, static-anchor SmoothL1 compares
predicted-x0 at each frame with its same-frame released GT latent, weight
`0.05`. Warp weight is `0.1`. Valid warp support below 70% or a Perspectivity
point delta below `-0.005` closes C2.

No combined arm may introduce a third untested mechanism during the same round.
If both component arms fail n24, their combination is not promoted.

### 5.4 Lightweight temporal residual refiner R1

R1 is not another attempt to run the terminally failed full SeedVR2 path. It is
a post-generation RGB residual network over five-frame windows in RGB `[0,1]`
resized to `320x240`: input Conv3d `3->128`, kernel `(1,3,3)`, padding
`(0,1,1)`; eight residual blocks containing
depthwise `3x3x3` temporal-spatial convolution plus `1x1x1` pointwise
convolution, padding 1, and GELU; then output Conv3d `128->3`, kernel
`(1,3,3)`, padding `(0,1,1)`. Frame boundaries use reflection padding by two
frames; stride-one windows emit only the center residual, so every output frame
has one contribution and no overlap averaging. The residual is clipped
to `[-0.05,0.05]`, bilinearly resized to `640x480`, set to exactly zero for frame
0, and multiplied by `1-0.75*robot_contact_mask` in action support. Refined RGB
is exactly `clip(stage1_rgb + attenuated_clipped_residual,0,1)`.

R1 trains only on model-train6 Stage-1 parent-generated videos whose manifest
and per-video SHA are frozen. Its target is the released same-episode GT RGB.
Its loss is SmoothL1 between refined RGB and released GT RGB plus `0.05`
fixed-Laplacian loss between refined RGB and released GT RGB and
`0.1*mean(abs(r_t-r_(t-1)))`. It cannot change the 121-frame count. At training
and deployment, `robot_contact_mask` is computed from the request-deployable
robot-only FlowWAM renderer plus motion residuals from the already generated
video by the exact I2 mask-builder code and SHA; it uses no GT, evaluator output,
dev, selector-fit, or holdout video.

The refiner has its own smoke and memory gate. Modifying frame 0, changing video
geometry, exceeding 8 GiB, or failing to improve Image by `0.01` while any of
JEPA, Background, or Photometric falls by more than `0.005` closes it. Failure
does not trigger topology, resolution, RoPE, or full-SeedVR2 retries.

### 5.5 Control arm Q0

Q0 is the existing historical standard-FM q/v evidence unless a service GPU has
no ready generation, scoring, teacher-cache, receipt, or targeted work. A new Q0
run is diagnostic only, has a separate budget, and cannot promote or consume a
targeted-candidate slot.

## 6. Eight-GPU asynchronous allocation

There is no campaign-wide barrier. Each GPU UUID independently claims a unique
work key and proceeds through smoke, train, generation, and scoring receipts.

Initial soft allocation is seven targeted proposals plus one shared service
lane:

- GPU class V: two visual/temporal arms, V1 and V2;
- GPU class I: two instruction/contact arms, I1 and I2;
- GPU class C: two combined/trajectory arms, C1 and C2;
- GPU class R: one lightweight refiner arm, R1;
- GPU class E: generation and full15 scoring; CPU sidecars perform selector
  aggregation, queue maintenance, and receipt analysis. Q0 may use this GPU only
  when no service or targeted work is ready.

Logical classes are not permanently bound to physical indices. They are soft
reservations. A legal idle GPU pulls the highest-priority compatible ready work
and can borrow another lane at a job boundary. Running jobs are never signalled
to rebalance the queue.

Priority order:

1. external-task safety and stale-claim adjudication;
2. gate-critical full15 scoring;
3. candidate generation required for an open gate;
4. ready targeted training or teacher-cache work;
5. cheap generated-only scoring and receipt validation;
6. Q0 control.

Backlog allocation:

- 1-15 unscored videos: at least one evaluator GPU;
- 16-63: at least two evaluator GPUs;
- 64 or more: at least four evaluator GPUs;
- gate-critical scoring: at most six evaluator GPUs;
- remaining legal GPUs continue useful train or generation work.

Utilization is not maximized with duplicate work, unapproved experiments, or
recomputation that cannot change a gate.

Queue claim selection runs under one scheduler flock: read ready and active
counts, calculate the desired evaluator count from the backlog, apply
gate-critical priority, aging, and soft reservations, then atomically create one
claim. If no legal ready work exists, an idle GPU is allowed; duplicate work is
never created to satisfy utilization.

## 7. Proposal, work identity, and recovery

Every executable proposal is a canonical JSON document with contract
`flowwam-targeted15-proposal/2` and these exact top-level fields:

- `proposal_id`, `family`, `mechanism`, `target_metrics`,
  `protection_metrics`, `resource_profile`, `stage_plan`,
  `expected_receipts`, `train_manifest`, `selector_fit_manifest`,
  `dev_gate_manifest`, `final_holdout_manifest`, and `pins`;
- `mechanism` contains kind, attention scope, exact module whitelist, objective
  names and coefficients, sampler, teacher/mask/refiner identities, rank, LR,
  and seed;
- `pins` contains campaign, proposal, parent, config, code, data, split, seed,
  mechanism, metric-policy, and scorer-manifest SHA256 values.

Proposal JSON uses RFC 8785 canonicalization. Its hash domain is the complete
document with `pins.proposal_sha256` replaced by JSON `null`; the registered
document then stores the resulting lowercase hex digest in that field. A
validator repeats the null-replacement before hashing. This removes hash
self-reference while keeping the registered proposal immutable. Missing fields,
a non-null preimage value, or a free-form mechanism description fail closed.

Every work key binds:

- campaign and proposal IDs;
- parent, code, objective, config, dataset, split, task, episode, and seed SHA;
- resource profile and capability class;
- expected terminal receipt schema.

Physical GPU index and UUID are recorded in the claim and terminal receipt, not
in the logical work key. Therefore moving compatible work to another legal GPU
does not change work identity or create a duplicate.

Before launch, terminal receipt and active claim are checked twice. A complete
business receipt with a missing queue terminal is reconciled into the terminal
without rerunning work. Otherwise stale recovery requires that owner and bound
child PID, proc start ticks, PGID, user, command line, and cwd are all absent;
the exact UUID is free for two observations spanning at least ten minutes; the
flock is exclusively obtainable; and checkpoint/partial SHAs validate. Recovery
writes `signal_sent=false` and creates a missing-only work item referencing the
old work key and checkpoint SHA. It never deletes a partial or signals an
external or ambiguous process.

## 8. Smoke, memory, and training gates

Each mechanism first runs one production-shape step with the real 121-frame,
640x480 contract. It must prove:

- exact trainable whitelist;
- finite loss components and nonzero gradient on the intended adapter;
- peak memory below 22 GiB with no hidden multi-GPU/DDP process;
- immutable parent and dataset hashes;
- checkpoint reload and one deterministic inference smoke;
- no black frame or structural video error.

Expected smoke ranges are diagnostic, not claims: V1/I2 `16-18 GiB`, I1
`16-18 GiB` with roughly two-times compute, C1/C2 `17-19 GiB`, V2
`20-22 GiB`, and R1 below `8 GiB`. The observed allocated and reserved peaks are
authoritative.

Any source/split/hash mismatch, forbidden evaluation input, unexpected
trainable parameter, zero intended gradient, NaN/Inf, memory at least 22 GiB,
wrong sigma direction, checkpoint reload failure, or structural-video failure
closes the work item. Component-specific close conditions in Section 5 are
also mandatory and cannot be relaxed by the scheduler.

The step-1 smoke is an independent, non-promoting clone of the immutable parent;
its weights are never a formal-training input. Passing arms restart from the
parent at global step zero and train to the exact formal endpoints 5, 10, 15,
20, and 25. Each 5-step quantum has an independent work key bound to input
checkpoint SHA and target global step, and outputs model, optimizer, RNG,
dataloader cursor, and trainable-whitelist SHAs. Backlog rebalance occurs only
after a quantum or generation/scoring shard ends. More than 25 formal steps
requires a separate proposal derived from n24 evidence.

## 9. Evaluation and automatic iteration

Training telemetry and kill gates run at every formal endpoint, but round-one
dev generation occurs only from the step-25 checkpoint. The smoke performs one
deterministic structural inference and no full15 evaluation; it counts toward
the 1,600-new-video budget but supplies no promotion evidence. A step-25 video
enters generated-only nine-metric selection and then full15 scoring as soon as
structural validation passes. Arms do not wait for peers. Generation and scoring
use eight-video shards so service work remains borrowable across legal GPUs.

For every dev or holdout episode, a proposal generates both seed1 and seed4.
After both videos pass structure, the frozen generated-only nine-feature
selector and policy choose exactly one; only then may the selected video be read
by full15 and matched-GT correction. The matched P0 row is the same episode
chosen by the immutable P0 seed1/seed4 selector, never a fixed-seed shortcut.
No raw or corrected full15 value may choose a seed. A separately proposed new
selector is legal only if it was trained on selector-fit6 and frozen, with its
feature schema and policy receipts, before dev-gate2 is opened.

Thus n24, n64, and n128 mean 24, 64, and 128 selected paired rows and require
48, 128, and 256 raw proposal videos respectively; later gates reuse earlier
rows and add only missing episodes. The 800 full15 budget counts every actual
full15 video evaluation, including newly scored P0 baselines and selector-fit6
labels. The 1,600-new-video budget counts both raw seeds, smoke videos, and any
selector-fit generation.

Every round-one proposal uses `selector_mode="frozen_existing"` with model SHA
`4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa` and
policy SHA
`aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`.
Its required `selector_fit_manifest` is a canonical sentinel with exactly
`{"contract":"selector-fit-manifest/1","mode":"not_applicable",`
`"reason":"frozen_existing","rows":[]}`; the proposal pins that sentinel's
SHA. An empty object, missing manifest, or implicit selector default fails
registration. A new-selector proposal is a later, separate mechanism and must
reserve its generation and full15-label budget before registration.

Before any round-one proposal registers, the budget ledger atomically reserves
all of the following from the 1,600-video pool: 512 raw videos for candidate and
matched-P0 final holdout, 256 for the reusable matched-P0 dev baseline, 208 for
one n24 champion's cumulative n64/n128 increments, and 343 for seven independent
smokes plus seven dual-seed n24 gates, totaling 1,319. The remaining 281 may be
claimed only by explicit later proposals. From the 800-full15 pool, 256 is
non-borrowable final-holdout reserve; the reusable P0 row for a given episode is
scored once and referenced by SHA across proposals. Round-one dev P0, seven n24
arms, and one champion through n128 reserve another 400, leaving 144 for later
selector-fit or targeted work.

Family target aggregates are unweighted means of these normalized `[0,1]`
metrics:

- V1/V2: Image, Aesthetic, JEPA, Subject, Background;
- I1: Instruction, Interaction;
- I2: Interaction, Trajectory, Instruction;
- C1: Image, Aesthetic, JEPA, Subject, Background, Instruction, Interaction;
- C2: Trajectory, Interaction, Background, JEPA;
- R1: Image, Aesthetic, JEPA, Background, Subject, Photometric.

All non-target metrics are protection metrics. Deltas and thresholds below are
on the `[0,1]` scale; only EWMScore_P reporting multiplies a mean by 100. A
strict win uses corrected15 delta above `1e-9`; absolute deltas at or below that
epsilon are ties.

n24 survives only if:

- corrected15 uplift over matched P0 is at least `0.002`;
- its primary target aggregate is positive;
- no protection metric point estimate is below `-0.015`;
- all 24 rows are finite, structurally valid, and hash-bound.

n64 promotes only if:

- corrected15 uplift is at least `0.003`;
- nonmotion12 uplift is at least `0.004`;
- primary target aggregate uplift is at least `0.02`;
- corrected15 one-sided 90% task-stratified paired-bootstrap LCB is above zero;
- JEPA, Trajectory, Instruction, and Perspectivity are each at least `-0.005`.

n128 locks only if:

- corrected15 uplift is at least `0.004` with paired LCB95 above zero;
- nonmotion12 uplift is at least `0.005` with paired LCB95 above zero;
- strict wins are at least `76/128`;
- every protection metric LCB95 is above `-0.005`.

The paired bootstrap uses 10,000 replicates and RNG seed
`SHA256(proposal_sha256 + ":paired-bootstrap")`. Each replicate samples tasks
with replacement with equal task weight, then episodes with replacement inside
each sampled task. LCB90/95 are the one-sided 10th/5th percentiles. Dynamic,
Flow, and Motion use per-episode GT-corrected values and additionally report cap
hit rate and remaining headroom.

After every terminal n24/n64/n128 result, the analysis worker writes:

- per-metric paired deltas and confidence bounds;
- target and protection aggregates;
- task-stratified failure clusters;
- loss/gradient/memory evidence;
- one of `close`, `promote`, `combine`, or `new-proposal`.

`new-proposal` may alter only one mechanism or one preregistered coefficient
family at a time. It receives a new work key and cannot overwrite a failed arm.
At most two new targeted proposals are opened after the seven approved round-one
proposals. Across later rounds, at most six additional targeted proposals may be
opened. Global limits remain three rounds, 800 full15 development video
evaluations, 1,600 newly generated videos, or 384 GPU-hours, whichever closes
first. Q0 does not count as a candidate but has a maximum 25-step diagnostic
budget.

The campaign falls back to immutable P0 after three rounds without at least
`+0.003` corrected15, or after the same mechanism fails twice.

## 10. Final holdout and submission boundary

After dev-gate2, exactly one preregistered champion may open final-holdout4 once.
It evaluates 32 selected rows per task, 128 total, with the same dual-seed,
generated9-first protocol and the same n128 requirements: corrected15 uplift at
least `0.004` with LCB95 above zero, nonmotion12 uplift at least `0.005` with
LCB95 above zero, at least `76/128` strict wins, and every protection-metric
LCB95 above `-0.005`. The holdout cannot change losses, modules, coefficients,
prompts, seeds, selector features, or thresholds. Failure closes the campaign
and falls back to immutable P0; it cannot trigger tuning, a second champion, or
a second holdout opening.

Official test1000 is inference-only. It is never used for hyperparameter search,
per-episode winner labels, or another training round.

Selector fitting and inference are separate stages. Any new selector is fitted
only on selector-fit6, with the frozen generated-only nine-feature schema. Its
full15 outcomes may be fit labels only inside that split. Before dev-gate2 or
final-holdout4 is opened, feature schema, model, threshold, tie policy, code,
and split SHAs are frozen. fresh40 permits only inference by the existing frozen
selector; no selector fitting or calibration uses it. CPU sidecars perform
selector fit and aggregation.

Before proposal registration, a scorer-manifest receipt must pin code and
weight SHA256 for all 15 metrics, the GT-match manifest, and the metric-policy
SHA implementing
`min(candidate,matched_GT)` for Dynamic, Flow, and Motion. Required business
receipts are: proposal registration, data/split, teacher cache, smoke,
train quantum/checkpoint, generation shard, L0 structural validation, raw15,
matched-GT/corrected15, aggregate/gate, selector fit/infer/decision, queue
terminal, and stale adjudication. Queue success is legal only when the expected
business receipt exists and its contract, work key, inputs, outputs, and SHAs
match. Generation receipts include manifest, per-video SHA, 121 frames,
640x480, and black0.

Official test1000 has no local matched-GT contract. It permits frozen inference,
generated-only diagnostics, and structural validation only; no local full15 or
GT-cap claim may be produced.

The final ZIP contract remains:

- root entries exactly `README.md` and `HZ-World/`;
- `HZ-World/episode1.mp4` through `episode1000.mp4`;
- README fields Model Name, Organization, Responsible Person, Contact Email;
- submission recipient `worldarenav2@outlook.com`;
- subject `{model_name}_{final_version}_Track1_Submission`;
- no email is sent without explicit user confirmation.

## 11. Explicit exclusions

- old Wan clean50/winner or test-derived winner supervision;
- pseudo-winner training;
- old breadth20 and seed-sweep launchers;
- the old 3/3/2 pure q/v scope sweep;
- a fourth SeedVR2 repair;
- full-backbone finetuning without a mechanism-specific smoke;
- using GPU occupancy as a reason to run duplicate or non-decision work.
