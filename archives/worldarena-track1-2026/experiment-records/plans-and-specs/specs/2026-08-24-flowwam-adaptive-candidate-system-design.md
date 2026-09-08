# FlowWAM Adaptive Candidate System Design

## Status and decision

This specification is the approved design for the final WorldArena 2.0 Track 1 campaign after the bounded SeedVR2, v16 step100, v16 inference-scale, and v16 self-attention-only experiments failed their gates.

The immutable rollback champion remains:

- model: Official FlowWAM Stage-1
- `flow_max_magnitude`: `20`
- inference seed: `4`
- dev-clean50 raw 15-metric arithmetic mean: `0.6586597567502485`

No experiment may overwrite, relabel, or weaken the validation evidence for that champion.

## Goal

Use variance already present in Official FlowWAM to build an original, receipt-gated candidate-selection layer, then reuse its zero-overlap training-partition winner data for one bounded score-directed self-training experiment. In parallel, test one deterministic action-to-text condition. Native-640 self-attention-only LoRA is a dated contingency, not part of the normal August 30 critical path.

The frozen P0 post-generation selector has already satisfied that campaign-success condition on clean50 and is the current formal champion. Any later P2 replacement must beat P0 under the stricter receipt-gated Breadth20 and clean50 contracts below; Official Stage-1 + flow20 + seed4 remains only the immutable rollback package.

## Non-goals

- No seed scan beyond seed1 and seed4.
- No prompt grid; action-to-text has exactly one deterministic template.
- No neural selector, end-to-end reward model, or clean50-trained router.
- No step100/200 continuation for v16 or v17.
- No cross-attention LoRA, full-model fine-tuning, Motion FT, or new action generation architecture. The only new objective authorized by the 2026-08-24 MoE extension is one bounded pairwise preference LoRA at step25.
- No fourth single-GPU SeedVR2 repair. The separately authorized two-GPU official sequence-parallel topology is governed by the terminal evidence below and may not be retried after its full-resolution and sole fallback failures.
- No Difficulty/OOD-adjusted EWMScore-P claim unless its public implementation becomes completely reproducible.

## Evaluation and leakage boundary

### Dataset partitions

The selector dataset contains exactly 200 episodes sampled from the existing 1,785-row training universe after the established dev-fast20 and dev-clean50 exclusions are applied.

Before any candidate generation, the pipeline writes a sorted manifest with:

- exactly 200 unique episode IDs;
- source paths and SHA256 hashes;
- task-stratum labels;
- a deterministic `160 train / 40 holdout` split;
- zero collision with dev-fast20 and dev-clean50;
- the selection algorithm and random seed;
- a manifest SHA256 receipt.

Selection is deterministic. Eligible rows are grouped by task, each task receives a Hamilton-apportioned share of 200 proportional to its eligible row count, and rows inside a task are ordered by `SHA256("selector-v1:" + episode_id)`. The 40 holdout slots use the same proportional apportionment inside the selected task counts and order rows by `SHA256("selector-holdout-v1:" + episode_id)`. Ties are resolved by the canonical episode ID. The manifest records the literal salts and allocation table.

The 40-row holdout is hidden from model fitting, feature normalization, hyperparameter selection, template changes, and threshold selection.

Dev-clean50 is opened once for the selector chosen from the 40-row holdout. Its result is terminal evidence, not feedback for another selector iteration.

### Scoring contract

Every generated candidate stores two aligned per-episode score views:

1. the pinned raw 15 official metric values and their arithmetic mean;
2. the latest GT-reference-corrected view, where only Dynamic Degree, Flow Score, and Motion Smoothness are capped by the matched GT reference result.

The scorer must fail closed on missing episode IDs, duplicate rows, missing GT matches, non-finite values, unexpected row counts, or hashes that do not match the receipt.

The corrected 15-metric mean is the primary selector label. The raw mean remains the comparable champion/submission gate.

## System architecture

### 1. Candidate corpus builder

For each of the 200 selected episodes, generate two videos with the same immutable Official Stage-1 checkpoint and identical conditions except for seed:

- candidate A: seed1;
- candidate B: seed4;
- `flow_max_magnitude=20`;
- 121 frames;
- 640x480;
- one MP4 per episode/candidate;
- black-frame count zero.

The two lineages are isolated and never overwrite existing dev-clean50 seed1/seed4 outputs. A generation receipt records the parent checkpoint hash, manifest hash, seed, encoder settings, per-video SHA256, frame count, dimensions, black-frame result, and GPU identity.

The winner-label table records raw and corrected means for A and B, the signed deltas, the winning seed, and an absolute-margin sample weight. Exact corrected ties are assigned seed4 deterministically and retained with zero sample weight so corpus accounting remains complete without influencing fitting.

### 2. Input-only action router

The input router predicts seed1 or seed4 before generation. Its features are available before inference:

- the immutable Official Stage-1 T5 encoder's mean-pooled non-padding-token instruction embedding, reduced to at most 32 components by PCA fitted only on the 160 training rows;
- left/right path length;
- left/right displacement and velocity summaries;
- dominant arm;
- fraction of frames with both arms active;
- left/right gripper close and open transition counts;
- motion phase count;
- action-flow magnitude mean, standard deviation, maximum, and active-frame ratio.

The model is a scikit-learn pipeline containing missing-value rejection, training-only PCA for the instruction embedding, standard scaling, and `LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=20260824)`. The only regularization choices are `C in {0.1, 1.0, 10.0}`, selected by `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260824)` on the 160-row training split. No other classifier is introduced during this campaign. The receipt pins the T5/model hash and serialized pipeline hash.

### 3. Post-generation selector

The post-generation selector uses the same input features plus order-aware A-minus-B and absolute-difference features for:

- Instruction Following;
- Interaction Quality;
- Perspectivity;
- Image Quality;
- Aesthetic Quality;
- Photometric Consistency;
- SAM3 trajectory validity and coverage;
- `action_projection_DTW`;
- candidate motion statistics.

Every post-generation feature must be computable from the request inputs and generated candidate alone. GT video, GT-only JEPA similarity, Depth Accuracy, official Trajectory Accuracy DTW, or any other feature unavailable in the deployed service is forbidden.

`action_projection_DTW` is a separate deployment feature with this locked definition:

1. read the request's 3D left/right end-effector action trajectory and its camera intrinsics/extrinsics;
2. transform each 3D point into camera coordinates using only that request calibration;
3. project visible points into normalized 2D image coordinates;
4. align the projected action path to the 121 generated-video frame timestamps without reading a GT video;
5. match left/right gripper identity to the SAM3-detected generated-video trajectories;
6. calculate normalized 2D DTW and emit validity/missing-point diagnostics.

A preflight checks that the required 3D trajectory, coordinate convention, intrinsics, extrinsics, timestamps, and arm identity are present for every selector-corpus request and in the deployment request contract. If any required field is unavailable, `action_projection_DTW` is removed from the feature schema before either selector is fitted; it is never imputed from GT or enabled only for a subset.

It uses the same logistic-regression pipeline and training-only regularization selection. At deployment it requires both seed candidates and therefore has higher latency and GPU cost than the input router.

### 3.1 Action-Flow Critic extension

Before fitting, the post-generation feature schema is extended with request-deployable action-to-video consistency features. Desired flow is the immutable Official FlowWAM robot-only renderer output for the request; generated flow is extracted from each generated candidate with the pinned RAFT implementation at the same geometry. Per candidate the critic emits robot-support cosine similarity, normalized endpoint error, direction hit rate, magnitude ratio, temporal phase correlation, active-arm agreement, inactive-arm leakage, flow energy outside expected support, renderer-mask versus generated SAM3 robot-mask IoU, left/right arm centroid separation, dominant-arm visibility, and `action_projection_DTW`. When a deployment-available SAM3 object track is valid it also emits object visibility, gripper/object distance after grasp, motion correlation after grasp, and object persistence after release.

Every feature records a validity flag and literal source contract. A feature unavailable for any corpus request or unavailable in deployment is removed globally before cross-validation; it is never filled from GT or enabled only for holdout. The critic is a feature extractor, not a neural reward model, and remains inside the same Logistic Regression selector.

### 3.2 Confidence cascade

The final selector is a frozen three-way cascade:

1. if input-router confidence is at least `tau_high`, generate only its predicted seed;
2. otherwise generate seed1 and seed4 and apply the post-generation selector with Action-Flow Critic features;
3. if post-generation confidence is below `tau_post`, return immutable seed4.

`tau_high` and `tau_post` are selected only from deterministic out-of-fold predictions on the 160 training rows. The threshold grid is exactly `{0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}`. The selected pair maximizes corrected out-of-fold uplift, then minimizes expected generations per request, then chooses the larger thresholds. The 40-row holdout may not change features, thresholds, models, or fallback behavior.

### 4. Selector decision gate

Both selectors are frozen before the 40-row holdout is read. The holdout report includes classification accuracy, confusion matrix, selected corrected/raw/non-motion score, seed4 corrected/raw/non-motion score, mean uplift, task-stratified paired bootstrap interval, and chosen-seed counts. Accuracy is diagnostic only and is not an advancement gate.

The paired bootstrap uses 10,000 deterministic replicates with random seed `20260824`. Each replicate samples the holdout task strata with replacement; for every sampled task it then samples that task's paired per-episode selected-minus-seed4 corrected-score deltas with replacement to the task's original holdout count. The replicate statistic is the episode-weighted mean across the sampled task clusters. This task-cluster step preserves uncertainty when a task has only one holdout episode. The one-sided 90% lower confidence bound is the 10th percentile of the replicate means.

A selector passes only if:

- all 40 rows are valid;
- black-frame count is zero for every underlying candidate;
- corrected mean uplift over seed4 is at least `0.003`;
- the one-sided 90% task-stratified paired-bootstrap lower bound is greater than zero;
- raw mean is at least the seed4 raw mean;
- the 12-metric non-motion mean is at least the seed4 non-motion mean;
- seed1 and seed4 are each selected at least `5/40` times.

If both pass and the input router's corrected uplift is within `0.001` of the post-generation selector, choose the input router. Otherwise choose the passing selector with the larger corrected holdout uplift. The choice is written before dev-clean50 is opened.

The chosen selector is evaluated once on the existing paired seed1/seed4 dev-clean50 artifacts. It becomes the final system only if:

- the selected corrected 15-metric mean exceeds the corrected Official+seed4 mean by more than `0.003`;
- the selected 12-metric non-motion mean is at least the champion non-motion mean minus `0.001`;
- Trajectory Accuracy and JEPA Similarity are each at least the corresponding champion metric minus `0.01`;
- the selected raw 15-metric mean is at least the champion raw mean minus `0.003`, as a no-catastrophic-regression check rather than the primary ranking objective;
- all 50 rows and all 15 metrics are finite;
- no per-video hash or lineage mismatch exists.

If no selector passes, the branch is permanently closed without fitting against clean50.

## Deterministic action-to-text experiment

Action-to-text converts the action sequence into one structured suffix containing only facts derived from the action:

- dominant and stationary arm;
- which arm moves first;
- gripper close/open ordering;
- coarse approach, grasp, transport, and release phases.

The template is fixed before generation, uses no language model, and must not mention object names absent from the original instruction. Exactly one Official Stage-1 + seed4 Breadth20 lineage is generated.

It passes only if:

- black-frame count is zero and valid count does not fall;
- Instruction Following and Interaction Quality both improve;
- raw and corrected fast means improve;
- Trajectory Accuracy and JEPA do not fall by more than `0.01` each;
- corrected motion-metric average does not fall.

There is no prompt rewrite, alternative wording, or second template after failure.

## Zero-generation-cost diagnostics and post-processing

### Flow expert oracle

Before any new flow generation, use only existing matched Breadth20 flow16, flow20, and flow24 artifacts. Recompute a per-episode corrected-score oracle with strict common-episode and receipt/hash validation. The branch closes without generation unless oracle uplift over flow20 is at least `0.004` and non-flow20 wins at least `6/20`. A passing oracle authorizes one input-only flow router trained on deployment-available action summaries and the same 160/40 leakage boundary; it does not authorize a new flow scan.

### Action-Support Background Projection

The original branch is terminally closed because its protected-support assets did not exist. Background-v2 is one independently receipt-gated successor, not a retry with relaxed prerequisites. Apply it once to the frozen P0-selected Breadth20 videos. For frame `t`, define protected dynamic support as the union of robot-only renderer support, expected flow support, pinned generated-RAFT active-flow support, and valid generated-video SAM3 robot/object support. Dilate by one recorded gripper-width estimate and feather the boundary. Inside support preserve the generated frame; outside support use `0.3 * X_t + 0.7 * I_0`, followed by the single frozen light temporal denoise and light unsharp operation. No ratio, dilation, threshold, denoise, or sharpening grid is allowed.

It advances only if corrected 15-metric Breadth20 mean improves over P0 by at least `0.005` and the arithmetic mean of Background Consistency, Photometric Consistency, Subject Consistency, Image Quality, and JEPA also improves by at least `0.005`; Trajectory, Instruction, and Interaction may not regress; black remains zero; and a deterministic seam diagnostic passes. A failed gate closes the post-process without a parameter search.

## Frozen ExpertPool-v2

P0 remains immutable. Winner-SFT step25 seed4 at LoRA scale `0.5` is the first Challenger because its receipt-gated Breadth20 result improves corrected mean over P0 by `0.006361`. The only additional generated expert is the already fixed ActionText template. Preference-LoRA may appear in zero-cost oracle evidence but is excluded from bulk generation. There is no new model, seed, prompt, expert, or selector architecture.

The gate is conservative and may abstain to P0. It is fitted only on the original selector160 training rows. A newly sampled, zero-overlap fresh40 is frozen before any score is opened and may not modify features, thresholds, classifier, templates, or the expert set. Fresh40 uplift tiers are `<0.004` close, `[0.004,0.006)` positive but retain P0-only complexity, `[0.006,0.008)` one-clean50 eligible, `[0.008,0.012]` strong main-line evidence, and `>0.012` immediate research freeze. Every advancing tier also requires deterministic one-sided bootstrap lower bound `>0`, raw and non-motion no regression, no structural Trajectory/JEPA regression, finite rows, black0, and exact hashes.

All research and policy selection ends at `2026-08-26T20:00:00+08:00`. At that point branches are receipt-closed and formal test-1000 begins. Test episodes are sharded; Official seed1, Official seed4, and only when eligible Winner scale0.5 are generated adjacently, scored and selected per completed shard. Formal test data and outputs are inference-only and never feed training or selection design.

## Score-directed self-training

Only the 160-row selector training partition may contribute pseudo-targets. The 40-row selector holdout remains excluded from feature fitting and LoRA training permanently. No second rejection-sampling generation is allowed.

The pseudo-target manifest keeps a training-row winner only when:

- the raw winner and corrected winner identify the same seed;
- the corrected winner margin is at least `0.003`;
- the row is not an exact tie.

The retained set is balanced to equal seed1/seed4 counts by deterministic downsampling of the larger winner class. Within each winner class, task quotas are Hamilton-apportioned from the eligible task counts; rows are ordered by descending corrected margin and then canonical episode ID. If fewer than 40 total examples or fewer than 10 examples for either winner seed survive, score-directed self-training is closed as under-supported. Every retained pseudo-target receives ordinary unit RGB-diffusion loss; holdout-derived weights and low-margin rows are never passed to the trainer.

The training contract is:

- parent: immutable Official FlowWAM Stage-1;
- target video: the high-confidence, raw/corrected-consistent training-partition winner selected by the pseudo-target manifest;
- reference/action inputs: the original matched episode inputs;
- trainable parameters: DiT self-attention q/v LoRA only;
- rank/alpha: `8/8`;
- learning rate: `1e-6`;
- batch size: `1`;
- gradient checkpointing: enabled;
- loss: official RGB diffusion objective only;
- first stop: step25;
- maximum stop: step50.

The trainable whitelist rejects cross-attention, K/O, MLP, FlowStream, T5, VAE, modulation, and all native weights.

Step25 requires a production-shape smoke receipt, LoRA artifact, optimizer artifact, hashes, and one paired seed1/seed4 Breadth20 gate. The two seeds are evaluated under the same new checkpoint and frozen inputs; each global seed is compared directly with the frozen `P0-selector-Breadth20` baseline, and only a P2 seed satisfying the P0-relative promotion gate below may advance. This is not a broader seed scan, and neither the Official parent nor the new checkpoint may use seed5+. Step25 advances to step50 only if it first passes the P0-relative promotion gate, corrected uplift over P0 is at least `0.006`, and Instruction, Interaction, and Image do not regress relative to P0. Step50 repeats the same paired seed1/seed4 gate and never continues to step100.

### Pairwise preference LoRA

After the high-confidence pseudo-target manifest exists, one independent step25 preference branch is allowed. It uses only the 160 training rows where raw and corrected winners agree, corrected margin is at least `0.003`, and the row is not tied. Winner and loser use the same episode, timestep, and noise. The frozen Official parent supplies reference loss differences; the trainable model is self-attention q/v LoRA rank/alpha `8/8`, learning rate `1e-6`.

For winner `+` and loser `-`, define `z = (loss_theta_minus - loss_theta_plus) - (loss_0_minus - loss_0_plus)` and optimize `-w * log(sigmoid(z)) + 0.1 * loss_theta_plus`, where `w` is the normalized corrected margin. No coefficient, block, rank, scale, or learning-rate grid is allowed. The branch stops at step25 and faces the same production-shape smoke, trainable whitelist, artifact/hash, and paired seed1/seed4 Breadth20 gate as Winner-SFT. Winner-SFT and Preference-LoRA global seeds are compared once against `P0-selector-Breadth20`; at most one P2 candidate that clearly passes may consume the new clean50 allocation.

## SeedVR2 official SP2 terminal evidence

The two-GPU official sequence-parallel experiment has already been executed on physical GPU4+GPU5 with `sp_size=2`, seed666, sample_steps1, alpha0.7, and the immutable seed4 input. It is terminally closed by:

`/data/di/worldarena2_track1_20260815/runs/flowwam-official/automation/seedvr2-official-sp2-stopped.complete.json`

The 720x1280 smoke failed in official sequence all-to-all while allocating 400 MiB. The sole contract-authorized 640x480 fallback failed in official `get_axial_freqs` while allocating 7.88 GiB. Neither produced a smoke receipt. The stop receipt decision is `stop_seedvr2_no_more_attempts`; no additional GPU topology, RoPE patch, resolution, or compatibility attempt is allowed in this campaign.

## Contingency native-640 v17 route

The independent native-640 route is excluded from the normal August 30 campaign. Its existing resumable CPU/network download may finish as contingency preparation, but no extraction, GPU smoke, training, generation, or evaluation may start unless both the selector P0 holdout gate and the action-to-text P1 Breadth20 gate have terminal failure receipts by `2026-08-26T23:59:59+08:00`.

If and only if that dated contingency unlocks, its source is the pinned `YixiangChen/FlowWAM_WorldArena` revision with independent `640/` RGB supervision and matched `320/` reference input.

Its frozen contract is:

- parent: immutable Official Stage-1;
- native 640x480 supervision, never 320-to-640 interpolation;
- self-attention q/v LoRA only;
- rank/alpha `8/8`;
- learning rate `1e-6`;
- official RGB objective only;
- step25 and conditional step50 only.

It uses the same production-shape smoke, whitelist, paired seed1/seed4 Breadth20, corrected-motion, and checkpoint receipt gates as score-directed self-training.

## Execution priority

1. **P0:** build the 200x2 seed corpus, complete raw/latest scoring, extract Action-Flow Critic features, fit input/post-generation selectors, freeze the confidence cascade, and apply the 40-row holdout decision.
2. **P0 zero-cost in parallel:** compute the existing flow16/20/24 oracle and run the single fixed Action-Support Background Projection on existing seed4 Breadth20.
3. **P1 in parallel:** run the one fixed action-to-text Breadth20 without delaying P0. SeedVR2 SP2 is already terminally closed and consumes no GPU.
4. **P2:** build the high-confidence pseudo-target manifest from only the 160 training rows, run Winner-SFT step25 and Preference-LoRA step25 independently, require a paired seed1/seed4 Breadth20 decision for every new checkpoint, compare once, and allow only the passing winner to consider step50 where its own contract permits it.
5. **Stop:** choose the final eligible system and allocate at most one new clean50. Native-640 v17 remains inactive unless the dated contingency condition above was satisfied.

## Clean50 allocation

The selector uses existing paired seed1/seed4 clean50 videos and creates no new clean50 generation.

The selector's passing one-time clean50 receipt promotes the frozen P0 post-generation selector to the current formal champion. Official Stage-1 + flow20 + seed4 remains the immutable rollback champion, not the comparison baseline for later P2 promotion.

Before any P2 clean50 allocation, replay the frozen P0 model/policy on the exact Official seed1/seed4 Breadth20 pair and receipt-gate `P0-selector-Breadth20`. Each Winner-SFT or Preference-LoRA global seed is compared directly against this P0 baseline. It advances only with corrected mean `>= P0+0.003`, raw mean `>= P0`, 12-metric non-motion `>= P0+0.002`, Trajectory and JEPA each `>= P0-0.005`, at least12/20 strict paired corrected wins, and black0. It is forbidden to train a new P2 selector, choose P2 seeds per episode, or reuse the opened40 holdout. Winner-SFT step50 additionally requires the step25 P0 gate to pass, corrected uplift `>=0.006`, and no Instruction, Interaction, or Image regression relative to P0.

Among the adaptive cascade, a passing flow router, Action-Support Background Projection, action-to-text, Winner-SFT, Preference-LoRA, and native-640 v17, at most one candidate may receive a new clean50 generation and full evaluation. That candidate is selected from zero-overlap holdout/Breadth20 evidence before clean50 starts. A candidate that fails its earlier gate cannot consume the clean50 allocation.

The same corrected-first final gate applies to that one new clean50 candidate: corrected mean improvement greater than `0.003`, 12-metric non-motion mean no worse than `-0.001`, Trajectory and JEPA individually no worse than `-0.01`, and raw 15-metric mean no worse than `-0.003`, all relative to the current P0 selector champion scored by the same receipt-pinned pipeline. Official+seed4 remains the immutable rollback package.

## GPU and process isolation

- Physical GPU7 is forbidden.
- GPU0-GPU6 may be used only after read-only PID ownership and memory checks show that the target GPU is free of unrelated work.
- Existing unrelated processes always take precedence, even if the GPU was previously authorized.
- Every GPU job owns a per-GPU lock and records physical GPU identity in its receipt.
- Episode shards are disjoint, manifest-derived, and safe to resume without duplicate generation.
- With the observed free set GPU0/GPU4/GPU5/GPU6, the 200 episodes are assigned as four deterministic 50-episode shards. Each shard generates both seeds on the same physical GPU, in the same runner, and in adjacent seed1-then-seed4 order, for 100 videos per GPU. A changed free set triggers a new recorded allocation receipt rather than implicit remapping.
- Dataset download, CPU feature extraction, and GPU generation may run concurrently when their files and locks are independent.

## Receipts and lineage

All new remote state lives under:

`/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/`

Required terminal receipts are:

- `manifests/selector-200.complete.json`;
- `candidates/seed1/generation.complete.json`;
- `candidates/seed4/generation.complete.json`;
- `scores/latest-wa2.complete.json`;
- `selectors/input-router.complete.json`;
- `selectors/postgen-selector.complete.json`;
- `selectors/action-flow-critic.complete.json`;
- `selectors/cascade.complete.json`;
- `selectors/selection.complete.json`;
- `flow-oracle/flow16-20-24.complete.json`;
- `background-projection/quality-gate.complete.json`;
- `action-text/quality-gate.complete.json`;
- `self-training/train-step25.complete.json` and its quality gate;
- `preference-training/train-step25.complete.json` and its quality gate;
- conditional self-training step50 receipts;
- native-640 download/setup/smoke/training/gate receipts;
- `final-decision.complete.json`.

No PID, log line, CSV, or checkpoint alone constitutes completion.

## Failure handling

- A generation shard may retry only missing or hash-invalid episodes.
- A scorer failure may be repaired only with a minimal evidence-backed compatibility/runtime change; validation may not be relaxed.
- A selector that fails holdout is closed; no feature or threshold revision may use holdout or clean50 feedback.
- A training branch that fails a quality gate stops at the best earlier checkpoint.
- Three distinct failed repair hypotheses close the affected runtime branch.
- The rollback champion remains deployable throughout.

## Verification

Implementation is test-first. Tests must cover:

- deterministic partitioning and zero evaluation overlap;
- paired candidate lineage and hash validation;
- latest scorer raw/corrected preservation and GT mismatch rejection;
- action-feature calculations and deterministic action-to-text output;
- training-only feature fitting and holdout isolation;
- logistic model serialization and exact inference reproduction;
- deterministic task-cluster paired bootstrap and selector gate behavior at every threshold;
- permanent exclusion of the 40 holdout rows from the pseudo-target manifest;
- raw/corrected winner agreement, margin filtering, and seed-balanced pseudo-target selection;
- `action_projection_DTW` calibration preflight and schema-wide removal when deployment inputs are incomplete;
- corrected-first clean50 gate and its non-motion, Trajectory, JEPA, and raw-regression guards;
- trainable-parameter whitelist and step25/50 continuation gates;
- GPU lock and physical-device validation;
- atomic receipt writing and resume behavior.

Before final completion, the three campaign reports are updated and re-read against the decisive remote receipts:

- `reports/2026-08-24-wan-action-v1-v15-total-experiment-report.md`;
- `reports/2026-08-20-v15-flowwam-balanced-campaign.md`;
- `reports/2026-08-23-worldarena-track1-final-score-report.md`.

The deployment gate requires the returned/package MP4 SHA256 to equal the generator receipt SHA256 so no service-layer transcode can silently change the evaluated video.
