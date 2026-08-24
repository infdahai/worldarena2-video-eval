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

The campaign succeeds only if a pre-registered candidate beats the immutable champion under comparable raw metrics while remaining sound under the latest GT-reference-corrected motion metrics.

## Non-goals

- No seed scan beyond seed1 and seed4.
- No prompt grid; action-to-text has exactly one deterministic template.
- No neural selector, end-to-end reward model, or clean50-trained router.
- No step100/200 continuation for v16 or v17.
- No cross-attention LoRA, full-model fine-tuning, new auxiliary loss, Motion FT, or new action architecture.
- No fourth SeedVR2 repair.
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

Step25 requires a production-shape smoke receipt, LoRA artifact, optimizer artifact, hashes, and a matched seed4 Breadth20 gate. It advances to step50 only if Instruction does not regress, black is zero, valid count does not fall, raw and corrected means improve, corrected motion average does not fall, and Trajectory/JEPA remain within `0.01` of baseline. Step50 applies the same gate and never continues to step100.

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

It uses the same production-shape smoke, whitelist, matched Breadth20, corrected-motion, and checkpoint receipt gates as score-directed self-training.

## Execution priority

1. **P0:** build the 200x2 seed corpus, complete raw/latest scoring, fit both logistic selectors, and apply the 40-row holdout decision.
2. **P1 in parallel:** run the one fixed action-to-text Breadth20 experiment without delaying P0 corpus generation or scoring.
3. **P2:** build the high-confidence pseudo-target manifest from only the 160 training rows and run score-directed self-training step25, then conditional step50.
4. **Stop:** choose the final eligible system and allocate at most one new clean50. Native-640 v17 remains inactive unless the dated contingency condition above was satisfied.

## Clean50 allocation

The selector uses existing paired seed1/seed4 clean50 videos and creates no new clean50 generation.

Among action-to-text, score-directed self-training, and native-640 v17, at most one candidate may receive a new clean50 generation and full evaluation. That candidate is selected from zero-overlap holdout/Breadth20 evidence before clean50 starts. A candidate that fails its earlier gate cannot consume the clean50 allocation.

The same corrected-first final gate applies to that one new clean50 candidate: corrected mean improvement greater than `0.003`, 12-metric non-motion mean no worse than `-0.001`, Trajectory and JEPA individually no worse than `-0.01`, and raw 15-metric mean no worse than `-0.003`, all relative to the immutable Official+seed4 champion scored by the same receipt-pinned pipeline.

## GPU and process isolation

- Physical GPU7 is forbidden.
- GPU0-GPU6 may be used only after read-only PID ownership and memory checks show that the target GPU is free of unrelated work.
- Existing unrelated processes always take precedence, even if the GPU was previously authorized.
- Every GPU job owns a per-GPU lock and records physical GPU identity in its receipt.
- Episode shards are disjoint, manifest-derived, and safe to resume without duplicate generation.
- Diagnostic v16 prompt-triplet jobs are stopped after their current atomic episode because their parent self-only branch already failed.
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
- `selectors/selection.complete.json`;
- `action-text/quality-gate.complete.json`;
- `self-training/train-step25.complete.json` and its quality gate;
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
