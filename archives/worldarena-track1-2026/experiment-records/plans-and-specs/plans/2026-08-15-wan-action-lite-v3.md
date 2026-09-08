# Wan-Action-Lite+ v3 implementation plan

Spec authority: the user-approved Wan-Action-Lite+ v3 design and
`docs/superpowers/specs/2026-08-15-wan-action-lite-v3-adaptive-compression-design.md`.
WorldArena evaluator authority is commit
`7b3feee108427bee3380064bb5154970ed7468b5`.

## Global constraints

- Keep V-JEPA, Depth, DMD, GAN, object masks, trajectory auxiliary heads, and
  randomized-500 out of scope.
- Never touch unrelated GPU processes. Training and cache launchers wait for all
  eight RTX 4090 devices to be free.
- Preserve existing v2 caches; v3 uses schema-versioned, separate cache roots.
- Keep every persistent artifact under `/data/di/worldarena2_track1_20260815`
  (models, datasets, caches, manifests, statistics, replay, checkpoints, logs,
  evaluations, videos, and submission archives). `/home/huazhi/nlh` is
  source/config only; temporary build files are not persistent outputs.
- External reuse is deny-by-default. The only user-authorized exception is an
  exact-name lookup for currently missing Wan/RoboTwin assets under `/data/fjy`.
  Reuse requires the fixed upstream revision, exact byte size, and content hash
  to match before an atomic copy into `/data/di/worldarena2_track1_20260815`;
  training never reads an external path directly. Do not use hard links or
  symlinks, and do not inspect unrelated external contents. `/home/fjy` remains
  out of scope unless the user explicitly authorizes a new lookup.
- Render model conditions directly at 80x60 and preserve 640x480 only for audit.
- Use TDD for every behavior change and run remote PyTorch tests before training.

## Task 0: trusted reuse and segmented input readiness

Do not make the complete 34.2 GB Wan snapshot or all 50 clean archives a single
Stage-1 blocking condition. Validate and record independently usable components:

- `wan_backbone`: `config.json`, the safetensors index, and the three DiT shards;
- `wan_vae`: `Wan2.2_VAE.pth`;
- `wan_text`: the UMT5 checkpoint plus its pinned tokenizer files;
- `stage1_tasks`: the four deterministic train tasks and all 200 episodes;
- `train40`: all 40 deterministic train tasks used by pose statistics;
- `clean50_full`: all train/dev/blind tasks, required for final development and
  blind evaluation but not for the first Stage-1 optimizer step.

For every missing component, first perform the narrowly authorized `/data/fjy`
exact-name lookup. Copy only independently verified matching files into a
temporary path below the formal artifact root, fsync, re-check the destination,
and atomically rename it into place. If no valid source exists, download only the
currently blocking component with one worker; keep the complete snapshot download
as a resumable background activity. A partial receipt must never masquerade as the
full `.worldarena_manifest.json`.

The Stage-1 readiness dependency is:

```text
wan_backbone
  + wan_vae -> four-task latent cache
  + wan_text -> four-task text cache
  + stage1_tasks -> action/cache inputs
  + train40 -> schema-v3 pose statistics
  + eight free GPUs -> 8-rank smoke -> Stage-1 prefix/A/B
```

`clean50_full`, dev-clean-50 generation, native-Wan no-regression calibration,
and official test generation continue behind separate gates and cannot change a
checkpoint already selected from the frozen development protocol.

Operational audit on 2026-08-15 found that the formal artifact root already has
all three DiT shards and the VAE. `/data/fjy` has the same three DiT shards but
does not contain the missing UMT5 file, the six missing train-40 archives, or
their extracted task directories. Therefore no external copy is currently
needed: prioritize the remaining UMT5 bytes and the six train-40 archives, then
start the staged caches and training without waiting for the final clean-50
archive.

## Task 1: v3 geometry, raster, pose, and cache contract

Implement per-arm five-channel 80x60 raster conditions, explicit 21-step token
support, separate 21x30x40 loss weights, camera-frame normalized 11D pose, and a
versioned train-40 statistics artifact. Add anchored zero/reverse/swap SE(3)
counterfactual builders. Add FK/endpose, projection, schema, shape, and
counterfactual tests.

## Task 2: causal low-resolution adapter

Replace the high-resolution Conv3d encoder with a frame-wise 2D shared arm
encoder and explicit frame-0/causal-four-frame temporal packing. Keep independent
left/right raster projections and add independent left/right pose projections at
blocks 0/8/16/24. Add action-present conditioning and tests for impulse alignment,
null-vs-hold, zero-init equivalence, gradients, and residual norms.

## Task 3: weighted-FM and official evaluator parity

Use cached per-sample 21x30x40 loss weights with one normalization in weighted-FM.
Vendor the official detector/scorer semantics from the pinned commit, retaining
FastDTW and both missing-frame repair stages. Report official mean inverse distance,
raw mean/median distance, paired win rate, detection coverage, and action-adherence
DTW. Add fixture parity and four-condition Stage-1 gate tests.

## Task 4: deterministic common-prefix Stage-1

Implement replay manifests for sample index, timestep/noise seeds, and dropout
flags. Train raster-only through step 50, fork A/B from the same checkpoint, and
continue both through step 125 with B's zero-initialized pose branch as the sole
difference. This is an aggregate ceiling of 200 optimizer steps: 50 shared plus
75 for each branch. Run both at dev-fast-20 and advance only the winner to
dev-clean-50. Preserve resumability and add launcher/checkpoint compatibility
tests. Extra branch steps are fail-closed outside the default launcher and require
an explicit inconclusive or trade-off decision.

## Task 5: three-pass CFG and conservative Stage-2 LoRA

Implement null-text/null-action, text/null-action, and text/action sampling with a
global action scale in {1.0, 1.25, 1.5, 2.0}. Limit rank-16 alpha-16 LoRA to
self-attention q/v in blocks 8-29, use a 50-step warmup, default to 400 steps,
save every 100, and record adapter/LoRA update and per-layer residual norms. An
explicit continuation to 600 is allowed only after the step-400 trajectory,
no-regression, and plateau gates pass. Step 800 is not a default option. Add
inference, LoRA target-selection, and extension-policy tests.

## Task 6: integration, remote gates, and launch

Run the complete local and remote suites, generate 100 real-episode geometry gates,
and run an 81x480x640 eight-GPU production-shape memory gate. The gate uses the
exact Stage-1 FSDP/BF16/action configuration with micro-batch one and executes
three consecutive `forward -> backward -> optimizer.step -> zero_grad` iterations.
It records per-rank allocated/reserved peaks and step durations; every peak must
remain below 22 GiB. T5 and VAE module loading is forbidden because Stage-1/2
must consume only cached latents and text contexts. Verify downloads and disk
reserves, sync reviewed files, and start only the v3 Stage-1 launcher.
The launcher consumes the Task-0 component receipts rather than the full-snapshot
marker, remains resumable, and must not start while any GPU is occupied. The
background full-snapshot downloader must use at most one worker and must not write
the formal full manifest until every pinned asset has been verified.

The FK/HDF5 geometry gate uses frame 0 from each episode as the strict mapping
reference and preserves the original maximum thresholds of 10 mm and 2 degrees.
RoboTwin records drive targets under `joint_action` but derives `endpose` from the
physics link pose, so all-frame target-versus-observation maxima are reported as
tracking diagnostics and do not redefine URDF mapping correctness. Projection,
positive-depth, and no-edge-clamp checks still cover the full trajectory.

## Task 7: official Track-1 evaluation and submission contract

### 正式评测策略

评测分为开发评测与最终官方评测。

快速评测在 checkpoint 100/200/300/400 上运行，以 trajectory 为主。完整
开发评测默认只在 checkpoint 200/400 上运行官方完整指标集合，包括
`WorldArena`、`WorldArena_VLM` 和 `WorldArena_JEPA`；这些结果属于官方评测器
下的开发集评测，不表述为完整官方 test 结果。若 step 400 满足延长门禁，
checkpoint 600 再运行一次完整开发评测。

Trajectory Accuracy 作为主要优化目标；JEPA Similarity、Subject
Consistency、Image Quality、Motion Smoothness 等作为相对 Wan2.2 基线的
no-regression gate。

V-JEPA 仅作为冻结的外部评测器运行，不进入 Wan-Action-Lite+ 的训练分支、
训练损失或关键路径。

最终 checkpoint 完全根据开发集确定后，只在冻结的官方 test-1000 上运行
一次完整评测。

checkpoint 与 CFG 必须在 2026-08-24 冻结。2026-08-25 至 2026-08-29 默认
仅用于 test-1000 生成、三套官方评测、打包、校验和故障恢复；除明确 blocker
外不得继续修改模型。

基础 `modelname_test` 的 1000 个视频复用于 `WorldArena`、`WorldArena_VLM`
与 `WorldArena_JEPA`。`modelname_test_1` 和 `modelname_test_2` 仅属于单独的
action-following 多动作链路，待确认当前正式提交合同是否仍要求后再生成。

最终判断：训练方案不用因为 JEPA 而加入 V-JEPA；但 checkpoint 评测必须
补上 JEPA，且 JEPA 要成为画质和时序语义的防退化门禁。真正需要修正的是
评测分层、目录定义和官方脚本，而不是轨迹优先的模型设计。

### 工程硬约束

- 固定完整 `video_quality_ood` evaluator 到 WorldArena2 commit
  `7b3feee108427bee3380064bb5154970ed7468b5`。
- 三套隔离环境和所有权重、结果、视频、提交包均位于项目 artifact root。
- Wan2.2 原生基线使用多个固定随机种子估计正常波动，再确定各项
  no-regression 阈值；不得主观指定 epsilon。
- JEPA wrapper 强制满足 `expected_count == gt_count == generated_count ==
  stem_intersection_count`，拒绝缺失、损坏或被上游静默跳过的视频。
- 修复并测试官方聚合脚本中 `JEPA Similarity` 与 `JEPA_Similarity` 的字段
  不一致，但保留原始官方源码和补丁 provenance。
