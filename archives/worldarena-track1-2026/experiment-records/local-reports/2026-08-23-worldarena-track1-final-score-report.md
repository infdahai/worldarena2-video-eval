# WorldArena Track 1 最终冲分报告

日期：2026-08-24  
范围：Official FlowWAM Stage-1、Seed4、v16 q/v LoRA、SeedVR2 4090 兼容链、Adaptive Candidate System  
结论状态：历史评测已闭环；Official+seed4 是不可覆盖回滚冠军，Adaptive 主线尚未产出最终裁决

v1–v15 全版本总谱系：`reports/2026-08-24-wan-action-v1-v15-total-experiment-report.md`

## 1. 当前回滚决定

在 Adaptive Candidate System 完成前，部署回滚冠军冻结为：

```text
Official FlowWAM Stage-1
flow_scale = 20
seed = 4
resolution = 640×480
frames = 121
```

选择 Seed4 的依据不是单个 PID 或局部分数，而是完整证据链：50条视频全部通过 SHA256 与
逐帧解码验证；package、Base、VLM、JEPA、aggregate 五个正式 receipt 全部完成；最终 CSV
包含50行，15个正式指标均为有限值。相对 seed1，15项未加权均值从 `0.656995` 上升到
`0.658660`（`+0.001665`）。该均值只用于同数据、同 evaluator 的方向判断，不冒充官方隐藏
总分公式。

v16 step50 最后以 seed4、flow20 对 inference scale `0.50/0.75/1.00` 做了正面对比；三组
Breadth20 快速九项均值都低于 Official+seed4，未达到 `+0.003` 晋级门，因此没有运行 LoRA
候选 clean50。v16 step100 未通过预注册质量门，因此停止且不运行 step200。SeedVR2 在第三次
且最后一次 4090 memory repair 后仍发生 chunked RoPE OOM，正式停止。

## 2. Seed4 完整15项结果

| 指标 | Seed1 | Seed4 | 变化 |
|---|---:|---:|---:|
| Subject Consistency | 0.798286 | 0.797553 | -0.000733 |
| Aesthetic Quality | 0.387585 | **0.389571** | +0.001985 |
| Image Quality | **0.521135** | 0.520689 | -0.000446 |
| Background Consistency | **0.870120** | 0.869634 | -0.000486 |
| Dynamic Degree | 0.356577 | **0.357140** | +0.000562 |
| Interaction Quality | 0.664000 | **0.668000** | +0.004000 |
| Perspectivity | **0.892000** | 0.884000 | -0.008000 |
| Instruction Following | 0.692000 | **0.704000** | +0.012000 |
| Semantic Alignment | 0.911016 | **0.926963** | +0.015947 |
| Flow Score | **0.262336** | 0.261372 | -0.000964 |
| Depth Accuracy | **0.995980** | 0.994602 | -0.001378 |
| Trajectory Accuracy | 0.629107 | **0.630883** | +0.001776 |
| Photometric Consistency | **0.162124** | 0.159870 | -0.002255 |
| Motion Smoothness | 0.754825 | **0.758854** | +0.004029 |
| JEPA Similarity | **0.957833** | 0.956766 | -0.001066 |
| 15项未加权均值 | 0.656995 | **0.658660** | +0.001665 |

Seed4 的主要正收益是 Semantic `+0.015947`、Instruction `+0.012000`、Interaction
`+0.004000`、Motion Smoothness `+0.004029`、Aesthetic `+0.001985` 和 Trajectory
`+0.001776`。Perspectivity `-0.008000` 是最大单项回撤，但完整15项同口径仍为正。

## 3. v16 LoRA 裁决

训练合同保持冻结：Official Stage-1 parent SHA256
`e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`，rank/alpha
`8/8`，仅 DiT attention q/v，LR `2e-6`，batch1，gradient checkpointing，121帧，
640×480，flow20，官方 RGB diffusion objective，无额外 loss。

### Step50：通过 Breadth20 门

| 指标 | Official Stage1 | v16 step50 | 变化 |
|---|---:|---:|---:|
| Interaction | 0.710000 | **0.720000** | +0.010000 |
| Image Quality | 0.520868 | **0.522006** | +0.001138 |
| Instruction | **0.710000** | 0.700000 | -0.010000 |
| JEPA | 0.945533 | **0.946085** | +0.000551 |
| valid episode | 19/20 | **20/20** | +1 |
| coverage | — | — | +0.014198 |
| common-valid mean DTW | 0.067697 | **0.057559** | -0.010138 |
| paired W/L/T | — | 11/9/0 | — |
| black | 0 | 0 | PASS |

门禁决定为 `allow_step100`。step50 LoRA checkpoint SHA256：
`37ab801d3f04125a2dc0fe1ce6924dd90d84577c518bd18812a0806c9c1fff42`。

### Step100：未通过，停止 v16

| 指标 | Official Stage1 | v16 step100 | 变化 |
|---|---:|---:|---:|
| Interaction | 0.710000 | **0.720000** | +0.010000 |
| Image Quality | **0.520868** | 0.519589 | -0.001279 |
| Instruction | **0.710000** | 0.700000 | -0.010000 |
| JEPA | 0.945533 | **0.947961** | +0.002428 |
| valid episode | 19/20 | **20/20** | +1 |
| coverage | — | — | +0.027160 |
| common-valid mean DTW | 0.067697 | **0.057144** | -0.010553 |
| paired W/L/T | — | 7/13/0 | — |
| black | 0 | 0 | PASS |

轨迹、JEPA、black 通过，但 Instruction/Interaction/Image 三个主质量指标仅 Interaction
提升，不满足“至少两项提升”。最终 receipt 为 `passed=false`、
`decision=stop_v16_at_step100`；step200 未启动。最佳 v16 checkpoint 回退到 step50，但由于
缺少完整 clean50 15项证据，不替换最终 Seed4 冠军。

## 4. 关键故障与最小修复

1. **SeedVR2 4090**：unfused norm 与 VAE memory-safe 已越过，但第三次最终 repair 在首个
   DiT forward 的 chunked RoPE `torch.empty_like(values_hld)` 再申请 `362 MiB` 时 OOM。
   按预注册上限停止，不做第四次兼容实验。
2. **v16 staged resume**：首次恢复只加载0个 LoRA key，且 AdamW state 在 CPU 导致首个
   optimizer step device/dtype 不一致。增加 fail-closed 240-key/120-pair/rank8 精确恢复与
   optimizer state 对齐后，step51 和 step100 均真实完成。
3. **v16 VLM 证据污染**：旧 VLM receipt 含 GPU7 OOM error/空 metrics，且 candidate 也无法
   证明 GPU6-only。旧证据被隔离；output gate 改为逐行无 error、三项 normalized score
   有限，并强制 `CUDA_VISIBLE_DEVICES=6` 后重跑。
4. **Seed4 Base**：pinned `--detect_gt` 实际是 `store_false`，launcher 传参反而关闭 GT。
   通过回归测试后只删除该反向开关，本地与远端 evaluator 测试均 `32/32 PASS`。
5. **Seed4 JEPA**：checkpoint-scoped `TMPDIR` 过长，multiprocessing resource sharer 创建
   AF_UNIX socket 报 `OSError: AF_UNIX path too long` 并卡死。回归测试先失败，随后只将
   JEPA 的 `TMPDIR` 设为 `/tmp`；本地与远端均 `32/32 PASS`，重启后 JEPA 得分
   `0.9567664861679077` 并自动完成 aggregate。

## 5. 决定性收据、结果与哈希

### Seed4 生成与完整评测

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/stage1-dev-clean50-seed4/stage1-only.receipt.json
/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0600/dev-clean-50/receipts/package.complete.json
/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0600/dev-clean-50/receipts/base.complete.json
/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0600/dev-clean-50/receipts/vlm.complete.json
/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0600/dev-clean-50/receipts/jepa.complete.json
/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0600/dev-clean-50/receipts/aggregate.complete.json
/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0600/dev-clean-50/csv_results/aggregated_results.csv
```

- aggregate receipt SHA256：`834137eec209870a4c00063640bc3def6175bdd2c9852de4c883e2668eafad69`
- aggregate CSV SHA256：`4304828918719322f622f06d61c8a7c0650d7b9597d407386c2f4366fabaa2e7`
- package：50条；manifest SHA256
  `2793a0cc1db33cb6885f36c7706f550702487aae381036571e9781c42e11ba17`
- VLM：50/50，error=0；JEPA：`0.9567664861679077`

### v16

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/train-step50.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/checkpoints/step-50.safetensors
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step50/quality-gate.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/train-step100.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step100/quality-gate.complete.json
```

- step50 LoRA SHA256：`37ab801d3f04125a2dc0fe1ce6924dd90d84577c518bd18812a0806c9c1fff42`
- step50 optimizer SHA256：`ff4e10792b35c6c5c190f69b004c1c8b9d17381d3a805d580b0c33b5e436c150`
- step50 gate SHA256：`6447d1cc5f49231b1ce5918b77810ff3b79eb9f3bd8cdf0f63dc43411c4eeece`
- step100 LoRA SHA256：`445b811e301a7e70658a4c8179ba64da38ec52b222b87465a1ebf079719bb285`
- step100 optimizer SHA256：`bfa60478fbf389d4cf19df3e45d9c3d4ece15f330933b89fbb2dcf2f646f7aff`
- step100 gate SHA256：`1ed2c984f7692bd28c0c62ca257f70e48fe183126fbe1fa0d69bb4b377ed4461`

### 保留的失败证据

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/stage1-dev-clean50-seed4/full-eval-gpu0.failed-inverted-detect-gt-20260823.log
/data/di/worldarena2_track1_20260815/runs/flowwam-official/stage1-dev-clean50-seed4/full-eval-gpu0.failed-jepa-tmpdir-20260824.log
```

## 6. 停止项与当前剩余动作

停止且不再扩展：SeedVR2、v16 step200、更多 seed、prompt grid、Motion FT、新 action 生成架构。

2026-08-24 批准的 Adaptive Candidate System 是对旧“仅剩上线”结论的明确修订。剩余动作是：

1. 生成200个 zero-overlap episode 的 seed1/seed4 双候选（共400条），完成 raw/corrected
   latest-WA2 scorer；
2. 在160/40隔离分区上训练并裁决 input router 与 post-generation selector；
3. 并行完成唯一固定 action-to-text Breadth20；只有硬门通过才允许最多一个 clean50 候选；
4. 若所有新候选均未通过，保持本报告冠军，并执行服务健康检查、提交 smoke 和 MP4 SHA256
   端到端一致性验证。

## 7. 最终 seed4 × LoRA inference-scale 结果

固定 step50 LoRA SHA256
`37ab801d3f04125a2dc0fe1ce6924dd90d84577c518bd18812a0806c9c1fff42`、seed4、flow20、
20条、121帧、640×480，仅改变 LoRA inference scale。三组均有独立生成/视频验证 receipt，
20/20 SAM3、black=0，VLM逐行无error且有限，Base与JEPA有限。

| scale | 九项均值 | Δ vs baseline | DTW | JEPA | Instruction | Interaction | Image | 决定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Official+seed4 | 0.716048 | — | 0.051752 | 0.944844 | 0.730 | 0.740 | 0.517448 | incumbent |
| 0.50 | 0.712895 | -0.003153 | 0.050372 | 0.943039 | 0.700 | 0.740 | 0.519164 | reject |
| 0.75 | 0.712950 | -0.003098 | 0.060261 | 0.943600 | 0.710 | 0.740 | 0.518746 | reject |
| 1.00 | 0.715295 | -0.000753 | 0.053727 | 0.944762 | 0.720 | 0.750 | 0.519879 | reject |

selection receipt：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/seed4-scale-selection.complete.json
```

其 SHA256 为 `398d60330e96cae5952e83f794d3bb290685007d4b76a3d3aa97f3007c2e1513`，内容为
`winner=null`、`decision=freeze_official_seed4`。因此不运行候选 clean50，最终冠军保持完整15项
均值 `0.6586597567502485` 的 Official Stage-1 / flow20 / seed4。

三组 gate 路径分别为：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step50-seed4-scale050/quality-gate.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step50-seed4-scale075/quality-gate.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step50-seed4-scale100/quality-gate.complete.json
```

gate SHA256 依次为 `5474c52382897bae5610a47f6239ce5cb88446f939c3013b1534c81b360e5aa3`、
`dfbe5965a677caf4c6b39c777c635fa6f32bf1d091058a75be7a50efada23d38`、
`4b2e7d8e40e662909b556694239cf53c38ec17c9f13f8c16bb0e158a8080c792`。

附加检查：seed1/seed4 oracle 上限 `0.6683917311343808`、相对 seed4
`+0.009731974384132314`，两者各赢25条。旧轮次曾按原预注册规则不做 selector；该判断已由
2026-08-24 批准的 zero-overlap Adaptive Candidate System 合同取代。输出编码链是 generator
单次编码、package 原字节打包。上线必须验证返回/打包 MP4 SHA256 与 generator receipt 一致。
两项 receipt 为：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/automation/seed1-seed4-oracle.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/automation/track1-output-encoding-audit.complete.json
```

快速 gate finalizer 曾因读取 raw Photometric 而 fail closed；改为读取官方 normalized per-video
结果并新增回归测试后，本地与远端均 `4/4 PASS`。未放宽门禁、未改变实验合同。

历史评测结束时 GPU6 已释放；该状态不代表 Adaptive Candidate System 已完成。

## 8. Adaptive Candidate System 当前进展

截至 2026-08-24 20:16 的实时远端审计：历史冠军、Seed4 clean50、v16 step50/step100、
seed4×LoRA-scale、self-attention-only 消融、SeedVR2 停止证据均已完成或形成终止 receipt；新
Adaptive 主线尚未产出主体 artifact。

| 项目 | 进展 | 完成门/缺口 |
|---|---:|---|
| 200×2 seed 候选 | **0/400** | 缺 manifest、双 lineage generation/video-validation receipts |
| latest-WA2 scorer | 本地 `8 passed` | 缺远端 GT-reference motion receipt 与400条 raw/corrected 分数 |
| Input router / postgen selector | 0/2 | 缺40条隔离 holdout 与 paired-bootstrap receipt |
| Action-to-text | 0/20 | 缺固定模板 Breadth20 generation/quality receipt |
| Score-directed self-training | 未解锁 | 只可使用160条训练分区中 raw/corrected 同向且 margin≥0.003 的高置信 winner |
| v17 数据 | 7/100 archives，约8.5GB | 无 `download.complete.json`；仅为 P0/P1 双失败后的条件分支 |

Selector 主门不是 accuracy，而是 corrected uplift 至少 `+0.003`，task-stratified paired
bootstrap 10,000次的单侧90%下界大于0，raw/12项 non-motion 不回退，且 seed1/seed4 各至少
被选择5/40。最终最多开放一个 clean50：corrected mean 必须高于冠军 `+0.003`，12项
non-motion 均值不低于冠军 `-0.001`，Trajectory/JEPA 各自下降不超过 `0.01`；raw15 只作
catastrophic-regression guard。

设计合同与目标产物根：

```text
docs/superpowers/specs/2026-08-24-flowwam-adaptive-candidate-system-design.md
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system
```

复核时 GPU0/GPU4/GPU5/GPU6 空闲；GPU1/GPU2/GPU3/GPU7 有其他用户进程，未触碰。数据盘
剩余约839GB。完成判断继续只接受 exact receipt、artifact hash/count 和 finite-output validation。

### 8.1 21:32 CST 状态更新：四卡语料生成已启动

Adaptive 正式清单已完成，不再是 `0/400` 的空目录状态：200 episode、train/holdout=`160/40`、
GPU0/GPU4/GPU5/GPU6各50条，manifest SHA256
`4b147c15ba6b9fad45d66cdc542e57bbe9a967c0cd8d05a5f267cad7c702a18e`。决定性 manifest receipt：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/manifests/selector-200.complete.json
```

四卡进程 PID 为 GPU0 `1232329`、GPU4 `1232332`、GPU5 `1232339`、GPU6 `1232348`。冻结合同是
Official Stage-1 / flow20 / 121帧 / 640×480，同一 episode 在同卡按 seed1→seed4 紧邻生成；
每卡只加载一次模型。聚焦测试本地与远端均 `42 passed`，四个 dry-run 均为50 episode/100 video。
当前精确缺口是四份 `paired-generation.complete.json` 及400条逐视频 SHA/black=0 验证；在这些
收据形成前，冠军仍是 Official Stage-1 / flow20 / seed4，且不得从 PID 推断完成。SeedVR2 SP2
已经由终止收据关闭，不再运行。

### 8.2 四卡进度 45/400；正式评分链已排队

最新精确计数为 GPU0 `11/100`、GPU4 `12/100`、GPU5 `10/100`、GPU6 `12/100`，合计
`45/400`。四个生成 PID `1232329/1232332/1232339/1232348` 均存活，GPU利用率均100%、显存约
15.3GiB，错误标记均为0，磁盘剩余约814GiB。仍缺四份 `paired-generation.complete.json`，所以
正式冠军保持 step-0600 Official Stage-1 / flow20 / seed4。

200行 action feature receipt 已生成，输出 SHA256 为
`4a09c3f5fa1c593f111efd62d9a490c075455e576165f7d315633d09fc19e0d8`：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/features/action-features.complete.json
```

eval queue PID `1325650` 只等待最终 receipt，不提前占卡。完成后将 seed1/seed4 分别写入
step-0740/0750，四卡并行 Base/VLM/JEPA，保存 raw15 与 GT-reference Motion corrected15，最后执行
160/40 input router/post-generation selector 裁决。相关扩展测试本地和远端均为108项通过；不会
覆盖冠军或使用40条 holdout 拟合。

### 8.3 22:35 CST 运行复核：89/400，定时上报已恢复

正确 PID/日志路径复核确认四个生成 PID `1232329/1232332/1232339/1232348` 与 eval queue PID
`1325650` 均存活，先前的空 PID 只是监控脚本误读 `generate.pid` / `eval/` 所致。当前逐视频
MP4/receipt 为 GPU0 `22/100`、GPU4 `23/100`、GPU5 `21/100`、GPU6 `23/100`，总计 `89/400`；
四卡日志持续刷新且错误标记为0，磁盘剩余约812GiB。

仍缺四份 `paired-generation.complete.json`、combined receipt、step-0740/0750 aggregate、
latest-WA2 scorer 和 selector holdout receipts，所以系统未完成，冠军继续冻结为 step-0600。
既有 `worldarena-final-score-pipeline` 自动任务已由 PAUSED 修正为 ACTIVE，每15分钟主动上报，
不会创建重复任务。

### 8.4 Adaptive 生成闭环并进入正式评分

200个零重叠episode的seed1/seed4配对生成已达到400/400。GPU0/4/5/6各100视频，四份
`paired-generation.complete.json`均确认121帧、640×480、black=0、逐视频SHA和Official
Stage-1 SHA `e211e32b…96c4`；收据SHA依次为 `40a068df…af3b3`、`f29acf16…83134`、
`e242a5a2…744be`、`674cfb05…ec406`。

首次评测 staging 因输入HDF5软链接的允许根写错而停止。最小修复显式区分 artifact-root 输入与
adaptive-root生成视频，并通过本地/远端 `4 passed`。预检同时发现step-0740已有既有20条
self-only血统，因此没有覆盖或误复用，正式Adaptive两侧改用空闲step-0750/0760。新队列
PID `2185267` 已产生200×2合并血统收据（SHA256 `f6785cd3…9779344`）：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/combined/combined-lineages.complete.json
```

当前正在打包并准备Base/VLM/JEPA/aggregate及latest-WA2修正评分；selector的40条holdout尚未
打开，最终冠军与部署结论仍保持Official Stage-1 / flow20 / seed4。

### 8.5 Adaptive 评测运行时修复与 action-to-text 准备

两条200视频 lineage 的package receipt已完成。并发Base暴露官方评测默认分布式端口29500冲突：
step-0750 Base退出，step-0760 Base及seed1/seed4两条VLM保持运行。已按物理GPU固定
`MASTER_PORT=29500+gpu`，远端聚焦测试 `4 passed`；恢复器 PID `3382065` 将在三个现存phase
结束后只补跑缺失Base，不重复活跃任务。

唯一 action-to-text Breadth20 的20条确定性输入已形成 receipt，SHA256
`82c0a7323d9bd5ae0cee15e5214738c70bd351e315290581596992d4d6e3358d`。它尚未占GPU、未生成视频、
未通过质量门，因此不能参与冠军选择。selector holdout仍未打开，冠军和部署结论不变。

### 8.6 09:17 CST receipt-gated 快照

候选生成仍为400/400，四个 shard 与 combined lineage 收据齐全。正式评测在批准的 GPU4/5/6
持续运行：step-0760 Base 73/200、seed1 VLM 173/200、seed4 VLM 185/200；GPU0 空闲，未触碰
GPU1/2/3/7。恢复器 PID `3382065` 等待活跃 phase 后只补缺失 step-0750 Base，固定模板
action-to-text queue PID `3522020` 等待 step-0760 Base receipt，当前仍是0/20输出。

flow16/20/24 oracle 已形成终态关闭收据。原因是既有三条生成血统只具备旧 ATR/SAM3 诊断结果，
不具备严格 matched corrected-15 分数，而合同禁止新增 flow16/24 生成；因此决策为
`close_missing_prerequisites_no_new_generation`，不会训练或启用 flow router：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/flow-oracle/flow16-20-24.complete.json
SHA256 15430fea559c75bdaa925eaf6b98f5f5f3c8b7096e5eec2717909f73b24ec26c
```

Base/VLM/JEPA/SAM3/aggregate/latest-WA2、selector/cascade、action-to-text gate、训练分支与
`final-decision.complete.json` 仍缺。当前没有候选获得 clean50 配额，Official Stage-1 / flow20 /
seed4 继续作为不可变冠军。

### 8.7 Base receipt 与 Action-to-text 修复启动

step-0760 Base 已形成200/200终态 receipt，SHA256
`51e405da36ad679a0e938bb7da7f0e2bb01e84549a36dcb33679a296978ac686`。P1 队列随后暴露单一启动
参数错误：已设置 GPU4，但 Stage-1 调用遗漏 `--physical-gpu "$GPU"`，因此在任何视频生成前
按默认GPU6保护失败。修复严格限于补传参数，并以先红后绿的回归测试验证；本地和远端均
`4 passed`，同步脚本 SHA256 为
`58a3e810f97119bb7cc6921c78d72e777d49c3816bd634b489029a51a35bcfae`。

只重启了失败的 owned action-to-text queue。PID `3772998`、runner `3773008` 现于 GPU4
`GPU-651bd630-9a1d-c4c2-1089-8501d6ef1051` 正式运行，已进入首条 DiT 去噪且没有新错误；此时
视频仍为0/20，故尚无 video-validation/quality-gate receipt。VLM seed1/seed4 当前约179/200、
190/200；最终候选与 clean50 配额不变。

### 8.8 Seed4 VLM receipt-gated 完成（2026-08-25 09:58 CST）

seed4 VLM 已完成200/200并形成正式收据，SHA256
`4606b8ff85cbadb6a9b34c3afa912013795bbf115539e573dcfd46c53edf37c2`。输出复核为200行、
1200个数值字段全部有限、error=0；GPU6 已释放，没有用 PID 存在代替完成判断。seed1 VLM
当前192/200且继续刷新，预计约18分钟；恢复器仍只等待后补 step-0750 Base。

action-to-text 当前9/20，GPU4 runner PID `3773008` 和 generation lock 匹配、无错误，预计约30分钟。
其余 decisive receipts 与 `final-decision.complete.json` 仍缺，因此尚未分配 clean50 候选，部署仍保留
Official Stage-1 / flow20 / seed4。

### 8.9 双 VLM 收据闭环与 Base750 交接（2026-08-25 10:16 CST）

seed1 VLM 已完成200/200，receipt SHA256
`2bd5d98a8f29ac25f36c6603c760fdd6f5902c2e5a90b4aacb70c62d336476a2`；200行输出、1200个数值
字段均有限，error=0、空 metrics=0。结合上一节 seed4 收据，两条 VLM lineage 均已正式闭环。

恢复器已仅启动缺失的 step-0750 Base，PID `3929854`、物理 GPU0、eval lock 归属均与批准链一致，
没有重复已完成任务。action-to-text 为15/20且无错误。Base、JEPA、aggregate/Motion、selector、
训练和 final-decision 仍以未形成终态收据计，冠军与 clean50 配额不变。

### 8.10 Action-to-text 20/20 视频血统闭环（2026-08-25 10:35 CST）

唯一 action-to-text Breadth20 已生成20/20，生成 receipt SHA256
`1270106c53855e638020445afe5f0ab69fd37233c583a844571dfcc2990a7ddc`。原队列遗漏了生成后的
video-validation；已测试先行补上 CPU 验证器和以验证收据为终点的队列条件，复用现有视频、未发生
第二次生成。本地9项、远端5项测试通过。

新视频验证 receipt SHA256 `20ed7a9a2650254abecc0b9ae4cb551ba1773f8dce7e28626c7a7503773cefff`，
结果为20/20、121帧、640×480、black=0、20个唯一 SHA，并绑定生成收据。P1 质量门仍缺正式
VLM/JEPA/Trajectory/raw/latest-corrected 证据，不能进入 clean50；Base750 仍在 GPU0 正常推进。

### 8.11 11:35 CST 决策快照：主链运行、P1 已恢复

没有新的冠军候选，也没有分配 clean50。正式评分收据现状为：seed1/step-0750 已有 package 与
VLM，Base 在 GPU0 运行，JEPA/aggregate 缺失；seed4/step-0760 已有 package、Base、VLM，
JEPA/aggregate缺失。主 queue PID `3382065` 的活跃 Base 子链与 GPU0 锁/UUID一致，当前子指标
75/200，日志无 Traceback/OOM/RuntimeError。两侧 VLM 与 seed4 Base 的正式 SHA 与上一报告一致，
不能以当前 PID 或进度条代替剩余 receipt。

Action-to-text 质量队列的 step-0751 Base 首次运行因与主队列共用默认端口29500而触发
`EADDRINUSE`；40/40轨迹预处理和既有20条视频未损坏。旧 PID `4083982` 已退出后，只在空闲
owned GPU4 以独立端口29504恢复，queue PID `29400`、子 PID `31494`、GPU4 UUID与环境回读
均匹配；当前 semantic_alignment 14/20且无新错误。step-0751/0752 package receipt 已存在，
quality-gate receipt 仍缺，因此 P1 仍不具备候选资格。

Task1--2 已生成400/400并通过四 shard 与 combined lineage；flow oracle 已证据关闭；P0评分、
P1质量门、背景投影、selector/holdout、两条条件LoRA和最终决策仍需各自终态收据。LoRA 的
train-only高置信筛选与共享噪声合同聚焦测试12项通过，但尚未达到训练启动前置条件。预计3--5小时
得到 P0/P1 首轮决定，完整最终决策约6--22小时，取决于是否解锁训练和一次 clean50。当前部署动作
仍为保留 Official Stage-1 / flow20 / seed4；不可变 clean50 raw mean仍为
`0.6586597567502485`，`final-decision.complete.json` 尚未形成。

### 8.12 P1 incumbent Base 获得正式收据（2026-08-25 11:55 CST）

step-0751 Base 已完成20/20并形成 receipt，SHA256
`83bde0774a3e42fb993be4c0277dbe97316025e5548821c7249b2b58a85ebb1e`；端口恢复由正式收据而非
PID存在证明成功。同一 queue PID `29400` 已转入 incumbent VLM 3/20，日志无错误。P1仍缺
VLM/JEPA/aggregate、step-0752 candidate全阶段和最终quality gate，因此候选资格、clean50配额与
Official Stage-1 / flow20 / seed4 部署决定均未改变。

### 8.13 加速后决策路径（2026-08-25 12:19 CST）

seed1 Base 与两侧 JEPA 已获得正式终态收据：Base SHA256
`2421f57c15303540d3f3a5a735a509f0c7c688db3472c6a2f1fe96c81f55430f`，seed1/seed4 JEPA SHA256
分别为 `8073d9d35962bb29a8a0fdff6dcaf33449814d0b5fbe71ec6915efcb6398417e`、
`e64c36c7b9943ef0a254446e24b3e80eb90a48a2bc17bd61a0bf21fcb6fd22ea`。一次 live 脚本替换导致
aggregate 在缺 JEPA 时失败关闭，没有形成错误收据；失败日志已保留，phase 与 GT receipt-skip
保护已完成本地/远端4项回归。

当前 P0 仅等待 GPU0 GT-motion 后续接 aggregate/latest-WA2/selector；P1 incumbent VLM 在GPU4，
candidate Base 已提前在GPU5运行。恢复器 PID `191763` 以 exact PID 和 receipt 为门，不复制作业。
预计 P0/P1首轮决定约1--2小时；在此之前没有候选资格或 clean50 分配变化，冠军仍为 Official
Stage-1 / flow20 / seed4。

### 8.14 P0 首轮正式通过（2026-08-25 12:55 CST）

P0原始与最新版GT-capped评分已完成200对：seed1/seed4 raw15 分别
`0.6696694547460158` / `0.6695985385719183`，corrected15 分别
`0.6666362780797835` / `0.6665121650242082`。GT三项各200条，收尾文件命名不一致通过
receipt-only canonical finalize 修复；对唯一 Semantic Alignment `1.000977` 采用有测试的0.001
FP舍入容差，保留pinned raw值并拒绝更大超界。

冻结40条holdout仅打开一次。input router失败；post-generation selector通过：corrected uplift
`+0.003067329787`、单侧90% bootstrap lower `+0.0028273658672715294`、raw
`+0.0030790212375193136`、12-metric non-motion `+0.003942522790525055`、seed1/4选择16/24。
policy receipt SHA256 `aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`，
因此该策略获得唯一 clean50 候选资格，但尚未消耗配额。

训练分区高置信支持为seed1 71、seed4 58，平衡后116条，P2 Winner-SFT/Preference-LoRA step25
已满足数据门；正式pseudo-target receipt和production-shape smoke为下一步。P1 incumbent已闭环，
candidate Base在owned GPU4继续；一次GPU5提前预跑与原queue重叠已通过只停止duplicate修复。
最终替换门尚未执行，部署仍保留 Official Stage-1 / flow20 / seed4。

### 8.15 P2 数据血统正式闭环（2026-08-25 13:05 CST）

正式 pseudo-target receipt SHA256 为
`4b9d270a94b13ba0192d7d8b6b416e815efa721c0ed83bf1ac20a0f5543ab966`：160条训练分区中筛得
seed1/seed4 eligible 71/58，确定性平衡后58/58、共116条；40条 holdout 明确排除，每条
winner/loser视频路径与SHA256均已绑定，构建器本地13项、远端12项测试通过。因此两条P2分支只
解锁 production-shape smoke，尚无合格 checkpoint 或最终候选。

P1 candidate Base 已收据闭环，唯一 queue PID `29400` 在 owned GPU4 进入 VLM，13:05进度2/20，
日志无错误；GPU0/5/6空闲。按当前速度P1质量门预计40--60分钟。clean50仍未消耗，当前最终部署
动作不变：保留 Official Stage-1 / flow20 / seed4。

### 8.16 P2 双分支通过真实 smoke（2026-08-25 13:43 CST）

Winner-SFT/Preference-LoRA production-shape smoke receipt SHA256分别为
`947e3bcb52b9a61f15cae38c838ee581c937325292141c049c4390500827f940` 与
`b5a75c5f6f2c6c889a178cd0b12cfc455860cb4d34f84e06080cc973bc7007d0`。两者均完成真实
batch1、121帧、640x480前反传，显存低于22GiB，并证明可训练集合只有30层self-attention q/v
LoRA A/B共120个张量。13:43正式step25进度为20/25与9/25；Preference共享timestep/noise证据逐步
落盘且loss有限。P1 candidate VLM为19/20；质量门、两条checkpoint质量门和最终决策仍待回执。

### 8.17 最新候选裁决状态（2026-08-25 13:52 CST）

P1 action-to-text已终态关闭（quality receipt SHA256
`67ccea4098b8093e80d164a248a586224a999f0a68e9b6e79a5f16181c9bc342`）：corrected +0.007663、raw
+0.007413、Interaction +0.03，但Instruction +0、corrected motion -0.000413，不满足冻结门。Winner-SFT
step25已落盘，训练receipt SHA256为`0586df9004f4bc977fbeb568847ddb4bd309d7c7c242f23d22ea7762f8fda999`，
LoRA/optimizer哈希均独立匹配。其seed1/seed4 Breadth20首启在任何视频生成前由旧过滤器安全拦截；已用
显式self-qv source-layout做最小修复并保持旧all-qv路径不变，失败尝试0视频，仅恢复缺失行。

### 8.18 P2两条step25均进入配对质量门（2026-08-25 14:14 CST）

Preference-LoRA正式训练receipt SHA256为
`623af5d3e9d888b583fc600794aaec7e4b373ed972ac233b0a5fa942cae03e51`，checkpoint/optimizer哈希已独立
验证；共享episode/timestep/noise合同25/25成立。Winner seed1/seed4当前5/20、6/20，Preference的
seed1/seed4已分别在GPU4/GPU5启动，四条线均只使用冻结Breadth20。最终候选、clean50分配和部署
仍未裁决，回滚冠军保持Official Stage-1 + flow20 + seed4。

### 8.19 P0 selector获得clean50正式替换资格（2026-08-25 14:34 CST）

已立即执行冻结selector的一次性clean50终态门，仅组合既有seed1/seed4各50条视频和钉死的raw/
latest-WA2 corrected分数。策略没有因clean50反馈重新拟合；27维特征仍只含部署时可得的九项生成
指标、两seed值及差值。选择seed1/seed4为27/23，50条15指标全部有限，black0、121帧、640x480，
返回MP4 SHA与各自generator receipt逐条一致。

新候选corrected mean为`0.6592060841461832`，超过corrected champion
`0.6549900090676842`共`+0.004216075078498971`；raw mean为`0.662720295147882`，超过不可变raw
champion `+0.00406053839763354`。12项non-motion、Trajectory、JEPA分别提升
`+0.005244083882822492`、`+0.004067166005776723`、`+0.0005757522583007546`。所有最终替换护栏
通过，terminal receipt SHA256为
`1d72aa6b1a6f260c963756309f3c74a57504381b929663b9361a8201248b187e`。因此当前排序为：P0 selector
已具备最终部署资格；Official+seed4保留为不可变回滚；P2仅在Breadth20证据明显强于P0时才竞争
唯一新生成clean50资格。Winner双seed已完成20/20；Preference seed1/seed4仍为13/20、14/20。

### 8.20 P2动作门已裁决，三条九项门并行（2026-08-25 16:12 CST）

四组step25 Breadth20均已20/20、black0并有Base receipt。复用已落盘SAM3轨迹的低成本动作门结果
为：Winner seed1通过（valid 19/19、coverage `-0.001234568`、common-valid DTW
`-0.004719808`）；Winner seed4通过（20/19、`+0.043209877`、`-0.015589791`）；Preference
seed4通过（20/19、`-0.006172840`、`-0.001441817`）；Preference seed1因coverage
`-0.020987654`和DTW `+0.012798740`双重越界而终态关闭。该裁决没有修改checkpoint或评分门槛。

幸存三组从16:06起在GPU0/GPU6/GPU5并行跑VLM→JEPA→aggregate→latest-WA2 corrected；16:12均
2/20，速度135--137秒/条，无错误，预计16:52--16:56出齐九项数据、17:05前裁决。无P2晋级时
约17:30--18:00可收口；若一条P2赢得唯一新clean50配额则完整结束约3--5小时，Winner step50仅在
step25明确过门且仍有收益余量时解锁，极端上界约5--8小时。Task 6 Action-Support Background
Projection尚无远端终态receipt，已作为显式未完成项进入最终清单，不能计为完成或静默跳过。
当前可部署排序不变：P0 selector是唯一通过clean50最终门的候选，Official+seed4是回滚冠军。

### 8.21 Task 6终态与最新ETA（2026-08-25 16:17 CST）

Action-Support Background Projection已完成证据审计并终态关闭。现有20条seed4视频、首帧、
robot-only HDF5及SAM3 centroid trajectory完整，但冻结四项support union需要的逐帧renderer/
expected-flow mask、generated-RAFT active-flow tensor与SAM3 robot/object mask均未落盘，受审根目录
计数为0。按合同不得以centroid伪mask、部分union或新阈值搜索代替，故未生成投影视频、未修改门槛。
终态receipt为`adaptive-candidate-system/background-projection/quality-gate.complete.json`，SHA256
`3a0c5f149ed34b6b55437650909ba77f99ebf676e4cd2c1932db1d4256e757a8`，状态
`closed_prerequisite_failed`。Task 6现已全部receipt-gated。

三条P2幸存组16:17均为VLM 3/20、无错误，预计16:52--16:56出齐、17:05前完成Breadth20裁决。
若无P2取得clean50资格，约17:30--18:00完成最终决策和文档收口；若一条取得唯一新clean50配额，
完整结束约3--5小时；若Winner step25明确通过并条件解锁step50，极端上界约5--8小时。当前部署
排序不变：P0 selector领先，Official+seed4保留为不可变回滚。

### 8.22 当前正式冠军与最终P2比较合同（2026-08-25 16:44 CST）

当前正式冠军已更新为冻结P0 post-generation selector：clean50 corrected
`0.6592060841461832`、raw `0.662720295147882`，收据SHA256
`1d72aa6b1a6f260c963756309f3c74a57504381b929663b9361a8201248b187e`；Official Stage-1 + flow20 +
seed4是不可变回滚冠军。后续任何P2只超过旧seed4不构成晋级。

正在用相同Official seed1/seed4 Breadth20构建冻结`P0-selector-Breadth20`：seed1 corrected mean已为
`0.6652987549716861`，GPU4只补seed4缺失的Base/Trajectory和aggregate，未生成新视频。三条P2
幸存组16:44均约VLM 17/20。它们必须相对P0同时达到corrected `+0.003`、raw不低、non-motion
`+0.002`、Trajectory/JEPA不低于`-0.005`、paired wins至少12/20及black0，才可使用唯一新clean50。
预计15--25分钟得到Breadth20硬裁决；若均失败，约45--90分钟形成最终决策；若唯一分支通过，完整
clean50约再需2--4小时。Winner step50默认不运行，除非step25先过P0门、corrected uplift至少
`0.006`且Instruction/Interaction/Image均无结构性回撤。

### 8.23 最终冠军、关闭分支与官方测试边界（2026-08-25 17:39 CST）

Adaptive MoE Task 1--9已形成receipt-gated终态。冻结P0 Breadth20为corrected/raw
`0.6729899737`/`0.6754179853`。Winner-SFT seed1为`0.6661127522`、wins `8/20`；Winner-SFT
seed4为`0.6801319441`、相对P0 `+0.0071419704`，但wins只有`11/20`且Instruction回撤`-0.01`；
Preference seed4为`0.6725589567`、wins `11/20`。三者均未完整通过冻结门，Preference seed1已在
动作门关闭，Winner step50与native-v17均按合同跳过，没有新clean50消耗。

最终正式冠军为P0 post-generation selector：dev-clean50 corrected/raw
`0.6592060841461832`/`0.662720295147882`，选择seed1/seed4 `27/23`，50条有限、black0、生成与返回
MP4 SHA一致。Official Stage-1 + flow20 + seed4仅是不可变回滚。最终收据：
`/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/final-decision.complete.json`，
SHA256 `2ad2b56f162b3863f6acfee2944dc443613d42bb7322e198a85dbcea5bfc8d59`；状态
`complete_retain_p0`，deployment SHA gate为true。冻结model和policy分别为
`selector/postgen/postgen-selector.model.json`（SHA `4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa`）及
`selector/policy-selection.complete.json`（SHA `aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`）。

此处“最终”仅指已批准的Adaptive MoE开发活动。当前官方WorldArena 2.0 Track 1指定测试资产
`WorldArena/WorldArena2.0/dataset_track1.tar.gz`尚未下载到远端，故上述分数不得称为官方测试或
leaderboard分数。正式测试集后续只能用于一次性最终推理、封装和提交，禁止回流训练或候选裁决。

### 8.24 正式测试资产已补齐，尚未执行官方推理（2026-08-25 18:15 CST）

8.23中的“尚未下载”现已闭环，但P0的`0.6627202951`仍只是dev-clean50结果，不能改称官方测试分。
正式Track 1资产已固定到revision `af1ac34d3881f84096345542c631fbb1b9540d50`，压缩包SHA256为
`7973bf1e9222086279aa12eb593b4c7adb88f781a5811b05ae5134d52c96bb41`。解包后episode1--1000连续，
HDF5/首帧/instruction均为1000份，black/生成评分尚未开始。

远端路径：`/data/di/worldarena2_track1_20260815/datasets/WorldArena2.0-official-track1`。主数据收据
SHA256为`d6f66953860abca36b2b495ac96879924fa965f08761f38521386bc817b73eb1`，1000行episode manifest
SHA256为`97ef493062d9e21b4042a9987dd550c12f34763cfa345de786f40eca27f8e825`，零污染收据SHA256为
`4d44235e3081d4d5a4b43564533125b14bdefc6981fb88bc3341b04b56f84d0a`。正式测试资产和解包文件均只读；
5811个既有运行文本文件与活动进程均无该路径引用。WorldArena 1.0/2.0公开非测试split可另行加入训练和
离线测试，但正式Track 1 test必须保持独立，只用于最终推理、封装与提交。

### 8.25 当前冠军不变，六卡Challenger正在裁决（2026-08-25 19:54 CST）

当前冠军仍是P0 selector（clean50 raw `0.6627202951`），不是新模型。新增Oracle证明Winner rollout
作为局部expert有上限：P0+Winner为`0.6843627765`，相对P0 `+0.0113728028`；ActionText的独立增量
不足`0.003`并继续关闭。唯一一次scale0.5已经完成20/20生成并进入step774全评分；在它通过相对P0
门以前，不改变冠军、不生成新clean50、不启动step50。

全新fresh40 manifest/receipt SHA分别为`49d89ac6c3450aff8e1705819f455b8fbab7eaff444f67bd0f55b92ede586908`/
`a86ceed7be985c3d8632c5a12b0011c1cf52a0710cbec4686e4fa0986b196a20`，与旧200、fast20、clean50隔离，
正式test未打开。GPU0/2/3/4/5/6现组成六卡流水；GPU1外部任务严格避让，GPU7不用。GPU2/3分别
跑Base/VLM，GPU4已完成JEPA，其余卡开始fresh40 P0双seed生成并设置收据门控接力。

硬截止为8月27日白天研究冻结、8月27日晚正式test-1000启动。六卡理想完整正式流水约28--30小时；
保留60--72小时窗口。任何正式test结果都不得回流调模型、selector或阈值。

### 8.26 最终冻结：P0保留，ExpertPool与Safe-Winner关闭（2026-08-26 07:18 CST）

ExpertPool-v2已完成全部train160/Fresh40生成与Base/VLM/JEPA/aggregate/latest-WA2评分。policy在
Fresh40前冻结；由于train160 OOF候选对Trajectory/JEPA存在结构性回撤，最终threshold为1.0。
Fresh40一次性结果为P0 40/40、Winner/ActionText均0/40，corrected uplift与90%单侧bootstrap
下界均为0，终态closed_uplift_below_0_004。收据SHA256：
76df6b8733234eba4e03ea5b5498ad097483a40497f499573ea3f0c634bced29。

Safe-Winner-25未启动。已有收据给出的训练时间为1182.57s，六卡Breadth20和完整评分门的实测
下界分别约900s和3360s，总计约91分钟，超过07:12至08:30仅78分钟的资源窗口；因此按
closed_resource_deadline形成终态，SHA256
bbb04f6a2ab7763bc843287cb8cf34fd596ab3196612e57c7e0fe7904a0af23b。

当前正式开发冠军仍是P0 post-generation selector：dev-clean50 corrected/raw
0.6592060841461832/0.662720295147882，seed1/seed4选择27/23，50条全有限、black0、
deployment MP4 SHA一致。最终决策路径为
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/final-decision.complete.json，
SHA256 5eb9e43450bef05e24e8f8f600b5668f66971370b9c31e85422f63e272fad9cf；状态
complete_retain_p0、deployment SHA gate为true，并包含25项terminal outcome。冻结模型/策略路径
分别为selector/postgen/postgen-selector.model.json与selector/policy-selection.complete.json。
07:18已停止全部自有轮询作业；GPU0/2/3/4/5/6无自有WorldArena进程、利用率0%、无活动锁，完成
交卡。以上仍是开发验证分数，不是官方test-1000或leaderboard分数。

### 8.27 Track 1发布就绪烟测：链路通过，官方1000条未执行（2026-08-26 12:20 CST）

冻结冠军仍是P0 post-generation selector，没有重新训练、调阈值或打开新候选。新增发布工具对
final decision/model/policy/官方数据receipt做精确SHA门，建立1000条双seed计划，并将冻结P0选择、
视频校验、确定性打包、HF上传和匿名下载串成单一receipt链。clean50独立重放得到与原终态完全一致的
predictions SHA `102184cde847bcdf4d9fb7a4f47a43b5d6f1762b32af4b5490951da209655ea0`、27/23选择、raw
`0.662720295147882`和corrected `0.6592060841461832`。

正式Track1 episode1的seed1/seed4已在GPU0/GPU4成功生成，121帧、640x480且生成收据/视频SHA通过。
随后使用现有P0视频构造1条严格smoke包并上传公开HF dataset
`clusternlh/worldarena-track1-p0-submission-smoke`；commit为
`8165ebbaa2a178e38d9e7e5fae8897236466c60c`。清除HF认证后经7890代理匿名回下载，archive SHA与本地
完全相同：`83873237aad18f353a542b722092c744e63146a65ffa717791b48d094b54d609`；仓库非private、非gated、
根目录仅一个tar.gz。总收据：
`/data/di/worldarena2_track1_20260815/submission/releases/p0-r1/track1-release-verification.complete.json`，
SHA256 `bfccefb91925af8dcaa57a56405273f3a585f4cde177d8144153cd7b0cc07c73`，标记
`ready_for_full_test1000=true`、`full_test1000_executed=false`、`official_submission_not_triggered=true`。

因此当前结论是“发布链路可执行”，不是“已拿到官方test分”。正式1000条仍需另行启动，且只能使用
冻结P0、generated-only九项特征和已固化的stage/package/HF门；不得根据test内容或结果修改任何研发
决策。运行手册位于`docs/TRACK1_P0_RELEASE_RUNBOOK.md`。
