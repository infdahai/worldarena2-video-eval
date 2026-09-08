# WorldArena Track 1：Wan-Action v1–v15 总实验报告

日期：2026-08-24  
项目：WorldArena 2.0 Track 1 / action-conditioned video generation  
正式产物根：`/data/di/worldarena2_track1_20260815`  
远端源码：`/home/huazhi/nlh/baseline`  
当前状态：v1–v16 历史实验已按 receipt 闭环；Official Stage-1 / flow20 / seed4 是不可覆盖的
回滚冠军。Adaptive Candidate System 已批准进入执行，但尚未产生候选裁决，不能写成“全部结束”。

## 1. 总结论

v1–v13 逐步验证了 action raster、双臂身份隔离、时序路由、SE(3)、counterfactual、native
attention、persistent controller、显式时序边运输和视觉质量微调。工程上出现了大量可复用
组件，但反复得到同一个研究结论：**FM、hidden separation、probe 或训练 loss 改善，不能
替代 matched RGB 与官方轨迹证据。** `clean-gated-step10` 是旧 Wan 血统中唯一稳定保留的
action parent，但后续更复杂的 action controller 都没有形成更强的 rollout 证据。

v14 首次切换模型血统，直接采用公开 Official FlowWAM WorldArena Stage-1，把 robot flow
作为共享 DiT 中的原生条件。它在 matched dev-fast20 和 Breadth20 上显著超过 Custom-v15，
因此 v15 不再扩 action 架构，而是围绕 Official Stage-1 做有界推理、完整评测、Seed4、
SeedVR2 兼容与一次极短 q/v LoRA continued fine-tuning。

当前证据充分的回滚冠军为：

```text
Official FlowWAM Stage-1
flow_scale = 20
seed = 4
121 frames
640×480
```

Seed4 的 dev-clean50 完整15项均为有限值，15项未加权均值由 seed1 的 `0.656995` 提升到
`0.658660`。v16 q/v LoRA step50 随后以最终 seed4 正面对比了 inference scale
`0.50/0.75/1.00`；三者 Breadth20 快速九项均值变化分别为 `-0.003153/-0.003098/-0.000753`，
全部未过预注册 `+0.003` 门，因此未启动任何候选 clean50。step100 也未通过冻结质量门，未运行
step200。SeedVR2 在第三次且最后一次 4090 memory repair 后仍于 chunked RoPE OOM，正式停止。

## 2. 证据等级

| 等级 | 含义 | 本报告允许的结论 |
|---|---|---|
| E0 | 设计、数学、形状和单元测试 | 只能证明合同存在 |
| E1 | production-shape smoke、梯度、显存、checkpoint | 只能证明工程可运行 |
| E2 | held-out hidden/probe/counterfactual audit | 只能证明内部机制信号 |
| E3 | matched RGB 小样本 proxy | 可用于有界候选筛选 |
| E4 | pinned official SAM3 / WorldArena dev 评测 | 可声明该开发集上的正式相对结果 |
| E5 | 官方最终 test/submission | 本轮尚未执行，不能提前声称 |

全路线最重要的纪律：E0–E2 的成功不能写成 RGB 或官方分数提升；只有 receipt、视频完整性、
matched 条件和相应 evaluator 同时成立时才允许晋级。

## 3. v1–v15 一页总览

| 版本 | 核心变化 | 最高证据 | 关键结果 | 最终状态 |
|---|---|---|---|---|
| v1 | 8通道 action raster + zero-init residual | E0/E1 | 几何、shape、梯度与 identity 合同成立 | 保留 raster/zero-init 思想 |
| v2 | 真实 Wan Adapter/LoRA 初代训练 | E4 | S1A125 成长期 baseline；Stage2 长训黑屏，LoRA 未过门 | 淘汰长训 recipe |
| v3/v3.2 | 10通道、11D pose、81→21 因果、support、weighted FM、严格 lineage | E4 | support-gated step10 首次形成 matched RGB 正信号 | 保留 `clean-gated-step10` |
| v4 | 双臂独立 geometry stream、Pose-FiLM、late fusion | E3 | arm-swap proxy 改善，但 DTW 约恶化110.7% | 淘汰 |
| v5 | 每臂 causal TCN motion correction | E2 | router 无未来泄漏，但 Wan 输出 leakage 仍增 | 淘汰 |
| v6 | frozen gripper probe + position/velocity supervision | E2 | probe 有效，position/velocity correction 均变差 | 淘汰 correction，保留 probe |
| v7 | arm-grouped SE(3) gate-only attention | E2 | 数学与运行正确，shift/swap separation 不稳 | 淘汰 gate-only |
| v7.1 | geometry-only Q/K/V/O LoRA | E2 | 容量与梯度成立，weighted FM 不辨 action correctness | 淘汰 FM-only |
| v7.1-CF | v7.1 + counterfactual ranking | E2 | audit20 三类门均失败 | 结束 frozen-small-branch 路线 |
| v8 | blocks 8–13 native Q/K/V/O partial unfreeze | E2 | reverse 16/20、swap 17/20，hard shift 5/20 | 保留 native-band 思路 |
| v9 | four-token phase-locked action cross-attention | E2 | 工程健康，ranking 接近随机 | 淘汰 |
| v10 | relation action state + native Q/K/V/O + staged losses | E2 | timing audit 强，swap margin 仍负 | 保留 phase/relation 组件 |
| v11 | persistent bimanual closed-loop controller | E2 | EEF loss `0.693→0.070`，双向 phase 不稳 | 保留骨架，不续训 checkpoint |
| v12 | 21-state/20-edge IMT-Edge causal transport | E2 | 工程/梯度/FM成立，direction/isolation/destination 均失败 | step8 硬停 |
| v12.1 | Pairwise Relational Action，物理 ±12px pair | E2 | FM -5.19%，action response 方向近随机且弱/泄漏 | audit256 硬停 |
| v12.2 | per-arm native Q/K modulation | E2 | direction 51.19%、cosine -0.199、magnitude 2.51% | audit64 硬停 |
| Final-NativeBand | v3 raster + v8 native band + v12.2 canary | E2 | 128 exposures 稳定，direction 42.86%、局部性失败 | 永久结束 action 架构线 |
| v13 | frozen gated parent + late Q/V/O LoRA + mask/temporal/depth | E4/RGB8 | 0 black、像素微升；coverage -18.18%、共同 DTW +123.30% | RGB8 FAIL |
| v14 | 切换到公开 Official FlowWAM Stage-1 | E4 | dev-fast20 13/5/2；Breadth20 对 Custom-v15 18/2/0 | 成为主 incumbent |
| v15 | Official Stage-1 完整评测、Seed4、SeedVR2、bounded q/v LoRA | E4 | Seed4 完整15项均值最高；SeedVR2停止；LoRA step100未过门 | 冻结 `flow20 + seed4` |

## 4. 架构演进主线

```text
v1–v3   additive action raster 工程化
   ↓
v4–v7   双臂身份、时序、轨迹监督、SE(3)
   ↓
v7.1–v12.2  counterfactual、native attention、phase、persistent state、edge transport
   ↓
Final-NativeBand  最后一次高容量 action-causality 验证失败
   ↓
v13     停止 action 架构搜索，尝试视觉/物体/深度总分优化，但 RGB trajectory 退化
   ↓
v14     切换至公开 Official FlowWAM 原生 flow-conditioned 模型血统
   ↓
v15     冻结 Official Stage-1，完成 flow20、Seed4、完整15项和一次 bounded LoRA
```

真正发生质变的不是某个更复杂的 side branch，而是 v14 的模型血统切换：desired robot flow
直接进入所有共享 DiT block，与 RGB token 联合 self-attention；旧路线则一直试图在 Wan RGB
主干外围建立 action 控制权。

## 5. 旧 Wan 血统的关键裁决

### 5.1 唯一保留的旧 action parent

`clean-gated-step10` 保留为旧血统 reference 与故障回滚。它证明 support-gated raster 能在
matched RGB 上形成正信号，但并未达到 Official FlowWAM 的覆盖率和 DTW 水平。

### 5.2 为什么 v4–v12.2 没有继续

- 增强 arm identity 不等于 trajectory 正确；
- causal router 自身无泄漏，不代表 frozen Wan 内部不会重新混合时间；
- position/velocity probe 可用，不代表优化 probe loss 会改善 decoded RGB；
- counterfactual margin、FM、EEF representation 或 native attention gradient 非零，都不保证
  action 对最终输出形成正确方向、幅值、局部性和左右臂隔离；
- 连续多个版本在不同结构上重复出现“内部机制改善、rollout 不晋级”，因此不是继续调 LR、
  rank、block 或 exposure 能合理解决的问题。

### 5.3 v13 的最后一次旧血统总分尝试

v13 完成150/150 exposures、16/16 matched RGB、0 black，PSNR/SSIM略升；但官方 SAM3 RGB8
显示 frame coverage `-18.18%`，唯一共同有效样本 DTW `+123.30%`，且 object consistency
没有明确正证据。该结果明确拒绝“late Q/V/O LoRA + mask/temporal/depth”作为比赛主线。

## 6. v14：Official FlowWAM 血统切换

v14 的正式 dev-fast20 matched 结果：

| 指标 | 旧 incumbent | Official FlowWAM | 变化 |
|---|---:|---:|---:|
| failure-aware W/L/T | — | 13/5/2 | paired win 65% |
| valid episode | 11/20 | 13/20 | +2 |
| mean coverage | 0.135802 | 0.380247 | 约 +180% |
| common-valid mean DTW | 0.207973 | 0.051040 | -75.46% |
| black | 0 | 0 | PASS |

进入 v15 后，Official Stage-1 相对 Custom-v15 的 Breadth20 结果进一步扩大：`18胜/2负/0平`，
valid `9/20→19/20`，coverage `+0.54630`，8个共同有效 episode 的 mean DTW
`0.15388→0.06815`，改善约55.7%，black=0。至此 Official Stage-1 成为不可覆盖 Top-1。

## 7. v15 完整冲分闭环

### 7.1 Official Stage-1 seed1 完整 profile

dev-clean50 的50条正式生成均为121帧、640×480。完整15项揭示：Trajectory、JEPA、Depth、
Motion Smoothness 健康，主要提升空间集中在 Instruction、Interaction、Image、Aesthetic 与
Photometric。因此后续不再做 Motion FT，而只保留 Seed4、SeedVR2 和一次极短 LoRA。

### 7.2 Seed4 完整15项

| 指标 | Seed1 | Seed4 | Seed4 - Seed1 |
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

主要收益来自 Semantic、Instruction、Interaction、Motion Smoothness、Aesthetic 和 Trajectory。
Perspectivity `-0.008` 是最大单项回撤，但完整同口径15项仍为正。

### 7.3 SeedVR2 4090

三轮有界修复依次处理 Apex fused norm、VAE 显存和 RoPE 中间张量。最终第三次 repair 已越过
unfused norm 与 VAE memory-safe 路径，但首个 DiT forward 在 chunked RoPE
`torch.empty_like(values_hld)` 申请额外 `362 MiB` 时 OOM。按预注册上限停止，不做第四次
兼容实验，不进入候选。

### 7.4 v16 q/v LoRA（v15 内的唯一训练分支）

训练合同：Official Stage-1 parent；rank/alpha `8/8`；仅 DiT attention q/v；LR `2e-6`；
batch1；gradient checkpointing；121帧；640×480；flow20；官方 RGB objective；无 SAM、
Depth、JEPA、trajectory loss。

step50 matched Breadth20 通过：Interaction `+0.01`、Image `+0.001138`、JEPA `+0.000551`，
Instruction `-0.01`；valid `19→20`、coverage `+0.014198`、common-valid DTW
`0.067697→0.057559`、paired `11/9/0`、black=0。

step100 只保留 Interaction `+0.01`，Image `-0.001279`、Instruction `-0.01`，虽然 JEPA
`+0.002428` 且 trajectory 继续健康，但不满足三个主质量指标至少两项提升。最终 gate 为
`passed=false`、`decision=stop_v16_at_step100`，未运行 step200。step50 作为研究备选保留，
但缺少完整 clean50 15项，因此不替换 Seed4。

## 8. 关键工程修复谱系

| 问题 | 根因 | 最小修复 | 结果 |
|---|---|---|---|
| v14 推理 OOM | 纯推理未关闭 autograd | pipeline 前全局禁用 gradient | 越过23.44GiB OOM |
| Base action metric import | pinned checkout 缺兼容属性/生成 pyc 污染 | 隔离 compatibility layer、`python -B`、只恢复生成 pyc | Base/aggregate 闭环 |
| CLIP/SEA-RAFT 权重 | 下载/格式与官方 contract 不一致 | ModelScope优先、SHA门、SEA-RAFT strict=False预期键校验 | 权重闭环 |
| v16 staged resume | LoRA key 前缀不匹配，optimizer state device/dtype错误 | 240-key/120-pair/rank8精确恢复与 state 对齐 | step51/100真实完成 |
| VLM 伪 receipt | GPU7 OOM 行仍被旧 gate 接受 | 隔离旧证据、强制 GPU6、逐行 error/有限分检查 | 两侧VLM有效 |
| Seed4 Base GT 缺失 | `--detect_gt` 实为 `store_false` | 删除反向开关并加回归测试 | Base完整完成 |
| Seed4 JEPA 卡死 | 长 TMPDIR + socket名超过 AF_UNIX 108字节 | 仅 JEPA 使用 `/tmp`，本地/远端32/32测试 | JEPA与aggregate完成 |

## 9. 最终收据与哈希

### 9.1 Seed4

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
- package manifest SHA256：`2793a0cc1db33cb6885f36c7706f550702487aae381036571e9781c42e11ba17`
- VLM：50/50、error=0；JEPA：`0.9567664861679077`

### 9.2 v16

- step50 LoRA SHA256：`37ab801d3f04125a2dc0fe1ce6924dd90d84577c518bd18812a0806c9c1fff42`
- step50 gate SHA256：`6447d1cc5f49231b1ce5918b77810ff3b79eb9f3bd8cdf0f63dc43411c4eeece`
- step100 LoRA SHA256：`445b811e301a7e70658a4c8179ba64da38ec52b222b87465a1ebf079719bb285`
- step100 gate SHA256：`1ed2c984f7692bd28c0c62ca257f70e48fe183126fbe1fa0d69bb4b377ed4461`

## 10. 可复用资产与明确淘汰项

### 可复用

- v3 raster/pose/support/causal packing 与 strict lineage；
- frozen gripper probe 和 observability/gate 工具；
- v8 native attention band、v10 phase/relation、v11 persistent controller 的诊断实现；
- physical action-pair、counterfactual 和 final-output response 审计；
- Official FlowWAM renderer、flow cache、checkpoint loader、GPU lock、receipt 与 evaluator；
- v16 fail-closed LoRA resume/hotload 与 matched quality gate；
- Seed4 完整官方评测 package。

### 不得作为发布或 warm-start parent

- v2 Stage2 长训、v4、v5、v6 correction、v7 gate-only、v7.1/CF、v9、v12、v12.1、v12.2、
  Final-NativeBand、v13 exposure150；
- v16 step100；
- SeedVR2 当前 4090 兼容分支；
- 任何 invalid GPU7 VLM 旧证据。

## 11. 详细报告索引

| 范围 | 文档 |
|---|---|
| v1–v14 详细架构谱系与 v15 早期日志 | `reports/2026-08-19-wan-action-v1-v11-architecture-experiment-lineage.md`（历史文件名未随内容扩展） |
| v4 | `reports/v4-clean-data-scale-preparation-20260817.md`、`reports/v4-step3600-matched-fast20-assessment.md` |
| v5 | `reports/v5-temporal-motion-correction-20260818.md` |
| v6 | `reports/v6-gripper-trajectory-supervision-20260818.md` |
| v7/v7.1 | `reports/2026-08-18-v7-se3-single-gpu-mechanism-probe.md`、`reports/2026-08-18-v71-se3-geometry-lora-mechanism.md`、`reports/2026-08-18-v71-se3-geometry-lora-cf.md` |
| v8 | `reports/2026-08-18-v8-direct-action-band.md` |
| v9 | `reports/2026-08-19-v9-phase-locked-action-cross-attention.md` |
| v10 | `reports/2026-08-19-v10-staged-loss-training-report.md` |
| v11 | `reports/2026-08-19-v11-stagea-detailed-experiment-report.md` |
| v12 | `reports/v12-imt-edge-experiment-report.md` |
| v12.1 | `reports/2026-08-19-v121-pra-experiment-report.md` |
| v12.2 | `reports/2026-08-19-v122-nqm-native-qk-control-experiment-report.md` |
| Final-NativeBand | `reports/2026-08-19-final-native-band-experiment-report.md` |
| v13 | `reports/2026-08-19-v13-scoreboost-experiment-report.md` |
| v14 | `reports/2026-08-20-v14-flowwam-wa2-experiment-report.md` |
| v15 全执行日志 | `reports/2026-08-20-v15-flowwam-balanced-campaign.md` |
| 最终冲分与上线结论 | `reports/2026-08-23-worldarena-track1-final-score-report.md` |

## 12. 历史边界与当前剩余动作

仍不启动：更多 seed、prompt grid、Motion FT、SeedVR2 第四次修复、v16 step200 或新的 action
生成架构。2026-08-24 新批准的 Adaptive Candidate System 不覆盖冠军、不扩 seed，只利用
seed1/seed4 已知互补性建立 zero-overlap 路由/选择证据；它是此前“仅剩上线”结论的明确后续修订。

当前剩余动作是：

1. 完成 200 个 zero-overlap episode 的 seed1/seed4 双候选语料（共400条视频）及 latest-WA2
   raw/corrected 双分数；
2. 仅用160条训练 Logistic Regression input router 与 post-generation selector，并用隔离40条
   holdout 按 corrected uplift 与 paired bootstrap 裁决；
3. 并行完成唯一固定模板的 action-to-text Breadth20；只有预注册门通过，才允许最多一个新候选
   使用 clean50；
4. 在新系统未通过完整裁决前，部署回滚合同始终保持 Official Stage-1 / flow20 / seed4；最终仍需
   执行服务健康检查、提交 smoke 与逐文件 MP4 SHA256 一致性检查。

## 13. 最终 seed4 × v16-step50 inference-scale 裁决

最后一个证据缺口已闭合：固定 step50 LoRA SHA256
`37ab801d3f04125a2dc0fe1ce6924dd90d84577c518bd18812a0806c9c1fff42`、seed4、flow20、
20条、121帧、640×480，只改变 LoRA inference scale。三组均有独立 generation receipt、
video validation receipt、20/20 SAM3、无错误有限 VLM/Base/JEPA 和 black=0。

| scale | 快速九项均值 | 相对 Official+seed4 | DTW | JEPA | Instruction | Interaction | Image | 门禁 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Official+seed4 | 0.716048 | — | 0.051752 | 0.944844 | 0.730 | 0.740 | 0.517448 | incumbent |
| 0.50 | 0.712895 | -0.003153 | 0.050372 | 0.943039 | 0.700 | 0.740 | 0.519164 | reject |
| 0.75 | 0.712950 | -0.003098 | 0.060261 | 0.943600 | 0.710 | 0.740 | 0.518746 | reject（trajectory） |
| 1.00 | 0.715295 | -0.000753 | 0.053727 | 0.944762 | 0.720 | 0.750 | 0.519879 | reject |

scale1.0 最接近基线，并保留 Interaction `+0.01`、Image `+0.002431`、Aesthetic
`+0.002091`，但 Instruction 与 Perspectivity 均 `-0.01`，九项均值仍为负；scale0.5/0.75
回撤更大。因此 selection receipt 明确为 `winner=null`、`decision=freeze_official_seed4`，按合同
不运行候选 clean50，冠军永久冻结为 Official Stage-1 / flow20 / seed4，完整15项均值为
`0.6586597567502485`。

决定性新增 receipt：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step50-seed4-scale050/quality-gate.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step50-seed4-scale075/quality-gate.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/breadth20-step50-seed4-scale100/quality-gate.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/v16-qv-lora-clean1785/seed4-scale-selection.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/automation/seed1-seed4-oracle.complete.json
/data/di/worldarena2_track1_20260815/runs/flowwam-official/automation/track1-output-encoding-audit.complete.json
```

三个 gate SHA256 依次为 `5474c52382897bae5610a47f6239ce5cb88446f939c3013b1534c81b360e5aa3`、
`dfbe5965a677caf4c6b39c777c635fa6f32bf1d091058a75be7a50efada23d38`、
`4b2e7d8e40e662909b556694239cf53c38ec17c9f13f8c16bb0e158a8080c792`；selection SHA256 为
`398d60330e96cae5952e83f794d3bb290685007d4b76a3d3aa97f3007c2e1513`。

零生成成本 oracle 上限为 `0.6683917311343808`，高于 seed4
`+0.009731974384132314`，seed1/seed4 各赢25条。原 inference-scale 轮次曾按旧“中间带”规则
不投资 selector；该结论已被 2026-08-24 批准的 zero-overlap Adaptive Candidate System 实验合同
取代。编码审计确认生成器单次编码、package 不转码；由于没有
在线 Track1 服务可做二次编码 A/B，上线硬门是返回/打包 MP4 SHA256 必须与 generator receipt
逐文件一致。

快速 gate 首次汇总曾错误读取 raw Photometric（`1.568996`）而非官方 normalized per-video
均值（`0.216574`）。最小修复改为从正式 `generated_results.json` 读取 normalized 值，并新增
回归测试；本地与远端均 `4/4 PASS`，没有放宽门禁或重算其他指标。

一句话收口：**v1–v13 证明了外接 action controller 的工程上限，v14 通过模型血统切换获得
决定性 rollout 提升，v15 用完整15项、有界训练和最终 seed4×LoRA-scale 正面对比收敛到
Official Stage-1 flow20 + seed4；它现在是新自适应系统必须击败、且不可覆盖的回滚基线。**

## 14. Adaptive Candidate System 扩展：实时状态

截至 2026-08-24 20:16（Asia/Shanghai）的远端复核如下。这里区分“历史实验完成”和“新主线
完成”：前者已闭环，后者尚未开始生成主体，因此不得从空闲 GPU 或 PID 结束推断完成。

| 工作项 | 当前状态 | Receipt / 缺口 |
|---|---|---|
| 历史冠军 | 完成 | Official Stage-1 / flow20 / seed4，raw 15项均值 `0.6586597567502485` |
| Adaptive 设计合同 | 已批准 | `docs/superpowers/specs/2026-08-24-flowwam-adaptive-candidate-system-design.md`；提交 `4280905`、`9b6cd42` |
| latest-WA2 scorer | 本地实现与测试通过 | 本地 scorer 相关测试 `8 passed`；远端 GT-reference motion receipt 尚缺 |
| 200×2 候选语料 | **0/400** | `/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/` 尚无 manifest/generation receipt |
| Selector holdout | 未开始 | 缺 scorer、router、post-generation selector 与 task-stratified bootstrap receipt |
| Action-to-text | 未开始 | 固定模板 Breadth20 尚无 generation/quality receipt |
| Score-directed self-training | 未解锁 | 只允许使用160条训练分区中的高置信 winner；40条 holdout 永不进入拟合 |
| v17 native-640 数据 | 下载中但为条件分支 | 7/100 archives、约8.5GB；无 `download.complete.json`，且正常冲分窗口不自动解锁训练 |

冻结裁决门以 corrected score 为主：holdout corrected uplift 至少 `+0.003`，task-stratified paired
bootstrap 10,000次、单侧90%下界大于0，12项 non-motion 不回退，每个 seed 至少选择5/40；最终
clean50 还要求 corrected mean 高于冠军 `+0.003`、12项 non-motion 均值不低于冠军 `-0.001`，
Trajectory 与 JEPA 各自不低于冠军 `-0.01`。Accuracy 只报告，不再作为主门。

本次复核 GPU0/GPU4/GPU5/GPU6 空闲；GPU1/GPU2/GPU3/GPU7 有其他用户进程，未触碰。数据盘
剩余约839GB。后续所有完成判断继续以 receipt、artifact count、hash 和有限值验证为准。

### 14.1 Adaptive 四卡执行已启动（21:32 CST）

原 `0/400` 是 20:16 的历史快照，现已被正式 manifest 和四卡启动证据更新：确定性选择200条，
train/holdout=`160/40`，GPU0/GPU4/GPU5/GPU6各50条；同一 episode 固定按 seed1→seed4 紧邻
生成，每卡只加载一次冻结 Official Stage-1。manifest SHA256 为
`4b147c15ba6b9fad45d66cdc542e57bbe9a967c0cd8d05a5f267cad7c702a18e`，源1785清单 SHA256 为
`6bcc85830fa2c6ccb45b2c3f7f593102674efce6a76a7a6e5d8871fcccdbae35`。dev-fast20 与
dev-clean50 排除清单 SHA256 分别为 `aaf9f4268a78e3ed160908780809ba9c23f12946176072787048d21ed7f50864`
和 `2793a0cc1db33cb6885f36c7706f550702487aae381036571e9781c42e11ba17`。

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/manifests/selector-200.complete.json
GPU0 PID 1232329
GPU4 PID 1232332
GPU5 PID 1232339
GPU6 PID 1232348
```

本地与远端聚焦合同测试均为 `42 passed`；四个 shard 的 dry-run 均确认50 episode/100 video、
121帧、640×480、flow20、seed顺序 `[1,4]`。当前仍不得写成生成完成：缺少四份
`paired-generation.complete.json` 和400条逐视频 SHA/黑帧收据。SeedVR2 SP2 已由既有 stop receipt
终止（720×1280 all-to-all OOM；唯一640×480 fallback RoPE OOM），不会重跑。

### 14.2 四卡持续运行与下游自动衔接（进度 45/400）

最新逐视频完成数为 GPU0 `11/100`、GPU4 `12/100`、GPU5 `10/100`、GPU6 `12/100`，总计
`45/400`。四个精确 PID `1232329/1232332/1232339/1232348` 均存活，四卡利用率均100%、显存约
15.3GiB，四份 generation log 的错误标记均为0；磁盘剩余约814GiB。完成门仍是四份
`paired-generation.complete.json`，不是这些 PID。

CPU action feature 已先行闭环：200行、train/holdout=`160/40`，输出 JSONL SHA256 为
`4a09c3f5fa1c593f111efd62d9a490c075455e576165f7d315633d09fc19e0d8`，receipt 为：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/features/action-features.complete.json
```

下游 eval queue PID `1325650` 已启动但在四份最终生成 receipt 前不占 GPU；之后按独立锁自动
执行 seed1/seed4 两条200视频 lineage、四卡 Base/VLM/JEPA、CPU aggregate、匹配 GT 三项 Motion
correction，以及 input router/post-generation selector 的160/40隔离裁决。正式评测固定为
step-0740/0750，不覆盖 step-0600 冠军。本地和远端扩展聚焦测试均为108项通过。

### 14.3 监控路径修正与定时上报（进度 89/400）

2026-08-24 22:35 CST 现场复核发现，正确监控文件是各 shard 的 `generation.pid` / `generation.log`
以及根目录下的 `evaluation/eval-queue.pid` / `evaluation/eval-queue.log`；此前一次状态脚本误读
`generate.pid` 与 `eval/`，造成空 PID 假象，但没有重启或干预任何作业。按正确路径复核后，四个
精确 PID `1232329/1232332/1232339/1232348` 和 eval queue PID `1325650` 全部存活，日志持续刷新，
错误标记均为0。

逐视频 MP4/receipt 当前为 GPU0 `22/100`、GPU4 `23/100`、GPU5 `21/100`、GPU6 `23/100`，合计
`89/400`；磁盘剩余约812GiB。四份 `paired-generation.complete.json`、combined receipt、两条
正式 aggregate、latest-WA2 scorer 和两个 selector holdout receipt 仍缺，因此本轮尚未完成。
现有 `worldarena-final-score-pipeline` heartbeat 已从过期的暂停审阅状态修正为 ACTIVE，每15分钟
按正确路径主动上报，不另建重复监控。

### 14.4 400/400 生成闭环与评测 staging 修复

四个 shard 已全部形成终态收据：GPU0/4/5/6各100视频，合计200个零重叠 episode × seed1/seed4
=400视频；逐视频合同均为121帧、640×480、black=0、独立SHA256，Stage-1 checkpoint SHA256为
`e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`。四份收据SHA256依次为
`40a068df…af3b3`、`f29acf16…83134`、`e242a5a2…744be`、`674cfb05…ec406`。

首次 eval queue 在 staging 时暴露出输入软链接白名单错误：代码错误地要求 dataset HDF5 的真实
目标位于 adaptive run root，而批准的数据实际位于同一 artifact root 的 `datasets/`。修复将生成
视频继续锁定在 adaptive root，仅把输入目标白名单扩到显式 `--artifact-root`；本地和远端聚焦
回归均为 `4 passed`。同时发现预定 step-0740 已被既有20条 self-only lineage占用，未覆盖该目录，
而是将两条新评测血统无损迁移到空闲 step-0750/0760。修复后的 eval queue PID为 `2185267`。

合并血统收据已验证200/200、seed1/seed4各200，SHA256为 `f6785cd3…9779344`：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/combined/combined-lineages.complete.json
```

当前进入 package/Base/VLM/JEPA/aggregate/latest-WA2 scorer 阶段；selector holdout与最终决策仍未
形成receipt，冠军继续冻结为Official Stage-1 / flow20 / seed4。

### 14.5 Adaptive Base 端口隔离修复与 Action-to-text 输入冻结

step-0750/0760 的 package receipt 均已形成。四卡并行 Base/VLM 时，step-0750 Base 在官方
`evaluate.py` 的分布式初始化处因两条 Base 共用默认 `MASTER_PORT=29500` 退出；step-0760 Base
及两条 VLM 未受影响并继续运行。最小修复在 `run_phase` 中按物理 GPU 固定
`MASTER_PORT=29500+gpu`，远端聚焦回归 `4 passed`、shell syntax 通过。恢复器 PID `3382065`
只等待三个已有 owned phase 自然结束，随后补跑缺失 receipt；没有重启或复制活跃作业。

唯一固定 action-to-text 模板的 CPU 输入准备也已闭环：20条原 instruction 均保留，仅附加由 action
确定性提取的主/静止手臂、先动手臂、夹爪顺序与粗阶段，不使用 LLM、不引入新物体名。远端测试
`4 passed`，输入收据 SHA256 为 `82c0a7323d9bd5ae0cee15e5214738c70bd351e315290581596992d4d6e3358d`：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/action-to-text/input/action-text-input.complete.json
```

该分支尚未占用 GPU、未生成候选，也未形成质量门收据；正式冠军不变。

### 14.6 Adaptive 正式评分进度与 Flow oracle 终态（2026-08-25 09:17 CST）

生成侧保持 receipt-gated 完成：GPU0/4/5/6 各100视频，合计400/400；四份 paired receipt、
逐视频 SHA、121帧、640×480、black=0 与合并血统 receipt 均未发生变化。当前正式评测只占用
批准的 GPU4/5/6：step-0760 Base 约73/200，seed1 VLM 173/200，seed4 VLM 185/200，日志持续
刷新且当前段无新错误；GPU0 空闲，GPU1/2/3/7 的其他用户进程未触碰。数据盘尚余约713GiB。
恢复器 PID `3382065` 仍按原计划等待三条 owned phase 结束，只补跑此前端口冲突缺失的
step-0750 Base；action-to-text queue PID `3522020` 等待 step-0760 Base receipt 后才启动唯一
固定模板，不会创建重复生成。

零新增生成的 flow16/20/24 oracle 已完成证据审计。现有三条 Breadth20 生成血统与旧
ATR/SAM3 选择结果存在，但没有三套严格 common-episode 的 corrected-15 分数；合同同时禁止补跑
flow16/24，因此该分支按规则写入 `close_missing_prerequisites_no_new_generation`，不以旧 DTW
替代 corrected oracle：

```text
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/flow-oracle/flow16-20-24.complete.json
SHA256 15430fea559c75bdaa925eaf6b98f5f5f3c8b7096e5eec2717909f73b24ec26c
```

当前仍缺 Base/VLM/JEPA/SAM3/aggregate/latest-WA2、selector/cascade、action-to-text 质量门、
两条训练分支和最终决策终态收据；冠军继续冻结为 Official Stage-1 / flow20 / seed4。

### 14.7 Action-to-text GPU 参数修复并正式启动（2026-08-25 09:30 CST）

step-0760 Base 已完成200/200并形成 receipt，SHA256 为
`51e405da36ad679a0e938bb7da7f0e2bb01e84549a36dcb33679a296978ac686`。固定模板队列随后按门启动，
但在生成前 fail-fast：shell 已设置 `CUDA_VISIBLE_DEVICES=4`，却未向 Stage-1 入口传
`--physical-gpu 4`，入口使用默认值6并拒绝运行。该故障未生成任何 MP4，也未占用 clean50。

最小修复只在 action-to-text 启动器补传 `--physical-gpu "$GPU"`，先新增失败回归断言，再验证
本地与远端均 `4 passed`、shell syntax 通过，脚本 SHA256 为
`58a3e810f97119bb7cc6921c78d72e777d49c3816bd634b489029a51a35bcfae`。失败日志已保留，且只重启
该 owned queue：PID `3772998`、runner PID `3773008`，GPU4 UUID
`GPU-651bd630-9a1d-c4c2-1089-8501d6ef1051`，约16.5GiB/100%，已经进入第1/20条 DiT 去噪，
当前无新错误。两条 VLM 仍在179/200与190/200，恢复器继续只等待并补缺失 seed1 Base。

### 14.8 Seed4 VLM 正式收据与 P1 持续生成（2026-08-25 09:58 CST）

seed4 的200条 VLM 已自然完成并写入正式 receipt，SHA256 为
`4606b8ff85cbadb6a9b34c3afa912013795bbf115539e573dcfd46c53edf37c2`。输出汇总严格复核为
200/200 行、每行三项 VLM 指标，共1200个数值字段全部有限，`error=0`，不是仅以进程退出或
receipt 文件存在判定完成。GPU6 已释放至约6MiB/0%，未启动重复任务。

同一时刻 seed1 VLM 为192/200，日志继续刷新且无错误，预计约18分钟完成；恢复器 PID
`3382065` 仍等待该 owned phase 自然结束后，仅补跑缺失的 step-0750 Base。固定模板
action-to-text 已生成9/20个 MP4，GPU4 的 runner PID `3773008` 与 generation lock 归属一致，
日志持续刷新且无错误，预计约30分钟完成。磁盘剩余约710GiB；selector、训练与最终决策门仍未打开，
冠军继续冻结为 Official Stage-1 / flow20 / seed4。

### 14.9 双侧 VLM 闭环与 Base750 受控恢复（2026-08-25 10:16 CST）

seed1 VLM 已完成200/200并形成正式 receipt，SHA256 为
`2bd5d98a8f29ac25f36c6603c760fdd6f5902c2e5a90b4aacb70c62d336476a2`。输出汇总复核为200行、
每行三项指标、共1200个数值字段全部有限，`error=0`、空 metrics=0。至此 seed1/seed4 两侧 VLM
均具备有效终态收据，GPU5/6 已释放。

恢复器随后按原合同仅启动此前端口冲突缺失的 step-0750 Base：父 PID `3929854`，物理 GPU0
UUID `GPU-66b5e202-9179-0f61-c0dd-34d5599ffde2`，eval lock 持有者与恢复器/子链一致；没有重复
step-0760 Base 或任一 VLM。action-to-text 同时推进至15/20，GPU4 runner 和 generation lock
保持一致、日志无错误。Base750 尚处初始化阶段，需首批样本后估算 ETA；P1 预计约15分钟完成生成。

### 14.10 Action-to-text 20/20 与视频验证链补全（2026-08-25 10:35 CST）

固定模板 action-to-text 已完成唯一允许的 seed4 Breadth20：20/20 MP4，生成 receipt SHA256
`1270106c53855e638020445afe5f0ab69fd37233c583a844571dfcc2990a7ddc`。生成结束后发现队列把该
receipt 当成整条 P1 终点，未执行计划要求的 video validation。根因是队列只有生成调用，没有任何
验证器或验证收据条件，不是 GPU/视频故障。

按测试先行只补齐该边界：新增真实 MP4 形状/黑帧/哈希测试并先观察缺失接口失败，再实现 CPU
验证器并把队列终点改为 `video-validation.complete.json`。本地组合测试9项、远端聚焦测试5项
全部通过，四个同步文件 SHA 一致；只恢复已退出的 CPU 队列，复用现有20个视频，没有重新生成。
最终验证收据 SHA256 为 `20ed7a9a2650254abecc0b9ae4cb551ba1773f8dce7e28626c7a7503773cefff`：
20/20、121帧、640×480、black=0、20个唯一视频 SHA，且绑定上述生成 receipt。

P1 仍未通过质量门；后续必须完成冻结的 VLM/JEPA/Trajectory/raw/latest-corrected 比较，不能以视频
验证代替。step-0750 Base 继续在 owned GPU0 正常运行，日志新鲜且无错误；GPU4 已释放。

### 14.11 P0 持续评分与 P1 端口隔离恢复（2026-08-25 11:35 CST）

本次按远端真实路径重新核对收据，避免把 `adaptive-candidate-system/evaluation` 下未命中的路径误判为
正式收据丢失。正式收据位于 `official_track1_eval/checkpoints/step-*/dev-clean-50/receipts`：step-0750
已有 package 与 VLM（VLM SHA256 `2bd5d98a8f29ac25f36c6603c760fdd6f5902c2e5a90b4aacb70c62d336476a2`），
Base 正在运行，JEPA/aggregate 尚缺；step-0760 已有 package、Base 与 VLM（Base SHA256
`51e405da36ad679a0e938bb7da7f0e2bb01e84549a36dcb33679a296978ac686`，VLM SHA256
`4606b8ff85cbadb6a9b34c3afa912013795bbf115539e573dcfd46c53edf37c2`），JEPA/aggregate 尚缺。
恢复器 PID `3382065` 的唯一活跃子链为 `3929854 -> 3940891`，物理 GPU0 UUID
`GPU-66b5e202-9179-0f61-c0dd-34d5599ffde2`；当前 Base 子指标进度75/200，日志新鲜且未发现
Traceback/OOM/RuntimeError。GPU5/6 保持空闲，供 Base 完成后的两侧 JEPA 并行使用。

Action-to-text 的 baseline/candidate package（step-0751/0752）均已形成 receipt，但 step-0751 Base
首次执行在40/40轨迹预处理后因默认 `MASTER_PORT=29500` 与主评测冲突而退出，错误为
`EADDRINUSE`。该故障不涉及视频、模型或指标合同。旧 PID `4083982` 已确认退出；在精确核对 GPU4
UUID、空闲进程和端口后，只恢复该 owned queue，并显式设置 `MASTER_ADDR=127.0.0.1`、
`MASTER_PORT=29504`。新 PID `29400`、评测子 PID `31494` 已在 GPU4
`GPU-651bd630-9a1d-c4c2-1089-8501d6ef1051` 运行，约16.6GiB/94%；环境回读确认端口为29504，
step-0751 semantic_alignment 子指标已推进至14/20，恢复日志无新错误。失败日志保留，没有重跑
20条生成，也没有触碰 GPU1/2/3/7。

生成与训练前置资产仍有效：400/400 paired 视频、四份 shard receipt、combined lineage SHA256
`f6785cd3ff388d164bfdced0543bde542d2cdacc597fdaf57f625b4599779344`、action features SHA256
`4bbb5eb15b0759c7049b67a39d58d28a544dec5661f66fe11ed88e610349fc3e`、P1 video validation SHA256
`20ed7a9a2650254abecc0b9ae4cb551ba1773f8dce7e28626c7a7503773cefff`。Winner-SFT/Preference-LoRA
的 train-only 高置信标签、共享 timestep/noise 与 q/v-only 合同实现已通过本地聚焦测试12项；但在
selector 的160条训练分数闭环前不会启动训练，也不会打开40条 holdout。磁盘尚余约703GiB。

当前无人工阻塞。接下来主队列完成 step-0750 Base 后并行两侧 JEPA 与 GT-motion，再做 aggregate、
latest-WA2、router/selector/cascade 和一次性 holdout；P1 独立完成 step-0751/0752 全指标与质量门。
预计 P0/P1 首轮决定还需约3--5小时；若高置信样本不足则6--10小时可收口，若两条 LoRA 均解锁并
需要 matched Breadth20/clean50，完整最终决策预计10--22小时。正式冠军继续冻结为 Official
Stage-1 / flow20 / seed4，`final-decision.complete.json` 尚未形成。

### 14.12 P1 incumbent Base 正式收据（2026-08-25 11:55 CST）

端口隔离恢复已通过真实终点验证：step-0751 incumbent Base 完成20/20 package validation并形成
正式收据，SHA256 `83bde0774a3e42fb993be4c0277dbe97316025e5548821c7249b2b58a85ebb1e`。
同一 queue PID `29400` 未退出或重复启动，已自然转入 step-0751 VLM，当前3/20；GPU4 子 PID
`68436`、约19.0GiB，日志无错误。主 step-0750 Base 同时在 GPU0 正常运行，当前子指标136/200；
GPU5/6仍空闲。当前阻塞只是正式评分计算时间，P1仍需 VLM/JEPA/aggregate、candidate step-0752
全阶段和 corrected quality gate，冠军不变。

### 14.13 四卡加速调度与 P0 JEPA 闭环（2026-08-25 12:19 CST）

step-0750 Base 已完成并形成正式 receipt，SHA256
`2421f57c15303540d3f3a5a735a509f0c7c688db3472c6a2f1fe96c81f55430f`。为利用空闲卡，先以失败
回归锁定并修正主队列“已有 phase receipt 仍执行”的反向条件，同时为 GT-motion 增加同等跳过保护；
本地与远端 `tests/test_adaptive_staging.py` 均4项通过，队列脚本 SHA256
`c5524f4a8a0c1c30d15a0c97383e095b0376039b0affea68abf93c08338c7987`。

同步脚本时旧 live shell 从被替换文件继续读取，导致一次 `ggregate: command not found`，另一侧
aggregate 按设计因缺 JEPA 拒绝；两份失败日志已另名保留，未产生 aggregate receipt 或错误结果。
随后在精确 UUID/空闲/锁核对后，把两侧 JEPA 分别放到 GPU5/6；二者均完成200/200并形成 receipt：
step-0750 SHA256 `8073d9d35962bb29a8a0fdff6dcaf33449814d0b5fbe71ec6915efcb6398417e`，
step-0760 SHA256 `e64c36c7b9943ef0a254446e24b3e80eb90a48a2bc17bd61a0bf21fcb6fd22ea`。

当前 GPU0 正运行 GT-motion（160/200 dynamic_degree 子指标），GPU4 正运行 P1 incumbent VLM
（14/20），GPU5 提前运行 P1 candidate step-0752 Base，GPU6 已在 JEPA 收据完成后释放。主恢复器
PID `191763` 只等待 exact owned GT PID 结束和 receipt 出现，随后用修复后的队列续接 aggregate、
latest-WA2 与 selector，不会重复 Base/VLM/JEPA/GT。当前三条活跃日志均无错误；P0 首轮结果预计
约30--60分钟，P1质量门约1--1.5小时，联合首轮决定收敛到约1--2小时。

### 14.14 P0 正式决策闭环与 P2 解锁（2026-08-25 12:55 CST）

GT-motion 三项各200条计算完成后，收尾程序因 WorldArena 按数据集目录名输出
`gt-reference-dataset_results.json`、而包装器固定寻找 `gt_reference_results.json` 而失败；600项
GPU结果未损坏。已按 RED→GREEN 增加 canonical materialization 与 `--finalize-existing`，本地聚焦
测试20项、远端19项通过；未重算GPU指标即形成 GT receipt，SHA256
`3cb9817db8f1c23b96edc6f0586fd735d89f52c6e74df602b553c2f92e7f3017`。随后发现 Semantic
Alignment 单点因FP16归一化为 `1.000977`；仅加入0.001舍入容差，保留 pinned raw 值且仍拒绝
`1.01`，路由与P1终结器使用同一合同。

两侧 aggregate、latest-WA2、输入路由、post-generation selector 和冻结40条 holdout 已一次性闭环。
seed1/seed4 raw15 为 `0.6696694547460158` / `0.6695985385719183`，corrected15 为
`0.6666362780797835` / `0.6665121650242082`。输入路由失败（corrected uplift
`-0.00038138157`）；post-generation selector 正式通过：corrected uplift `0.003067329787`、
one-sided 90% lower `0.0028273658672715294`、raw uplift `0.0030790212375193136`、non-motion
uplift `0.003942522790525055`、选种16/24。policy receipt SHA256
`aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`，允许唯一 clean50 候选。

train-only高置信筛选的只读预计算得到 seed1/seed4 eligible 71/58，按种子平衡后116条，超过40总数
与每种子10条门槛，因而 Winner-SFT 与 Preference-LoRA step25 均获准进入正式数据收据与
production-shape smoke。P1 incumbent 已有 VLM/JEPA/aggregate 收据；原质量队列进入 candidate Base
后与提前GPU5预跑短暂重叠，已只终止 owned GPU5 duplicate，保留原GPU4 queue。12:55 GPU4 candidate
Base正常、日志无错误，GPU0/5/6空闲。P1首轮质量门预计约45--70分钟；冠军在 clean50 最终门前仍
冻结为 Official Stage-1 / flow20 / seed4。

### 14.15 P2 正式伪目标收据与 P1 实时质检（2026-08-25 13:05 CST）

训练数据门已从只读预计算升级为正式、可复验的 `training/pseudo-targets.complete.json`，SHA256
`4b9d270a94b13ba0192d7d8b6b416e815efa721c0ed83bf1ac20a0f5543ab966`。收据只读取冻结 manifest
中160条 `train` 分区，明确排除40条 holdout；raw/corrected赢家一致且 corrected margin>=0.003 的
eligible 为 seed1 71条、seed4 58条，按任务和种子确定性平衡后各58条、合计116条。116条均绑定
赢家/输家 MP4 的绝对路径与 SHA256，且 episode ID 唯一。构建器本地13项、远端12项聚焦测试通过，
因此 Winner-SFT 与 Preference-LoRA 仅解锁到 production-shape smoke，尚未声明任何训练 checkpoint。

P1 原质量队列 PID `29400` 仍为唯一活跃队列，candidate step-0752 Base receipt 已形成，现转入 VLM；
13:05进度2/20，单条约138秒，GPU4 UUID和约19.5GiB显存与 owned 子进程一致，日志无
Traceback/OOM/RuntimeError。GPU0/5/6空闲，但在自注意力 q/v-only、LR1e-6、step25、optimizer/hash
收据与共享噪声偏好损失的 production 实现通过 smoke 前不会启动训练。按当前速率，P1质量门预计
约40--60分钟；训练分支仍取决于严格 smoke，不以数据门通过代替训练完成。

### 14.16 两条 P2 production-shape smoke 正式通过（2026-08-25 13:43 CST）

Winner-SFT 与 Preference-LoRA 已分别在 owned GPU0/GPU5 完成真实 batch1、121帧、640x480 的
前向/反向/optimizer smoke，而非结构桩。两份 smoke receipt SHA256 分别为
`947e3bcb52b9a61f15cae38c838ee581c937325292141c049c4390500827f940` 与
`b5a75c5f6f2c6c889a178cd0b12cfc455860cb4d34f84e06080cc973bc7007d0`；峰值 allocated 显存为
14.52/14.65GiB，均低于22GiB门。trainable whitelist逐项确认30个DiT block中仅
self-attention q/v LoRA A/B共120个张量，无cross-attention、K/O、MLP、FlowStream、T5、VAE、
modulation或native weights。

### 14.17 P1终态关闭与Winner-SFT step25落盘（2026-08-25 13:52 CST）

Action-to-text 的20条视频已全部通过121帧、640x480、black0验证，视频验证receipt SHA256为
`20ed7a9a2650254abecc0b9ae4cb551ba1773f8dce7e28626c7a7503773cefff`。其质量门receipt SHA256为
`67ccea4098b8093e80d164a248a586224a999f0a68e9b6e79a5f16181c9bc342`，终态为
`close_action_to_text_failed_gate`：raw/corrected均提升约+0.0074/+0.0077，Interaction提升+0.03，
但Instruction增量为0且corrected motion下降0.000413，因此不放宽门槛、不进入最终候选。

Winner-SFT已完成25步，正式训练receipt SHA256为
`0586df9004f4bc977fbeb568847ddb4bd309d7c7c242f23d22ea7762f8fda999`。LoRA/optimizer SHA256分别为
`fe9ed6fc61a0da9e85dbc661d07024e8ab7e3f743abb9003c0e35058a1bfbd34`与
`fadeb4f74b26cbf48325f7b155724ccdbe6d14ac5f8d4219330611d16935588a`，已独立复核。首次Breadth20
推理在生成前因旧过滤器要求含cross-attention的120个LoRA对而安全失败；新checkpoint按合同只有
self-attention q/v的60对（120个A/B张量）。已新增显式`self-qv` source-layout合同并经本地55项回归，
失败任务未产生视频，随后只恢复缺失的seed1/seed4验证。

### 14.18 Preference-LoRA step25落盘并四卡并行配对验证（2026-08-25 14:14 CST）

Preference-LoRA已完成25/25，正式训练receipt SHA256为
`623af5d3e9d888b583fc600794aaec7e4b373ed972ac233b0a5fa942cae03e51`。LoRA/optimizer SHA256分别为
`09b4317db5ce845f222e6e5dff81511b941281fe616d32b490834c59273e60a9`与
`1bac48245b4aaf79071b91907794e717f0b954dee2471f30c024d432b2fd96e0`，已独立逐文件复核；25步
winner/loser均绑定同一episode、timestep与noise SHA，loss有限且无GPU错误。Winner seed1/seed4在
GPU0/GPU6分别完成5/20与6/20；Preference seed1/seed4已在空闲且锁/PID/UUID核验通过的GPU4/GPU5
启动。四条生成只使用seed1/seed4与冻结Breadth20输入，不改变clean50分配。

### 14.19 P0 frozen selector clean50终态门通过（2026-08-25 14:34 CST）

不等待P2，已把冻结post-generation selector直接应用到既有seed1/seed4 clean50；没有重新生成、
重训或按clean50结果修改策略。执行器硬性拒绝历史100行拼接文件，严格使用最新版WA2回执钉死的
两份50行raw CSV、同一GT-reference motion caps、冻结27维generated-only schema与policy/model
SHA。50条选择为seed1 27条、seed4 23条，全部逐条复核生成receipt SHA、返回MP4 SHA、121帧、
640x480与black0。

终态门全部通过：corrected mean `0.6592060841461832`，相对Official+seed4 corrected冠军
`+0.004216075078498971`；raw15 `0.662720295147882`（`+0.00406053839763354`）；12项non-motion
`+0.005244083882822492`；Trajectory `+0.004067166005776723`；JEPA
`+0.0005757522583007546`。terminal receipt SHA256为
`1d72aa6b1a6f260c963756309f3c74a57504381b929663b9361a8201248b187e`，状态`pass`且deployment SHA
gate为true，因此P0成为当前具备最终替换资格的新冠军候选。Winner-SFT双seed已20/20并有视频收据；
Preference seed1/seed4为13/20、14/20，继续原owned GPU4/GPU5任务，无重复作业。

### 14.20 P2动作门裁决与九项门实时ETA（2026-08-25 16:12 CST）

Winner-SFT与Preference-LoRA的四组step25 seed1/seed4 Breadth20均已完成20/20、121帧、640x480、
black0及Base评分。复用Base阶段保留的SAM3 `traj.npy`，无需重跑检测器，按同一Official seed4
Breadth20基线（valid 19/20、common-valid DTW `0.0551485110`）形成四份动作门收据：Winner seed1
valid 19/20、coverage delta `-0.001234568`、DTW delta `-0.004719808`，通过；Winner seed4 valid
20/20、coverage `+0.043209877`、DTW `-0.015589791`，通过；Preference seed4 valid 20/20、coverage
`-0.006172840`、DTW `-0.001441817`，通过。Preference seed1虽然valid 20/20，但coverage
`-0.020987654`且DTW恶化`+0.012798740`，超过冻结的`-0.01/+0.01`动作门边界，终态关闭该seed，
不再消耗VLM/JEPA。

四份动作门receipt SHA256依次为Winner seed1
`e2cddb8ffc805f2b71aad0162eb54779f812be3ca494a346e46589cff4bcf208`、Winner seed4
`08725a2acd6ead47cfb7dc73ed41c57df65e7dae6e2b8da4d5ad411f8e1e042f`、Preference seed1
`1ddb517a32a2fa4b91930efa9fc5cb8ee19572390d9450ba7dbfa156b4037ce7`、Preference seed4
`b5e2667f2551062e220fd7d1906e24449d93edd73ae7ec13e203a8e7cf10f56c`。

16:06后，三条幸存线路分别在owned GPU0/GPU6/GPU5并行进入VLM→JEPA→aggregate→latest-WA2
corrected队列；16:12均为2/20，实测速率135--137秒/条，GPU显存各约17.47GiB，PID/UUID/锁与
huazhi任务一致，日志无Traceback/OOM/RuntimeError。参考step751/752历史VLM均约46分钟、JEPA约
31秒、aggregate约4秒，九项门数据预计16:52--16:56形成，裁决预计17:05前完成。若P2均不达到
更严格clean50资格门，约17:30--18:00可进入最终收口；若一条P2获唯一新clean50配额，完整生成与
全评分预计再需2--4小时；若Winner明确通过并按合同解锁step50，则完整上界约5--8小时。

全项目收据审计同时发现Task 6固定Action-Support Background Projection尚无远端终态收据；实现和
单元门存在，但正式Breadth20投影/评分尚未执行。该缺口已显式列入收口清单，不能被视为跳过；
GPU4当前空闲，可在不干扰三条九项门的前提下补齐。P0 selector仍是唯一已通过clean50最终替换门的
候选，Official+seed4继续作为不可变回滚冠军。

### 14.21 Task 6背景投影证据关闭（2026-08-25 16:17 CST）

已完成Action-Support Background Projection的全量前置资产审计。Official seed4 Breadth20的20条
视频、20张首帧、20份robot-only HDF5及20份SAM3 centroid trajectory均存在；但冻结合同要求的
renderer/expected-flow逐帧support mask、generated-RAFT active-flow tensor和generated-video SAM3
robot/object mask均未落盘，两个受审根目录内对应artifact计数均为0。因此无法精确构造四项union；
用centroid轨迹伪造mask、使用部分union或重新扫阈值都会改变冻结合同，均未执行。

该分支已按prerequisite failure终态关闭，未生成新视频、未占GPU、未写投影候选。远端receipt为
`background-projection/quality-gate.complete.json`，SHA256
`3a0c5f149ed34b6b55437650909ba77f99ebf676e4cd2c1932db1d4256e757a8`。至此Task 6的flow oracle与
background projection均有显式终态，不再是静默缺口。16:17三条P2幸存组均正常VLM 3/20，速率
约135--137秒/条，预计16:52--16:56完成，17:05前裁决；若均不获clean50资格，约17:30--18:00
完成最终收口，若一条获唯一新clean50配额则完整结束约3--5小时，Winner step50条件解锁时上界
约5--8小时。

### 14.22 P0正式冠军基线切换与P2直接对决（2026-08-25 16:44 CST）

P0 post-generation selector不再表述为“合格候选”，而是当前正式冠军；Official Stage-1 + flow20 +
seed4只保留为不可变回滚冠军。当前冠军clean50为corrected `0.6592060841461832`、raw
`0.662720295147882`，terminal receipt SHA256为
`1d72aa6b1a6f260c963756309f3c74a57504381b929663b9361a8201248b187e`。后续P2不允许仅超过旧seed4
而晋级，也不允许训练新selector、逐episode挑选P2 seed或再次使用已打开的40条holdout。

为建立严格同域基线，已新增并通过13项聚焦测试的冻结P0 Breadth20重放器：在相同20条Official
seed1/seed4视频的部署可用特征上重放既有模型/策略，写出逐条选择与收据。clean50与Breadth20
episode集合零重叠，因此没有错误复用clean50预测。Official seed1 Breadth20的latest-WA2 corrected
mean已形成，为`0.6652987549716861`；GPU4正在只补Official seed4缺失的Base/Trajectory及aggregate，
未重新生成视频。P0 Breadth20终态收据形成后，Winner seed1、Winner seed4、Preference seed4必须
分别满足corrected `>=P0+0.003`、raw `>=P0`、12项non-motion `>=P0+0.002`、Trajectory/JEPA各
`>=P0-0.005`、strict paired wins `>=12/20`和black0，才可竞争唯一新clean50。

16:44三条P2 VLM均为17/20左右，GPU0/GPU6/GPU5 owned PID、UUID、锁和日志正常，无错误；预计
约7分钟完成VLM，随后JEPA/aggregate/latest-WA2与P0直接裁决约15--25分钟。Winner step50默认
关闭，只有step25先过上述P0门、corrected uplift `>=0.006`且Instruction/Interaction/Image均不回撤
才解锁。按当前分支，无P2过门时约45--90分钟完成最终收口；若唯一P2过门并获新clean50，预计再需
2--4小时。

### 14.23 Adaptive MoE最终裁决与正式测试集边界（2026-08-25 17:39 CST）

三条P2已全部对冻结`P0-selector-Breadth20`完成直接裁决。Winner-SFT seed1 corrected/raw为
`0.6661127522`/`0.6688535567`、strict wins `8/20`，关闭；Winner-SFT seed4虽达到
corrected/raw `0.6801319441`/`0.6824085030`，相对P0为`+0.0071419704`/`+0.0069905177`，但只有
`11/20` strict wins且Instruction回撤`-0.01`，因此按冻结门关闭且不解锁step50；Preference-LoRA
seed4 corrected/raw为`0.6725589567`/`0.6749368485`、strict wins `11/20`，低于P0并关闭。
Preference seed1此前已因动作门失败关闭。没有P2获得唯一新clean50配额，也没有追加训练。

最终决策为`complete_retain_p0`：正式冠军是冻结P0 post-generation selector，Official Stage-1 +
flow20 + seed4仅作不可变回滚。远端最终收据为
`/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/final-decision.complete.json`，
SHA256 `2ad2b56f162b3863f6acfee2944dc443613d42bb7322e198a85dbcea5bfc8d59`；deployment SHA gate为true。
冻结model/policy路径分别为`selector/postgen/postgen-selector.model.json`和
`selector/policy-selection.complete.json`，SHA256分别为`4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa`、
`aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`。

必须区分开发门与正式比赛测试：上述`0.6627202951`是隔离dev-clean50证据，不是官方排行榜测试分。
截至本次远端全盘审计，当前IROS 2026 WorldArena 2.0 Track 1指定的
`WorldArena/WorldArena2.0/dataset_track1.tar.gz`尚未镜像到`/data/di`，也没有对应revision/SHA收据。
因此Adaptive MoE研发活动已完整收口，但正式测试集生成/提交仍是独立后续阶段；该测试集只能用于
最终推理和提交，禁止进入训练、selector、阈值或候选选择。

### 14.24 WorldArena 2.0正式Track 1数据落盘收据（2026-08-25 18:15 CST）

14.23的缺口已补齐。官方数据仓库revision固定为
`af1ac34d3881f84096345542c631fbb1b9540d50`；`dataset_track1.tar.gz`精确大小85,474,554 bytes，
SHA256为`7973bf1e9222086279aa12eb593b4c7adb88f781a5811b05ae5134d52c96bb41`。安全解包后共3000个常规文件、
156,497,293 bytes；episode1--1000连续，每条均包含HDF5、首帧PNG和instruction JSON，且1000行manifest
记录了全部3000个文件的独立SHA256。

完整资产位于`/data/di/worldarena2_track1_20260815/datasets/WorldArena2.0-official-track1`。主收据
`official-track1-dataset.complete.json` SHA256为
`d6f66953860abca36b2b495ac96879924fa965f08761f38521386bc817b73eb1`；manifest SHA256为
`97ef493062d9e21b4042a9987dd550c12f34763cfa345de786f40eca27f8e825`；零污染收据SHA256为
`4d44235e3081d4d5a4b43564533125b14bdefc6981fb88bc3341b04b56f84d0a`。复核结果为3000个资产文件均不可写，
5811个既有运行文本文件无资产路径引用、活动进程引用0，且正式测试下载晚于最终模型选择收据。

数据扩展策略更新为：WorldArena 1.0与2.0的公开训练/开发split可分别纳入训练和离线测试，并在自有
GPU0/4/5/6上按episode分片并行；本次正式Track 1 test split不可进入训练、selector、阈值或模型选择，
只用于冻结P0的最终推理、打包与提交。当前尚未产生官方测试/leaderboard分数。

### 14.25 Challenger延伸与六卡流水（2026-08-25 19:54 CST）

在不修改P0冻结策略和旧40条holdout的前提下，新增一次严格隔离的Challenger延伸。零成本Oracle已
终态：P0 Breadth20 mean为`0.6729899737`；Oracle(P0, Winner-SFT seed4)为`0.6843627765`，
uplift `+0.0113728028`、Winner strict wins `11/20`，因此只允许继续一次固定LoRA scale `0.5`；
ActionText虽有6条独立胜例，但在P0+Winner之上的额外uplift仅`+0.0013367522 <0.003`，继续永久关闭。
Oracle收据SHA256为`1c4d25c6b296d190f7910a949de1ea527423e4d25f08e54e3182604f5051367a`。

Winner scale0.5已在GPU0/4/5/6完成20/20、121帧、640x480、black0和四份分片收据，并合并为单一
Stage-1谱系；合并收据SHA256为`2d21c1d6786bc7110287f4f096ca30b11b6ec4836e319ac1a8a7f83385d61b30`。
独立checkpoint 774正在评分：GPU2跑Base、GPU3跑VLM，GPU4已完成JEPA。scale0.5仍是候选，只有
直接超过冻结P0 Breadth20门才会成为Challenger expert；P0仍是当前正式冠军。

新的fresh40从clean-1785确定性抽取，并与旧selector-200、dev-fast20、dev-clean50完全排除；正式
test内容未打开，只绑定零污染receipt。fresh40 manifest SHA256为
`49d89ac6c3450aff8e1705819f455b8fbab7eaff444f67bd0f55b92ede586908`，终态receipt SHA256为
`a86ceed7be985c3d8632c5a12b0011c1cf52a0710cbec4686e4fa0986b196a20`。分片已扩成GPU0/2/3/4/5/6的
`7/7/7/7/6/6`；GPU0/4/5/6正在生成冻结P0 seed1+seed4，GPU2/3评分结束后由receipt-gated watcher
自动接力。GPU1仍被其他项目占用且严格避让，GPU7按指令不用。

时间合同同步收紧：2026-08-27白天研究硬冻结，最晚当晚启动正式test-1000。六卡对2000候选的
理想生成约15.7小时，评分后完整理论下限约28--30小时，仍保留60--72小时正式测试窗口；正式test
不得反向影响模型、selector、阈值或候选选择。

### 14.26 ExpertPool-v2与Safe-Winner硬截止终态（2026-08-26 07:18 CST）

Winner scale0.5的train160与Fresh40候选已全部生成并完成Base、VLM、JEPA、aggregate及最新版
GT-capped motion评分。ExpertPool policy在打开Fresh40前已冻结；train160 OOF的所有候选阈值虽有
约+0.00139至+0.00199 corrected增益，但均违反Trajectory/JEPA保护门，因此冻结策略选择
threshold 1.0并回退P0。Fresh40一次性裁决最终选择P0 40/40、Winner 0/40、ActionText
0/40，corrected/raw/non-motion/Trajectory/JEPA uplift均为0，bootstrap单侧90%下界为0，
按closed_uplift_below_0_004关闭。终态收据路径为
/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/expertpool-v2/fresh40/decision/expertpool-v2.complete.json，
SHA256 76df6b8733234eba4e03ea5b5498ad097483a40497f499573ea3f0c634bced29。

条件分支Safe-Winner-25未启动GPU。实测下界为25步训练1182.57s、六卡Breadth20约900s、
完整Base/VLM/JEPA/corrected门约3360s，合计约91分钟，尚未计入目标构建与代码合同验证；
07:12距08:30终态截止仅78分钟。为避免以不完整分数替代门控，该分支按
closed_resource_deadline终态关闭。收据路径为
.../adaptive-candidate-system/safe-winner/resource-deadline.complete.json，SHA256
bbb04f6a2ab7763bc843287cb8cf34fd596ab3196612e57c7e0fe7904a0af23b。

最终决策已用现有验证器重新生成并纳入上述两条终态，共25项terminal outcome；冠军仍为冻结P0
post-generation selector，dev-clean50 corrected/raw为0.6592060841461832/
0.662720295147882，Official Stage-1 + flow20 + seed4仅为回滚。当前
final-decision.complete.json SHA256为
5eb9e43450bef05e24e8f8f600b5668f66971370b9c31e85422f63e272fad9cf，deployment SHA gate为true；
旧版决策保存在final-decision.pre-expertpool-20260826T0718.complete.json。07:18已精确停止六个
自有评测轮询PID；GPU0/2/3/4/5/6均为0%利用率、5--8MiB基础显存，无自有WorldArena GPU进程或
活动锁。GPU7仍有外部12.6GiB进程，未触碰。

### 14.27 P0发布冻结与HF提交链路全链烟测（2026-08-26 12:20 CST）

P0已从实验资产固化为独立发布流程。新增冻结配置
`source_inputs/flowwam_p0_track1_release.json`、核心实现
`src/worldarena_baseline/track1_release.py`、CLI
`scripts/run_track1_release.py`和运行手册`docs/TRACK1_P0_RELEASE_RUNBOOK.md`。发布器严格绑定当前
final decision、model、policy、官方test数据receipt与manifest SHA；支持确定性1000条seed1/seed4
配对计划、九项generated-only特征选择、视频逐条SHA/121帧/640x480/black门、确定性tar.gz、安全成员
校验、HF公开smoke上传与匿名下载复核。聚焦回归共14项通过，compileall与diff whitespace检查通过。

冻结selector已在独立目录对原clean50四份分数和两份生成谱系精确重放，predictions SHA仍为
`102184cde847bcdf4d9fb7a4f47a43b5d6f1762b32af4b5490951da209655ea0`，选择27/23，raw/corrected
仍为`0.662720295147882`/`0.6592060841461832`，视频validation SHA也逐位一致。正式Track1输入
episode1已分别在GPU0 seed1、GPU4 seed4真实生成，均为121帧、640x480；生成视频SHA分别为
`4aca98f79f2a9323fe337ead03c9683449574c6df4337bf6cf0e7db0b6904849`和
`51f725e96caacf744cf26c84e86bc0fa54f5921feda8174012396b6d6f15a09c`，两个任务均正常退出。

HF链路使用本项目独立token目录和远端`127.0.0.1:7890`代理，未读取或覆盖其他项目凭据。公开smoke
仓库为`clusternlh/worldarena-track1-p0-submission-smoke`，commit
`8165ebbaa2a178e38d9e7e5fae8897236466c60c`；上传后清除全部HF认证环境变量做匿名下载，得到与本地
一致的archive SHA `83873237aad18f353a542b722092c744e63146a65ffa717791b48d094b54d609`，仓库公开、非gated、
根目录恰有一个压缩包。总验证收据位于
`/data/di/worldarena2_track1_20260815/submission/releases/p0-r1/track1-release-verification.complete.json`，
SHA256 `bfccefb91925af8dcaa57a56405273f3a585f4cde177d8144153cd7b0cc07c73`，状态
`release_pipeline_smoke_verified`。本轮没有启动完整test-1000，也没有触发官方提交邮件；正式1000条
仍只能在冻结P0下执行，不能把test结果回流训练、selector或阈值。
