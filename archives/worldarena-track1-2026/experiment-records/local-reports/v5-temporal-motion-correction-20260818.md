# v5 Temporal Motion Correction 实验报告（2026-08-18）

## 结论

`v5-temporal-motion-correction` 已完成实现、七卡 production smoke、10 个 optimizer
steps 和 discovery-8 机制审计。实验按预注册硬门在 step10 停止，**不进入 step25/50，
不生成 matched-fast20 视频，不替换 gated-step10 incumbent**。

停止原因不是训练不稳定，也不是 causal router 失效，而是：

> router 层 future-to-current leakage 已为精确 0，但 Wan 输出层 leakage 未下降，
> 反而轻微上升 1.37%。因此继续训练这个 additive correction 没有机制依据。

正式 selection：`candidate_count=0`，
`decision=retire_v5_temporal_motion_correction`，
`stop_reason=wan_temporal_mixing_dominates`。

## 实现范围

- 冻结 gated-step10 parent 与 Wan2.2-TI2V-5B backbone。
- 每臂仅使用 flow-x、flow-y、opening delta。
- 左右臂共享参数、独立执行 local causal TCN 与 support-weighted causal context。
- 独立 activity gate，允许 BOTH、crossing 和 support overlap。
- 左右独立 zero-init projection，仅注入 blocks 8/16/24；block0 保持 parent-only。
- 唯一训练目标为现有 `weighted_flow_mse`。
- 无 LoRA、Pose、PRoPE、V-JEPA、Depth、GAN、新 loss 或新数据。
- 固定 GPU0–6；GPU7 未使用。
- 数据使用已有 clean-1000 cached lineage，并在启动时复核 zero-leakage receipt。

## 验证结果

远端正式 Python/PyTorch focused suite：89 项通过。覆盖：

- 81→21 causal grouping 与 future impulse 精确隔离；
- 左右 arm branch 独立、BOTH/overlap 合法、support 外严格为零；
- null action 精确为零；
- step-zero 与 frozen parent `torch.equal`；
- parent/backbone freeze 与 correction-only trainable whitelist；
- checkpoint parent SHA、world-size、rank map、config、optimizer/scheduler/telemetry；
- 5-step warmup、LR 分组、5/10/25/50 checkpoint schedule；
- deterministic 7-rank replay；
- leakage gate 与最多两候选 selection gate；
- GPU7 exclusion、50 GiB reserve 和 launcher dry-run。

## Production smoke

真实配置：7×RTX 4090、81 帧、480×640、micro-batch 1、正式 FSDP、bf16，连续
3 次 forward/backward/optimizer.step/zero_grad。

| 指标 | 七个 rank 结果 |
|---|---|
| max allocated | 6.273–6.274 GiB |
| max reserved | 7.799–8.025 GiB |
| step time | 首步约 3.94 s，后续约 3.44 s |
| projection gradient | 7/7 非零 |
| router gradient | 7/7 非零 |
| smoke hard gate | PASS |

## Step10 训练

- checkpoint SHA256：`a07fcab4841ea50986b0f8fd4fa4748176ce77194c0b683d43879dab296b039c`
- step10 loss：`1.1189642`
- projection LR：`1e-4`
- router/context LR：`5e-5`
- checkpoint、optimizer、scheduler、source/replay provenance 均完整。

## Discovery-8 机制审计

固定同一 latent、noise、timestep、context，比较 frozen parent 与 v5 step10：

| 指标 | Parent | v5 step10 | 判定 |
|---|---:|---:|---|
| model future→current leakage | 0.0235682 | 0.0238905 | 恶化 1.37%，FAIL |
| router future→current leakage | — | 0.0 | PASS |
| left routing ratio | 2.1097 | 2.1903 | 保持/略升 |
| right routing ratio | 2.2195 | 2.2161 | 基本保持 |
| inactive-arm leakage change | — | -0.000282 | 略降 |

空间 routing gate 通过；失败项只有 `insufficient_model_leakage_reduction`。这表明新
router 自身确实因果，但 additive residual 进入 Wan 后，没有抑制 backbone 内部的时间混合。

## 工程问题与处置

首次联调发现并用回归测试修复三项审计集成合同：Wan checkout symlink 的 canonical
path 比较、checkpoint 缺少 router 重建参数、统一 v3 condition 中 null Pose 字段的
raster-only 过滤。缺少结构参数的旧 r1 checkpoint 被保留但不采用；正式结论来自带完整
config 的 r2 checkpoint。训练数据与前 10 步 replay 未因审计修复发生变化。

## 决策

1. 保留 gated-step10 作为 incumbent。
2. v5 step10 不进入视频评测，因为机制硬门已经失败；继续到 25/50 属于无依据消耗。
3. 不扩大 TCN、不调 LR、不加训练步数。
4. 后续若继续解决 temporal leakage，应研究 arm-aware causal masked attention；若优先冲
   轨迹分，则回到轻量 gripper-trajectory supervision。两者不应同时开展。

## 正式产物

- Run：`/data/di/worldarena2_track1_20260815/runs/v5-temporal-motion-correction-ws7-r2`
- Smoke：`/data/di/worldarena2_track1_20260815/runs/v5-temporal-motion-correction-smoke-ws7-r2/production-smoke.v5.json`
- Audit：`audit-step-000010.json`
- Selection：`selection.json`

