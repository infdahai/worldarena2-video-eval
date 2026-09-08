# Wan-Action v6：直接 gripper trajectory supervision 实验报告

日期：2026-08-18  
状态：**已完成 bounded step25 验证；负结果，按门禁停止**

## 一页结论

v6 的工程链路、标签、frozen probe、七卡训练、FSDP 校准和显存预算均通过；失败的是核心研究假设：

> 在冻结 Wan + clean gated parent 的前提下，给 blocks 8/16/24 注入独立低秩左右臂 correction，并直接最小化 predicted clean latent 的 gripper position/velocity，**没有在未见样本上带来正确的 action-trajectory 因果改进**。

| 指标 | 结果 | 门槛 | 结论 |
|---|---:|---:|---|
| Position error improvement | -2.09% | >= +15% | 未通过 |
| Velocity error improvement | -1.94% | >= +15% | 未通过 |
| Correct 优于 shift/reverse/swap | false | true | 未通过 |
| FM regression | -0.067% | <= +5% | 通过，未伤害 FM |
| Routing retention | 1.0 | >= 0.9 | 通过，左右 support 没被破坏 |

因此本轮不是训练不稳定或标签不准，而是：**correction 可以稳定存在、也能保持左右臂 routing，但它没有把 probe 内的 trajectory 目标转成更正确的 Wan 行为。**

按预先冻结的 stop rule，实验停止在 step25；没有继续 step50/100，也没有用失败候选浪费 RGB 视频、SAM3 或官方评测预算。

## 1. 问题背景与 v6 假设

此前 clean-lineage、Adapter/LoRA、data-scale 和 v4/v5 实验共同暴露出一个矛盾：机械臂可检测性和局部 action routing 可以改善，但生成视频的真实轨迹 DTW 并不稳定变好。v6 不增加 PRoPE、V-JEPA、Depth、GAN 或新的 temporal backbone，而测试一个更窄的假设：

```text
如果直接约束 predicted clean latent 中的 left/right gripper image-space trajectory，
是否能把 Wan 的生成轨迹向 commanded trajectory 拉回？
```

为了可归因，以下内容固定：

- Wan2.2-TI2V-5B backbone 全冻结；
- clean gated step10 parent Adapter 全冻结；
- frame-local gripper probe 在开始 v6 前冻结；
- 训练集固定为 clean-1000；
- dev-fast20 永不进入 probe 或训练；
- GPU 固定为 0–6；GPU7 的同事服务全程不读取、不停止、不重启；
- action representation、raster、81 帧、480×640 和缓存合同不在本轮改变。

唯一新增可训练部分为左右臂完全独立的 rank-8 correction，在 Wan blocks `8/16/24` 进行 zero-init residual 注入：

```text
frozen Wan + frozen clean parent + frozen frame-local probe
                                      |
left action support  -> left rank-8 correction --+
right action support -> right rank-8 correction -+--> blocks 8/16/24
                                                          |
                                              z0_hat -> probe -> position / velocity loss
```

## 2. 数据、泄漏与 RGB observability 合同

### 2.1 训练/评测隔离

| 用途 | 固定资产 |
|---|---|
| v6 训练 | clean-1000 cached manifest：`/data/di/worldarena2_track1_20260815/cache/v4-clean-1785-v3/manifests/clean-1000.cached.jsonl` |
| probe fitting | clean-1785 deterministic task-stratified 90/10 split，1608 train / 177 held-out |
| 视频开发评测 | 固定 `dev-fast20`，未进入 probe 或训练 |
| 训练父代 | clean gated parent，仅从 clean S1A125 谱系复现 |

训练 manifest、cache identity、replay、leakage receipt、evaluation manifest 与父 checkpoint 全部在启动前绑定。任一 hash、source、world size 或 parent/probe 变化都会 fail closed。

### 2.2 RGB 标签不把“看不见”误当成“错误”

SAM3 video tracking 对每个 sample 产出 `(left, right, 81)` boolean observability sidecar。它表示“有独立 RGB 证据可用于 probe supervision”，不表示该臂必然存在。

特别修正的合同：单 episode 中左臂没有 EEF、右臂有有效 RGB 标签，应合法记录为 `[left=0, right>0]`，而不是错误的 `0/0`。单样本 smoke 的实际结果为 `[0, 42]`。

全量输出完成：

- 1,785 / 1,785 sidecar；
- 7 / 7 worker receipts；
- 每个 sidecar 原子发布，不接受 partial 文件；
- 同一检测区域可同时支持两个由 FK 身份绑定的 EEF，不把双臂 crossing/occlusion 强行做 one-to-one 互斥匹配；
- 早期 1,157 条只读快照中已有 left-only 362、right-only 339、both-any 370、same-frame-both 133、neither 86。该快照只用于证明标签分布不是全 `0/0`，不可当作完整语料统计。

已有标签是保守少标：缺少 RGB 证据只会剔除监督 token，不会伪造左右臂位置。未来可只对 `same-frame-both` 样本复跑更强 overlap matcher；单臂样本不需要全量重跑。

## 3. Frozen gripper probe：从失败到通过

probe 将每个 clean latent frame `(48, 30, 40)` 映射为 left/right 两个 `(60, 80)` heatmap。它严格 frame-local：改变未来 latent 帧时，当前帧输出必须 bitwise 不变。probe 训练只使用 clean-1785 train split；held-out 从未参与优化。

预设可靠性门槛不变：median <= 1.5 px、P90 <= 4 px、左右臂分别通过、crossing 通过、correct mapping 显著优于 swapped mapping、future independence 精确成立。

| Probe 版本 | 变化 | Epoch | Held-out median | P90 | Crossing median | 结论 |
|---|---|---:|---:|---:|---:|---|
| P0 | 单层 3×3 Conv；归一化坐标 position loss | 10 | 4.115 | 17.390 | 5.569 | 失败 |
| P1 | position loss 改为 raster-pixel Huber | 10 | 2.618 | 7.626 | 1.774 | 明显改善，仍失败 |
| P1 | 同 P1，只延长训练 | 30 | 1.889 | 6.833 | 2.161 | 仍失败，不能只靠更多 epoch |
| P2 | 仍 frame-local；三层 3×3 spatial stack | 30 | **0.619** | **2.254** | **0.457** | 通过 |

P0 的根因不是标签或阈值：heatmap cross entropy 约为 6，而 normalized-coordinate position term 对 4 px 误差的贡献约为 0.01，几乎没有优化压力。P1 改为像素尺度 Huber 后，定位立即改善。P1 到 30 epoch 仍未过门，因此没有继续扫学习率；P2 只增加同一帧的空间感受野，没有加入 temporal layer。

最终冻结 probe：

| 指标 | 最终值 | 门槛 |
|---|---:|---:|
| Overall median error | 0.619 px | <= 1.5 px |
| P90 error | 2.254 px | <= 4.0 px |
| Left median | 0.613 px | <= 1.5 px |
| Right median | 0.624 px | <= 1.5 px |
| Bimanual/crossing median | 0.457 px | <= 3.0 px |
| Correct-arm mean | 1.137 px | < swapped |
| Swapped-arm mean | 46.561 px | reference |
| Future independence | exact true | required |

Probe SHA256：`300b4c6a7aa246110625d2e1747273ee07e30cc4f6d92e544799567496531480`

这证明 probe 足以作为严格的 **latent-space localization instrument**。它不证明优化 probe loss 等价于 decoded RGB 视频轨迹，因此 v6 仍设置 independent RGB anti-exploitation gate；该 gate 未执行，因为更早的 latent counterfactual gate 已失败。

## 4. 父代、校准与资源证据

### 4.1 clean gated parent

历史 provisional/gated checkpoint 不允许直接成为 v6 父代。唯一合法 parent：

- 3-step seven-rank parent smoke：通过；
- 10-step adapter-only parent train：完成，loss/梯度均有限且 action adapter gradient 非零；
- 8 个 episode causality audit：`causal_signal_present`；
- parent path：`/data/di/worldarena2_track1_20260815/runs/v6-clean-gated-parent/clean-gated-step10.pt`；
- SHA256：`105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`。

parent smoke 每 rank 峰值约：allocated 7.77 GiB、reserved 9.50 GiB，低于 22 GiB 硬门。

### 4.2 Sigma 与 lambda calibration

probe 在真实 clean latent 上训练，不能假设对所有 `z0_hat` 噪声可靠。校准在固定 replay 上测各 sigma bucket 的 probe error，并冻结结果：

| Sigma bucket | Median px | P90 px | v6 trajectory weight |
|---|---:|---:|---:|
| [0.0, 0.1) | 0.485 | 1.347 | 1.0 |
| [0.1, 0.3) | 0.488 | 1.290 | 1.0 |
| [0.3, 0.6) | 0.624 | 1.197 | 1.0 |
| [0.6, 1.0] | 1.150 | 2.168 | 1.0 |

四个 bucket 全部低于 reliability gate。lambda 使用 correction 的 zero-init `B` projection 参数进行梯度标定，目标 `position/FM = 0.4`、`velocity/FM = 0.2`：

| 项 | 数值 |
|---|---:|
| FM B-gradient norm | 0.0026394 |
| Position B-gradient norm | 7.2507e-06 |
| Velocity B-gradient norm | 8.8446e-06 |
| lambda position | 107.750 |
| lambda velocity | 43.494 |
| 实际 scaled position/FM | 0.4000000 |
| 实际 scaled velocity/FM | 0.2000000 |

首次 calibration 暴露 FSDP activation checkpoint 与跨图 `autograd.grad` 的兼容问题：重算时参数 storage 已被 reshard 成空。生产路径改为对 FM/position/velocity 分别执行标准 `backward()`、读取 B gradients，再按同一公式生成 lambda；问题未被静默绕过。源码改变后旧 run 目录被 provenance 门禁拒绝，因此新建 `r2`，没有覆盖旧证据。

Calibration SHA256：`d4884f62b59613a07897dc06d32d54f7c40e13fcdf1111c5b4cd3ade550c3080`

### 4.3 v6 seven-GPU smoke

| 项 | 结果 |
|---|---|
| Ranks | 7（GPU0–6） |
| GPU7 | 未使用；同事服务持续保留 |
| B gradient | 每个 rank step1 均观察到 |
| A gradient | 每个 rank step1 后均观察到 |
| Peak allocated | 约 6.01 GiB/rank |
| Peak reserved | 约 7.72 GiB/rank |
| Step time | warmup 约 4.06 s；后续约 3.49 s |
| 22 GiB hard gate | 通过 |

因此 v6 的失败不能归因于 OOM、无梯度、错误 rank topology 或 GPU7 干扰。

## 5. 正式训练与 step25 discovery audit

### 5.1 训练执行

| Checkpoint | SHA256 | 状态 |
|---|---|---|
| step10 | `a6586685f1843ac8a5e1a38b4375143525a405bde91274c563f52c3a8f257f58` | 完成 |
| step25 | `87f93b27d14968789bc15ba7bca134f304919e7eb186fc25efb6380bb4f5c37b` | 完成，进入 audit |

训练中 FM、position、velocity 和 correction gradient 都保持 finite。不同 sample/timestep 的单 batch loss 波动不构成 trajectory 成功证据，因此未按 loss 选择 checkpoint。

### 5.2 Discovery audit 设计

step25 使用 8 个未见 discovery episodes（replay indices `900..907`），对同一 sample/noise 比较：

```text
candidate correct action
candidate temporal shift action
candidate reversed action
candidate left/right swapped action
baseline clean parent
```

只有 position/velocity 至少改善 15%、counterfactual ordering 正确、routing 保持、FM 不恶化，才允许进入 independent RGB anti-exploitation check，再考虑 step50/100 或视频生成。

### 5.3 Discovery audit 原始结果

| 量 | Baseline | Step25 candidate | 相对变化 |
|---|---:|---:|---:|
| FM error | 2.40635e-04 | 2.40473e-04 | -0.067% |
| Position error | 6.31322e-05 | 6.44509e-05 | -2.09% |
| Velocity error | 9.51133e-05 | 9.69547e-05 | -1.94% |
| Reverse position error | — | 6.17672e-05 | correct 更差 |
| Shift position error | — | 6.25805e-05 | correct 更差 |
| Swap position error | — | 6.34918e-05 | correct 更差 |
| Routing retention | — | 1.0 | 保持 |

**关键现象：**FM 基本不变、routing retention 完整，但 direct trajectory objective 没有让 correct 版本胜过三类 counterfactual。它不是“强约束破坏底座”，更接近“训练信号没有穿透为所需的 action-sensitive trajectory behavior”。

## 6. 这次结果证明了什么、不证明什么

### 已证明

1. RGB-backed labels 和 FK 身份绑定足以训练高精度、strictly frame-local 的 gripper probe。
2. 单臂 EEF 缺失不是标签错误；保守 observability mask 可以安全处理该情况。
3. 左/右独立 rank-8 correction 能在七卡 FSDP 中稳定训练，且不会天然破坏 action support routing。
4. `z0_hat -> frozen probe` 路径有非零 B/A 梯度，完整训练链路真实连通。
5. 在 25 step 内，direct latent trajectory loss 没有提供所要求的 action counterfactual 排序收益。

### 未证明，不能据此宣称

1. 不能宣称 v6 改善 decoded RGB 轨迹或官方 SAM3/WorldArena 分数：RGB 视频没有生成，符合 stop rule。
2. 不能把 probe 通过等同于 probe loss 就是视频 trajectory 的好代理：independent RGB anti-exploitation gate 尚未到执行阶段。
3. 不能由 step25 负结果推出所有 trajectory supervision 无效；它只否定当前的 **frozen-probe + additive rank-8 latent correction + 8/16/24 injection** 组合。
4. 不能把负结果归因于数据泄漏、显存不足、左右 arm label 混乱或训练 crash；这些边界均被独立验证。

## 7. 决策与下一轮边界

已执行决策：

```text
step25 discovery failed
    -> 不训练 step50/100
    -> 不生成失败候选视频
    -> 不跑官方/代理排行榜评测
    -> 保留 checkpoint、calibration、audit，作为负样本证据
```

当前最有价值的结论：瓶颈不再是“能否从 latent 读出夹爪位置”，也不再是“左右 arm routing 是否存在”。更值得检验的是 **action condition 对最终 video trajectory 的作用接口**：一个小的加性 latent correction 即使可微、可训练，也可能没有足够结构能力或正确的时序/空间耦合方式，去改变 Wan rollout。

任何下一轮架构都应在长训前通过最低成本 gate：

1. frozen action counterfactual：correct 必须优于 reverse/swap/shift；
2. 10–25 step 未见 discovery audit：position 与 velocity 都有正改善；
3. 少量 decoded RGB trajectory proxy 同方向改善；
4. 最后才花 dev-fast20 视频和官方 evaluator 预算。

## 8. 可复现资产清单

| 资产 | 位置 |
|---|---|
| Final clean parent | `/data/di/worldarena2_track1_20260815/runs/v6-clean-gated-parent/clean-gated-step10.pt` |
| Parent smoke | `/data/di/worldarena2_track1_20260815/runs/v6-clean-gated-parent-smoke-ws7/production-smoke.v3.2.json` |
| RGB observability sidecars | `/data/di/worldarena2_track1_20260815/probes/v6-gripper/observability-video-v2` |
| Probe split | `/data/di/worldarena2_track1_20260815/probes/v6-gripper/clean1785-probe-split.json` |
| Frozen probe | `/data/di/worldarena2_track1_20260815/probes/v6-gripper/gripper-probe.pt` |
| v6 r2 calibration | `/data/di/worldarena2_track1_20260815/runs/v6-gripper-trajectory-r2/calibration.json` |
| v6 r2 smoke | `/data/di/worldarena2_track1_20260815/runs/v6-gripper-trajectory-smoke-r2/production-smoke.v6.json` |
| step10 / step25 | `/data/di/worldarena2_track1_20260815/runs/v6-gripper-trajectory-r2` |
| step25 audit | `/data/di/worldarena2_track1_20260815/runs/v6-gripper-trajectory-r2/audit-step-000025.json` |

## 9. Verification record

- v6 scoped remote test suite：**44 passed**；
- Python compilation：passed；
- shell syntax checks：passed；
- `git diff --check`：passed；
- 所有持久实验产物写入 `/data/di/worldarena2_track1_20260815`；
- GPU7 全程排除。
