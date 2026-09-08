# Wan-Action v10 分阶段多损失训练报告

日期：2026-08-19  
正式运行目录：`/data/di/worldarena2_track1_20260815/runs/v10-action-relational-native-attention`  
训练设备：单卡 GPU6（RTX 4090 24 GiB）  
最终状态：**Stage A 在 step150 机制门失败，训练按合同停止；Stage B/C 未启动。**

## 1. 执行结论

本轮完成了数据封存、relation cache、损失梯度标定、production smoke、Stage A 的 step50 与 step150 训练及固定 audit-20 审计。

最重要的结论是：

- 训练工程链路健康：无 OOM、NaN、越权梯度或显存爬升，单步约 6.06 秒。
- 画质主目标没有被破坏：step150 的 FM regression 为 `-0.0261%`，即相对 frozen parent 略有改善。
- 时间相位识别稳定：`phase+1=19/20`、`phase-1=20/20`，step50 到 step150 基本不变。
- reverse/swap 语义只得到弱改善：两者 win count 都由 `9/20` 提升到 `12/20`，但 swap 平均 margin 到 step150 仍为负。
- step150 唯一失败原因是 `swap_margin_not_positive`。因此没有合法依据进入 Stage B，也没有生成 `step-000150-gated.pt`。
- 本轮**没有进行 RGB 视频解码、Trajectory proxy、SAM3 或官方 WorldArena 评测**，所以不能声称最终视频轨迹已经提升。

可保留的正式候选只有 `step-000050-gated.pt`；`step-000150.pt` 是诊断 checkpoint，不是晋级 checkpoint。

## 2. 本轮假设与架构

本轮验证的核心假设是：在部分解冻 Wan 原生 attention 的前提下，通过 relation action branch 和课程式损失，能否先学会 action semantics，再学习 timing 和最终轨迹。

固定 parent：

- 名称：`clean-gated-step10`
- SHA256：`105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`

可训练范围：

- Wan blocks `6–17` 的原生 self-attention `Q/K/V/O`。
- relation action encoder、relation Q/K、relation gate。
- 两个 hidden EEF heads。
- 总可训练参数：`453,560,260`。
- 所有非白名单参数必须无梯度；smoke 和训练均记录为 `outside_whitelist_gradients=0`。

课程式目标：

| 阶段 | Step | 主目标 | 实际状态 |
|---|---:|---|---|
| A：semantics | 1–150 | FM + CF + hidden EEF；hidden backbone 先 stop-gradient 再 ramp | 已完成，step150 gate 失败 |
| B：timing | 151–300 | 降低 CF，加入逐步增大的 phase loss | 未启动 |
| C：trajectory | 301–500 | 加入 position/velocity，hidden 降权或退出 | 未启动 |

这意味着 position/velocity 在 Stage A 中仅作为审计信号，没有作为正式轨迹优化阶段运行。

## 3. 数据与零泄漏合同

正式 action-video 数据共 `2,100` 条，划分为：

| 划分 | 数量 | 用途 |
|---|---:|---|
| optimizer | 2,060 | 训练 |
| audit | 20 | 固定机制审计 |
| dev | 20 | 后续开发评测保留 |

训练数据分层：

| Stratum | 数量 |
|---|---:|
| single_dominant | 917 |
| bimanual_heavy | 504 |
| mixed | 414 |
| quiet | 265 |

audit-20 覆盖 `bimanual`、`crossing_or_overlap`、`sequential`、`single_arm` 标签。optimizer、audit、dev 之间执行严格互斥；official test 标记为 unavailable，本轮未读取、未缓存、未参与任何选择。

关键数据哈希：

| 资产 | SHA256 |
|---|---|
| source manifest | `067a942538e71b796a2251efd5d9fa682af6f1245ca18f9788c03fcc2b3be712` |
| optimizer manifest | `720d1719a2f8393d0d03da171a985421a910baf5c3b40d95fe779e3ffd9c6124` |
| audit manifest | `17dbc1223811cd147d82641937b78523efd6c5380eec6f74c32cc497019e2d06` |
| dev manifest | `aaf9f4268a78e3ed160908780809ba9c23f12946176072787048d21ed7f50864` |
| metadata | `60a2d9ab0570abdbb8292e4ed1792ef1019ca46d4b103d223bf57ac14737e464` |
| replay-500 | `dec0b773cac1a625598e7d9c276bc1717e37486175d719ce22e53c2ee9e1e2b1` |

缓存状态：

- relation cache：`2,080/2,080` NPZ 完成，并生成 normalization。
- 扩展数据 T5/VAE cache：`315/315` 完成。
- 训练热路径未加载 T5/VAE，只读取缓存 latent、text embedding 与 action condition。
- 当时磁盘可用空间约 `1.1 TiB`，没有逼近保留空间红线。

## 4. 梯度标定

预检先在固定 calibration batch 上测量各目标对真正可训练 attention 参数的梯度范数，再计算固定 lambda。lambda 的目的不是对齐 loss 数值，而是对齐梯度贡献。

基准 FM 梯度范数：`0.0489040315`。

| Objective | 原始梯度范数 | 目标相对 FM 梯度比 | 固定 lambda |
|---|---:|---:|---:|
| CF | 0.0228354130 | 0.25 | 0.535396837 |
| hidden EEF | 7.0756635666 | 0.20 | 0.001382316 |
| phase | 0.0012829492 | 0.45 | 17.153300854 |
| position | 0.0000278136 | 0.30 | 527.483549909 |
| velocity | 0.0000509188 | 0.20 | 192.086226133 |

position/velocity 的 lambda 数值大，是因为损失在归一化坐标单位下梯度很小；标定后的实际目标梯度分别只有 FM 的 0.30 和 0.20，并不是梯度爆炸。该解释经过 smoke 的真实梯度与显存门验证。

## 5. Production smoke

合同：`wan-v10-production-smoke/1`。

连续执行三轮正式形状的 forward → backward → optimizer.step → zero_grad：

| Iteration | Step time | FM | Grad norm | Negative | 越权梯度 |
|---:|---:|---:|---:|---|---:|
| 1 | 6.0449 s | 0.19634 | 0.20129 | reverse | 0 |
| 2 | 6.0509 s | 0.22445 | 0.07551 | swap | 0 |
| 3 | 6.0407 s | 0.72649 | 0.07629 | reverse | 0 |

显存：

- peak allocated：`17,062,419,456 B`，约 `15.89 GiB`。
- peak reserved：`21,816,672,256 B`，约 `20.32 GiB`。
- 硬上限：`22 GiB`。
- 结果：通过，reserved 仍有约 `1.68 GiB` 安全余量。

梯度可达性：

| 参数组 | 最大记录梯度 |
|---|---:|
| native Q/K/V/O | 0.2861168981 |
| relation encoders | 3.935e-10 |
| relation gates | 1.4438e-05 |
| hidden EEF heads | 0.0064523132 |

relation encoder 梯度非常小，但非零；主要优化能力落在原生 Q/K/V/O 与 hidden heads 上。

## 6. 训练过程

正式可复现训练采用固定 parent、replay、seed 与 source closure。一次 step50 训练曾在保存阶段被旧 checkpoint lambda 上限拦截；该次没有写出 checkpoint。修正 validator 后，从相同 parent/replay/seed 进行确定性重跑，再从 gated step50 resume 至 step150。

`training.jsonl` 共 200 行：前 50 行来自被保存门拦截的首次运行，后 150 行是正式确定性重跑与 resume。统计学习曲线时必须按每个 step 的最后一次记录取值，不能把重复的 1–50 当成额外训练 exposure。

正式 1–150 step 的总 GPU step 时间约 `909.13 s`，即 `15.15 min`；审计、加载、checkpoint 保存不计入该数。

| Step 区间 | 平均 step time | 平均 FM | 平均 hidden loss | 平均 CF margin | 平均 grad norm |
|---|---:|---:|---:|---:|---:|
| 1–25 | 6.0509 s | 0.42287 | 0.48672 | 4.05e-05 | 0.56094 |
| 26–50 | 6.0495 s | 0.40714 | 0.27638 | 2.61e-05 | 0.14102 |
| 51–100 | 6.0651 s | 0.35713 | 0.14940 | 1.03e-05 | 0.39779 |
| 101–150 | 6.0673 s | 0.35842 | 0.07950 | 1.33e-05 | 0.12235 |

hidden EEF loss 从早期约 `0.65` 降到后期约 `0.08–0.15`，证明 hidden head 学习正常；但它没有自动转化成稳定的 swap separation。

## 7. Step50 审计

step50 的 gate 是训练健康门，不是最终 action-separation 晋级门。结果：`pass=true`，并生成 `step-000050-gated.pt`。

| 指标 | Step50 |
|---|---:|
| phase +1 | 19/20，margin 0.0100048 |
| phase -1 | 20/20，margin 0.0108126 |
| reverse | 9/20，margin -5.2591e-05 |
| swap | 9/20，margin -4.4347e-05 |
| FM mean | 0.3448077 |
| FM regression | -0.00200% |
| position improvement | +0.1996% |
| velocity improvement | +0.2844% |
| routing retention | 100% |

解释：相位能力已经很强，FM 与 routing 健康；reverse/swap 仍未学会。step50 允许继续 Stage A，只代表没有工程性崩坏。

## 8. Step150 审计

step150 结果：`pass=false`。

唯一失败原因：

```text
swap_margin_not_positive
```

| 指标 | Step150 | 相对 step50 |
|---|---:|---:|
| phase +1 | 19/20，margin 0.0100142 | 基本不变 |
| phase -1 | 20/20，margin 0.0108124 | 基本不变 |
| reverse | 12/20，margin +2.1905e-07 | wins +3；仅刚过零 |
| swap | 12/20，margin -1.4786e-05 | wins +3；仍为负 |
| FM mean | 0.3447247 | 改善 |
| FM regression | -0.0261% | 远低于 2% 上限 |
| position improvement | +0.0837% | 小幅正向，但弱于 step50 |
| velocity improvement | -0.0652% | 转为轻微负向 |
| routing retention | 100% | 保持 |
| relation enabled beats zero | false | 未建立稳定收益 |
| hidden EEF finite | true | 健康 |

step150 的信号不是完全无学习：reverse/swap 的 episode 胜数都增加了 3，swap 平均 margin 也从 `-4.43e-05` 收窄到 `-1.48e-05`。但 action-energy 差异仍处于 `1e-05` 量级，且不同 episode 正负抵消，尚未形成可依赖的语义 separation。

按照预先冻结的课程合同，Stage A 必须先建立稳定 correct-vs-swap 方向，才能进入专攻 timing 的 Stage B。这里失败后停止是正确行为；继续到 300/500 会把算力投入到已经很强的 phase 能力，而没有解决当前主要瓶颈。

## 9. 工程故障与修复记录

### 9.1 crossing/overlap 标签语义错误

初版 finalize 要求 simultaneous RGB detector observability 才标记 crossing/overlap，导致审计集合同无法满足。修复后：

- crossing/overlap 从 commanded EEF geometry 确定。
- RGB observability 仍作为 probe eligibility，而不是机械臂动作事实本身。
- 修复通过 TDD 与全量数据合同测试。

对应提交：`1a9a168`。

### 9.2 归一化轨迹损失 lambda 被错误上限拒绝

preflight 根据梯度比算出 position/velocity lambda 为 `527.48/192.09`，旧 validator 的 `[1e-4,100]` 上限误判为异常。根因是 normalized-coordinate loss 的原始梯度非常小，而不是爆炸。修复为：

- 仍以真实梯度比例作为安全依据。
- 合理扩大静态 lambda cap。
- checkpoint validator 与 preflight 使用同一合同。

对应提交：`49423a5`、`6253ef7`。

### 9.3 首次 step50 在保存门失败

第一次 step50 已完成计算，但保存 checkpoint 时仍命中旧 cap。系统 fail closed，没有产出不符合合同的 checkpoint。修正后使用相同 parent/replay/seed 重跑 1–50，正式 checkpoint 与 audit 均成功写入。因此日志存在 step 1–50 两份记录；报告统计已去重。

## 10. 产物与完整性

正式目录：

```text
/data/di/worldarena2_track1_20260815/runs/v10-action-relational-native-attention/gpu6
```

| 文件 | 大小 | 时间（Asia/Shanghai） | SHA256 |
|---|---:|---|---|
| preflight.json | 8,637 B | 2026-08-19 03:09:09 | `60afabb26c76a4dfde18c7059d2b3d9ce49497c7ad90114fc57e7ac7953e6799` |
| production-smoke.json | 1,581 B | 2026-08-19 03:10:18 | `37f16e55208ef8c685eb4df946d024848102d9d62077bd7d1369bdde104ffd89` |
| training.jsonl | 65,601 B | 2026-08-19 03:38:24 | `c98425b48b25cde7ecc1af965d5d943bbea4dc1757dfb2a8f388b3685fedae52` |
| audit-050.json | 45,259 B | 2026-08-19 03:26:57 | `08701376e6dabae325bfae81ca7decd68dd1eb243e0ad74bb685ef95ddf7c973` |
| audit-150.json | 44,960 B | 2026-08-19 03:42:08 | `b229a7a854a08be990da31519c4a3b69f658311159950fd07166df8df45f98d5` |
| step-000050-gated.pt | 2,724,133,117 B | 2026-08-19 03:27:00 | `192deae39c4727f82b4c233e6f92007f464a7ba068ffc37e10b7cf133eeb2b21` |
| step-000150.pt | 2,724,134,265 B | 2026-08-19 03:38:28 | `4bde963c6babe6b116367b73f0a8d345954009ae75b1f11b831e137b96e5700d` |

不存在 `step-000150-gated.pt`，这是预期的 fail-closed 结果。

最终验证：

- 远端 v10 focused/full selection：`42 passed`。
- 本地 `py_compile`：通过。
- `git diff --check`：通过。
- GPU6 任务结束后：显存 `6 MiB`、利用率 `0%`、无 compute PID。

## 11. 研究判断

本轮回答了一个窄而重要的问题：课程式多损失能保持 FM、routing、显存和训练稳定，也能让 hidden EEF head 收敛；但在 150 step 内，仍不足以让 correct action 稳定优于 swapped action。

证据支持：

- partial native-attention unfreeze 解决了“小分支没有容量”的部分问题；参数确实能更新，FM 也未退化。
- phase timing 已不是当前短板，±1 separation 从 parent 到 step150 都很强。
- 当前短板仍是 arm/action semantic binding，尤其 swap，而不是单纯训练不稳定或数据泄漏。
- hidden representation 变得可预测，不等于最终视频动力学真正依赖正确 arm action。

证据尚不支持：

- 不能据此声称 RGB trajectory、SAM3 Trajectory Accuracy 或 EWMScore 提升。
- audit-20 样本量有限，且 energy margin 很小；step150 不应作为发布候选。
- 尚未验证 Stage B/C 的 phase 与 trajectory loss，因为 Stage A 合同禁止越级。

## 12. 建议

1. 冻结并保留 `step-000050-gated.pt` 作为当前唯一可复现实验候选；`step-000150.pt` 仅用于诊断。
2. 不继续当前配置到 300/500，也不直接解码 step150 做正式比较。
3. 下一轮只改变 Stage A 的 action-semantic 监督路径或 arm binding 机制；保持数据、parent、replay、audit-20 和其余训练合同不变，才能归因。
4. 新方案必须先让 swap 平均 margin 稳定为正且 `relation_enabled_beats_zero=true`，再进入 RGB 视频与官方轨迹链路。

一句话总结：**v10 的课程式训练工程上是成功的，但 Stage A 的 swap semantics 仍未成立；系统在正确位置停止了训练。**
