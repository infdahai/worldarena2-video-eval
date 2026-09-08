# Wan-Action-Lite+ v4 step3600：matched fast20 后台评估报告

**结论：淘汰 `v4-step3600`；不进入官方 SAM3/WorldArena/VLM/JEPA。**

本报告只陈述已经落盘的开发集 proxy 结果。它不是 WorldArena 官方 test-1000 成绩，也不能替代未来的官方 SAM3 指标。

## 1. 实验目的与判定问题

本轮只验证一个问题：在保持 `gated-step10` 的轨迹收益基础上，v4 的“左右独立 geometry stream + late fusion + coordination branch”能否降低双臂时序角色泄漏，并改善实际视频轨迹。

直接 parent 是 `gated-step10`，长期 incumbent reference 是 `S1A125`。预先约定的进入官方评测条件为：

- 相对 `gated-step10` 的 raw DTW 改善；
- paired win rate >55%；
- 黑视频为 0；
- detector failure 不恶化；
- 双臂/identity 与 temporal-leakage 不退化。

## 2. 可复现实验合同

| 项目 | 固定值 |
| --- | --- |
| 候选 | `v4-step3600` |
| v4 checkpoint | `runs/v4-dual-arm-geometry-latefusion/step-003600.pt` |
| checkpoint SHA256 | `2dbc680e0c793b1875545f0fb1535c39a1ae82ef1c05a9e4859e070868e92707` |
| parent | `s1a125-support-gated-adapter25-ws7/step-000010.pt` (`gated-step10`) |
| 比较集合 | 固定 dev-fast-20，共 20 episode、4 task × 5 episode |
| 比较变体 | `S1A125`、`gated-step10`、`v4-step3600` |
| 匹配条件 | 同一首帧、prompt、action condition、seed；每变体 20 条，共 60 视频 |
| 生成合同 | 81 帧、640×480、50 sampling steps、action scale=1.0、text CFG=5.0 |
| 生成状态 | 60/60 done，0 failed，GPU0–6；GPU7 未使用 |
| 评测器 | project-owned learned arm proxy detector；`official=false` |

结果根目录：`/data/di/worldarena2_track1_20260815/eval/v4-step3600-vs-parents-fast20-r2/`。

## 3. 评测器有效性与边界

本轮使用的 proxy detector 本身通过了它的 GT 校准门：coverage **0.9985**、median pixel distance **0.802 px**、correct-beats-reverse **100%**、correct-beats-hold **80%**。

因此它适合做当前 checkpoint 的相对筛选；但它仍不是官方 SAM3，尤其双臂交叉/遮挡的 identity 判定可能有噪声。基于这一点，以下结论应表述为“v4 未通过 proxy 晋级门”，而不是“v4 在官方榜单必然退化”。

## 4. 总体结果

| 指标 | S1A125 | gated-step10（直接 parent） | v4-step3600 | v4 相对 parent |
| --- | ---: | ---: | ---: | ---: |
| mean DTW distance（低好） | 18.09 | **16.27** | 34.28 | **+110.7% 劣化** |
| trajectory score（高好） | 0.0809 | **0.0872** | 0.0457 | **-47.6%** |
| paired win rate vs parent | 40% | — | **20%（4 win / 16 loss）** | 未达 55% |
| mean coverage（高好） | 65.1% | **87.5%** | 48.3% | **-39.3 pp** |
| detector failure rate（低好） | 50% | **15%** | 80% | **+65 pp** |
| catastrophic rate（低好） | 65% | **45%** | 80% | **+35 pp** |
| endpoint error（低好） | 34.42 px | **30.01 px** | 41.76 px | **+11.75 px** |
| proxy arm-swap rate（低好） | 12.9% | 15.8% | **4.2%** | -11.6 pp |
| black-frame fraction | 0 | 0 | **0** | 通过 |

### 判读

v4 的唯一明确正向信号是 proxy arm-swap rate 下降；它说明分离 stream/late fusion 可能改善了**静态 identity 混淆**。但这没有转化为轨迹收益：可检测性、DTW、endpoint 都显著退化，且 16/20 paired episode 输给直接 parent。

因此这不是可接受的“identity 换 trajectory 小 trade-off”，而是当前 `step3600` 已跨过视频可用性门。

## 5. 按 task 切分

| Task | S1A125 DTW | gated-step10 DTW | v4-step3600 DTW | v4 赢 parent |
| --- | ---: | ---: | ---: | ---: |
| click_alarmclock | 26.80 | **20.14** | 30.24 | 2/5 |
| click_bell | **19.59** | 22.64 | 54.65 | 0/5 |
| dump_bin_bigbin | **10.58** | 11.11 | 12.71 | 2/5 |
| grab_roller | 15.40 | **11.21** | 39.53 | 0/5 |

劣化不是单一 task 的偶发异常。`click_bell` 与 `grab_roller` 最严重；`dump_bin_bigbin` 相对较接近，但仍未胜过 parent 均值。

## 6. 视觉安全门

`gated-step10` 与 v4 的 paired black-video gate：

- 20 对全部通过；
- candidate mean / max black fraction 均为 0；
- 无 paired black regression。

所以淘汰原因**不是黑屏或编码损坏**，而是机械臂检测可见性和轨迹本身的严重退化。

## 7. 训练审计与视频结果的分歧

训练在 step3600 正常结束，checkpoint 完整可复现；运行状态为 `completed_step=3600`、`stop_reason=completed`。训练期 action audit 曾显示 correct-vs-swap margin 为正（step3600 记录为约 0.0538），左右 branch 未触发 collapse。

但视频 rollout 表明：

> 冻结/局部 action audit 的语义 margin 不能替代生成视频中的 trajectory selection。

v4 的 residual 结构可能在内部指标上增强 identity separation，却同时改变了 Wan 原有的时序生成分布，导致 EEF 轨迹可检测性与运动稳定性下降。这是当前最应记录的架构研究结论，而不是“继续把 v4 训练更久”。

## 8. 决策

1. `v4-step3600` **淘汰**；不跑官方 SAM3，不进入 official dev-clean-50，也不作为 Stage-2/最终生成 parent。
2. 保留 `gated-step10` 为当前直接 incumbent：它在本 matched20 上 DTW 最低、coverage 最高、detector failure 最低。
3. 仅当需要区分“后期过拟合”与“v4 架构整体失配”时，运行 `v4-step2500` 的**相同三方 matched fast20**。不要直接进入新一轮训练。
4. `v4-step2500` 若同样未达到 parent 的 DTW 改善、win rate >55%、detectability 不退化，则停止 v4 late-fusion 线；后续改为 action representation / supervision alignment 研究，而非继续调 v4 step 数或 LoRA。
5. 在 proxy 或人工复核出现矛盾时，保留该报告和全部视频作为 taxonomy 证据；官方 SAM3 只用于已通过 proxy 的 finalist confirmation。

## 9. 原始证据

- `matched-fast20.jobs.jsonl`：60 个严格 matched 的生成作业。
- `arm-proxy.json`：逐 episode 的 detector metrics、ranking、calibration 与 guardrail。
- `gated-step10-vs-v4.video-sanity.json`：黑视频门。
- `done/*.json`：每条视频的原子完成记录和 SHA256。
- `videos/{s1a125,gated-step10,v4-step3600}/`：全部原始视频及 sidecar。

