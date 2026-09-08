# 实验记录索引

## 1. Wan action 系列

Wan v1-v15 覆盖 action conditioning、raster geometry、temporal correction、gripper trajectory、SE(3)、direct action band、phase-locked cross attention、relational/bimanual、PRA/NQM、score boost 等方向。完整过程、失败原因和可复用结论集中在：

- `experiment-records/local-reports/2026-08-24-wan-action-v1-v15-total-experiment-report.md`
- `experiment-records/local-reports/2026-08-19-wan-action-v1-v11-architecture-experiment-lineage.md`
- `experiment-records/remote-top/experiments/reports/2026-08-19-wan-action-v1-v121-architecture-experiment-lineage.md`
- 同目录各 v4-v15 单项报告与 `plans-and-specs/` 对应计划

这条链路用于排除无效机制和形成消融经验，没有进入最终 HZ-World 提交权重。

## 2. Official FlowWAM 与完整 15 项诊断

FlowWAM pivot、balanced campaign、adaptive candidate、targeted15、Object-Response Bridge、post-training/refiner 等实验记录位于：

- `experiment-records/local-reports/2026-08-20-v14-flowwam-wa2-experiment-report.md`
- `experiment-records/local-reports/2026-08-20-v15-flowwam-balanced-campaign.md`
- `experiment-records/local-reports/2026-08-23-worldarena-track1-final-score-report.md`
- `experiment-records/local-reports/2026-08-31-flowwam-round2-posttraining-experiment-log.md`
- `experiment-records/plans-and-specs/plans/2026-08-30-flowwam-targeted15-campaign.md`
- `experiment-records/plans-and-specs/plans/2026-08-30-flowwam-object-response-bridge.md`

关键结论是 motion 三项受 matched-GT cap 约束，继续全局强化运动收益有限且容易伤害 Image/JEPA；简单 LoRA、ORB、refiner 和后训练没有形成可靠的统一 Pareto 提升，因此最终停止改生成模型，转向已有候选的 inference-time routing。

## 3. 多 seed 与排序器

最终链路只使用已有 seed1/2/3/4：

1. 冻结 9 项生成侧特征。
2. 用 160 个历史 episode、seed1/4 共 320 行拟合 `alpha=10` 的多目标岭回归，预测 corrected14。
3. 以 P0 为 incumbent；只有预测均值严格提升超过 `0.003` 才替换。
4. 在新的原生 16 例上计算完整 15 项并执行一次性门禁。
5. 门禁通过后，对正式 1000×4 候选执行一次 CPU 重排。

主要记录：

- `experiment-records/deliverables/track1-predicted14-ranker-20260903/PLAN.md`
- `final-results/ranker-fit/fit.complete.json`
- `final-results/holdout16-full15/comparison.complete.json`
- `final-results/holdout16-full15/comparison.md`
- `final-results/test1000-routing/route.complete.json`
- `final-results/test1000-routing/test1000-predicted14-predictions.jsonl`
- `final-results/test1000-routing/test1000-predicted14-details.json`

## 4. 正式提交

最终 ZIP 包含根目录 `README.md` 和 `HZ-World/episode1.mp4` 至 `episode1000.mp4`。验收包括唯一编号、640×480、原生 FPS、逐 episode GT 帧数上限、全解码 black0、ZIP CRC、安全成员路径和逐成员 SHA。ZIP 本体不重复提交到 GitHub；公开精确 revision 下载链接和全部收据位于 `final-results/delivery/`。

本归档保留失败和被否决实验的收据，是为了让后续判断能够追溯；这些收据不代表对应候选已经晋级。
