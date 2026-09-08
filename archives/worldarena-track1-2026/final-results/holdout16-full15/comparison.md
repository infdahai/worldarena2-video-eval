# Fresh16: GO

local-public-protocol-not-official

EWM15: 61.961235 → 62.312463；Δ +0.351228 points。
非JEPA14 Δ +0.003795；单侧90%下界 +0.001385；双侧95% CI [6.103102678572444e-05, 0.008268028621186111]。
JEPA Δ -0.000452；Trajectory Δ +0.019513。

| 预注册门槛 | 通过 |
|---|---|
| ewm_gain_ge_0_3_points | True |
| nonjepa14_mean_gain_gt_0 | True |
| nonjepa14_one_sided90_lower_gt_0 | True |
| jepa_delta_ge_minus_0_005 | True |
| trajectory_delta_ge_minus_0_005 | True |

JEPA只用真实集合值，不构造逐例JEPA置信区间。其他14项按4 tasks × 4 episodes分层配对bootstrap，10000次，seed=20260903。

n=16, fixed tasks are not independent task-level replicates; episode-disjoint, not task-disjoint; parent pretraining exposure unknown; do not retune on these results; no official score/rank claim

逐项均值、差值、GTcap与SHA证据见 comparison.complete.json；不自动改正式提交选择。
