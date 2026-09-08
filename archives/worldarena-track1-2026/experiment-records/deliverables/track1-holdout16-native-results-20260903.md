# 留出集最终评测与提交决策

2026-09-03 18:03 CST 主控回读，全部评分和CPU汇总已于17:55完成。原GPU0/4评分及CPU接续进程正常退出，GPU7外部任务未触碰。

**结论：四seed未验证出优于P0，不建议直接以四seed替换P0。** corrected15×100：P0 **64.185950**，四seed **63.617427**，差 **−0.568523分**。这是16条留出集的本地复现结果，不是1000条官方成绩；配对非JEPA14项CI跨0，不能声称四seed已统计显著劣于P0。

推荐保守提交P0，但最终版本待用户确认；没有切换或覆盖任何已冻结包，没有发正式邮件。四seed包已通过格式/匿名下载验收，保留作为候选，格式通过不等于效果晋级。

主控独立核验：输入与代码SHA、两个CSV各16唯一episode/15项、三motion共96个min(raw,matchedGT)检查、其他12项共384个原值检查、所有均值重算均PASS。集合级JEPA不计作16个独立样本。

[完整JSON收据](./track1-final-four-seed/holdout16-evidence/comparison.complete.json) SHA `86262e2714f67a47486d54e4c3a08aab744d180c198dcf9736b41446eaddb673`；同目录有两组逐episode CSV、完成收据及远端原始报告。过程记录保留在 `/private/tmp/wa2-holdout16-tools/EXECUTION-20260903.md`。

---

## 完整公式与筛选边界（2026-09-03 用户确认）

本节明确后续目标，不代表四seed正式路由已经重算或新排序器已经训练。当前1000条仍是原冻结generated9路由；包、上传与邮件均未变更。

设S为每个episode恰选一个seed后形成的整组视频，G为同协议真实GT。M={Dynamic Degree, Flow Score, Motion Smoothness}。B为另外11个逐episode指标：Image、Aesthetic、Subject、Background、Photometric、Interaction、Trajectory、Depth、Perspectivity、Instruction、Semantic。J(S,G)为整组JEPA。

完整公开可复现目标：

```text
T14(i, s) = sum[m in B] raw_metric(i, s, m)
           + sum[m in M] min(raw_metric(i, s, m), matched_GT(i, m))

EWM15(S) = (100 / 15) * (mean[i=1..N] T14(i, selected_seed[i]) + J(S, G))
```

这保留全部15项各1/15权重。三motion必须先逐episode封顶再聚合，不得用min(两组均值)代替。其他12项（含JEPA）不做GT封顶，也不因弱项低就擅自提高最终计分权重。可公开复现目标不声称包含官方未公开的difficulty/OOD调整。

在有GT且所有项均已实测的开发数据上，替换i的一条视频应比较**整组**增量：

```text
delta_EWM15 = (100 / 15) * (
    (T14(i, candidate) - T14(i, incumbent)) / N
    + J(S_after, G) - J(S_before, G)
)
```

因此即使有GT，逐episode取某张“15项表”的最大值也不能自动保证整组JEPA最优；JEPA不能把单个seed整组的值当作该seed每条视频的固有标签。真实整组目标严格提升才接受，平分保留原选择；这只保证该次已测目标，并不保证新数据泛化。GT知情择优只能标为oracle诊断，不冒充部署策略的留出验证。

正式test1000缺少完整GT，不能直接求此真实目标。若做新部署排序器，输入只用测试可计算特征，训练/验证目标必须对齐上述完整EWM15：逐episode建模可加的14项，JEPA在整组验证，不造逐例JEPA标签；缺失的GT依赖项标为未知或预测，不补0、不丢项改分母、不以raw9均值代替真实15项。预测目标不提供真实成绩单调保证。

本轮只统一并复核公式，没有新训练、没有消耗留出集重新调参、没有重选1000条。新的排序器如果不能在独立验证中证明完整目标改善，则不替换P0。现有native16结果保留为验证证据，不用它反向拟合新排序规则。

主控使用上式重新拆算两个已完成结果，全部与原CSV/收据一致：

| 部分（先按episode平均，再求该部分指标和） | P0 | 四seed |
|---|---:|---:|
| 11项非motion且非JEPA之和 | 7.846392256716 | 7.755162204232 |
| 3项GTcap motion之和 | 0.856358719519 | 0.856173057416 |
| 整组JEPA | 0.925141572952 | 0.931278765202 |
| 完整EWM15 | 64.185950327916 | 63.617426845666 |

计分依据：[官方motion修正公告](https://v2.world-arena.ai/WorldArena-2.0/motion-quality-evaluation-update.html)。既有finalizer已验证96个逐episode min检查与384个未改其他指标值；本次不是重跑评分或修改官方scorer。

## Native16 完整15项配对结果

完成时间：2026-09-03T09:55:25.153353+00:00。差值均为 Final − P0。

| 指标 | P0 GTcap mean | Final GTcap mean | 差值 | 胜/平/负 | 配对95% CI |
|---|---:|---:|---:|---:|---|
| Subject Consistency | 0.843067 | 0.795432 | -0.047635 | 6/7/3 | [-0.143708, 0.000836] |
| Aesthetic Quality | 0.389370 | 0.389236 | -0.000134 | 4/7/5 | [-0.003460, 0.003234] |
| Image Quality | 0.518537 | 0.522826 | +0.004289 | 6/7/3 | [-0.000945, 0.009805] |
| Background Consistency | 0.909165 | 0.859605 | -0.049560 | 5/7/4 | [-0.151507, 0.002508] |
| Dynamic Degree | 0.179601 | 0.179601 | +0.000000 | 0/16/0 | [0.000000, 0.000000] |
| Interaction Quality | 0.662500 | 0.662500 | -0.000000 | 1/14/1 | [-0.025000, 0.025000] |
| Perspectivity | 0.912500 | 0.912500 | +0.000000 | 1/14/1 | [-0.025000, 0.025000] |
| Instruction Following | 0.725000 | 0.700000 | -0.025000 | 0/14/2 | [-0.062500, 0.000000] |
| Semantic Alignment | 0.890534 | 0.890808 | +0.000275 | 6/2/8 | [-0.014160, 0.014160] |
| Flow Score | 0.085438 | 0.085252 | -0.000186 | 0/14/2 | [-0.000371, 0.000000] |
| Depth Accuracy | 0.990127 | 0.994680 | +0.004553 | 5/9/2 | [-0.000703, 0.009843] |
| Trajectory Accuracy | 0.765959 | 0.778998 | +0.013039 | 4/9/3 | [-0.047707, 0.073685] |
| Photometric Consistency | 0.239633 | 0.248577 | +0.008944 | 7/7/2 | [-0.000948, 0.023841] |
| Motion Smoothness | 0.591320 | 0.591320 | +0.000000 | 0/16/0 | [0.000000, 0.000000] |
| JEPA Similarity | 0.925142 | 0.931279 | +0.006137 | 集合级，不计算 | 不计算 |

## 汇总

| 口径 | P0 | Final | 差值 |
|---|---:|---:|---:|
| raw/all15_point | 0.669359 | 0.663617 | -0.005742 |
| raw/nonmotion12_point | 0.730961 | 0.723870 | -0.007091 |
| raw/nonjepa14 | 0.651088 | 0.644498 | -0.006590 |
| corrected/all15_point | 0.641860 | 0.636174 | -0.005685 |
| corrected/nonmotion12_point | 0.730961 | 0.723870 | -0.007091 |
| corrected/nonjepa14 | 0.621625 | 0.615095 | -0.006530 |

非JEPA 14项逐episode均值：胜/平/负 9/2/5；95% CI [-0.023448039825331068, 0.006525456320847425]；区间包含0。未设置或推导GO阈值。

每项GT匹配数：{'Dynamic Degree': 16, 'Flow Score': 16, 'Motion Smoothness': 16}；发生cap数：{1230: {'Dynamic Degree': 16, 'Flow Score': 14, 'Motion Smoothness': 16}, 1231: {'Dynamic Degree': 16, 'Flow Score': 13, 'Motion Smoothness': 16}}。

配对 bootstrap：4 tasks × 4 episodes 分层，20000 次，seed=20260903。JEPA 只用于集合级点估计。

- n=16, 4 fixed tasks x 4 episodes; not task-disjoint
- Project holdout not used for tuning; parent pretraining exposure unknown
- Frozen generated9 selection only; do not retune using these results
- native FPS, 121 frames or 116/119-frame prefix cap, full-length GT; not old 81-frame protocol
- Public reproduced GTcap scores; not equivalent to private official final score

完整raw/corrected逐项结果、所有配对CI与输入哈希见同目录 comparison.complete.json。
