# Native16 完整15项配对结果

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
