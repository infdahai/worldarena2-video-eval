# HZ-World 预测试反馈（用户提供）

记录日期：2026-09-03。来源：用户本轮粘贴的官方预测试结果；本记录没有独立读取官方邮件或评分附件。反馈中未给出 ZIP SHA/revision，也未明确三项 motion 是否已经执行 GT cap，因此版本精确绑定和调整后总分仍需核对。

| 指标 | 分数 |
| --- | ---: |
| Image Quality | 0.5034 |
| Aesthetic Quality | 0.4432 |
| JEPA Similarity | 0.9071 |
| Dynamic Degree | 0.2330 |
| Flow Score | 0.1133 |
| Motion Smoothness | 0.6461 |
| Subject Consistency | 0.8111 |
| Background Consistency | 0.8856 |
| Photometric Consistency | 0.2400 |
| Interaction Quality | 0.7820 |
| Trajectory Accuracy | 0.6838 |
| Depth Accuracy | 0.9836 |
| Perspectivity | 0.9620 |
| Instruction Following | 0.8540 |
| Semantic Alignment | 0.8948 |

15项之和=9.9430，直接均值×100=66.2866667。非motion12项均值=0.7458833333。

若反馈三项 motion 已调整，66.2867 即按该口径复算的 EWMScore_P；若未调整，不能从汇总均值还原逐episode GT-cap分数。按[官方motion更新](https://v2.world-arena.ai/WorldArena-2.0/motion-quality-evaluation-update.html)，仅 Dynamic / Flow / Motion Smoothness 使用相应GT上限，其余12项不变。不能把低 motion 原值直接当模型缺口，也不重复封顶。

初步解读：绝对分数较低的非motion项集中于 Photometric、Aesthetic、Image；Instruction、Depth、Perspectivity较强。此为当前样本内诊断，不是与同协议榜首的差距排序；不据此推断排名。JEPA 0.9071 与旧开发集0.958来自不同样本/处理协议，不能认定模型回撤0.0509。

当前下一步：冻结模型与路由，不再新增seed或训练。对16条有真实GT、项目内episode未用于调参的样本，分别评价P0与四seed冻结路由；两组按原生FPS、仅超GT帧数时裁尾的方式评完整15项。该留出仅episode-disjoint，非task-disjoint，parent预训练暴露未知；结果仅补充验证，不再用于调参。JEPA为集合级，配对统计不把它伪装为独立逐episode观测。

用户已批准本次补评100GiB最低余量、20GiB新增预算；当前约141GiB空闲，尚无需删除任何数据。正式1000 ZIP/上传/邮件授权与本次磁盘批准分开。
