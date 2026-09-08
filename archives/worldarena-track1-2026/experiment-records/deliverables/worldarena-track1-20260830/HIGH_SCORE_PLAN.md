# HZ-World Track 1 冲榜方案（2026-08-30）

## 决策摘要

最终路线不是“8 个随机 LoRA 各生成 1000 条”。采用：

> P0 不可变底座 + 控制能力专家 + 主动预算候选生成 + generated-only 保守路由 + 条件式画质恢复。

P0 始终保留为逐 episode fallback。新增算力只投向可能换片的困难 episode：先用 160 条早筛，首轮最多扩到 600 条；只有实测 ROI 达标才动用剩余预算，绝不默认生成 8,000 条。

## 榜单事实

截至 2026-08-30，没有采用 8 月 29 日最终规则产生的正式榜。官方现有 Space 已被冻结为 interim；最终榜要等 9 月 4 日重新提交后评测。

按官方 Space 当前 `data_loader.py` 对 77 个 Track 1 JSON 的 15 指标算术平均复算，阶段榜头部为：

| 排名 | 模型 | 阶段均值 | 可复现性 |
|---:|---|---:|---|
| 1 | Inspatio_Curious_0.1 | 66.11 | 闭源 |
| 2 | GeWu | 65.91 | 闭源 |
| 3 | K1X | 65.61 | 闭源 |
| 4 | watermelon-plus | 65.04 | 闭源 |
| 5 | X-WM | 64.94 | 闭源 |
| ≈22 | FlowWAM-FiveAges | 63.87 | 完整代码/权重可得 |

HZ-World P0 的 `clean50 corrected15 ≈ 65.92` 是本地开发 proxy，不是官方成绩，不能据此声称排名。

完整 1000 P0 与榜首可直接比较的 6 个非 motion 指标中，最大缺口是 Image Quality：`0.50895 vs 0.6064`；Interaction 是第二缺口：`0.7936 vs 0.8160`。Instruction `0.8666 vs 0.8560` 已达到头部，Perspectivity、Aesthetic 也非常接近。缺失的 JEPA、Trajectory 只能用 clean50 诊断，不能冒充正式 1000 分解。

新规则只 cap Dynamic Degree、Flow Score、Motion Smoothness；clean50 上即使把三项全部完美补到 GT cap，总榜最多约增加 `0.109`。继续追求极端运动、帧数或 FPS 的收益低且可能触发人工审核。

## 当前 8 卡 Winner-SFT 的真实含义

P0 不是一个可训练 checkpoint，而是 Official FlowWAM Stage1 生成的 seed1/seed4 候选经过九项 selector 后的 1000 条视频集合。

当前 8 个 Winner-SFT LoRA：

- 父模型：Official FlowWAM Stage1；
- 训练监督：从 P0 已评分 seed1/seed4 配对中提取的 winner；
- 参数：self-attention q/v LoRA rank8，LR=1e-6，25 steps；
- 作用：把 P0 的选择偏好蒸馏回生成器，属于“继承我们的数据与选择经验”，不是“继续训练 P0 权重”。

因此 loss 降低不代表分数提高。只有在 task-heldout、完整 15 项、GT-cap-corrected 评测中击败 P0，checkpoint 才有资格生成正式候选。

## 四层高分架构

### 1. 生成器专家层

保留三种互补候选，不扩展八个随机 seed：

1. **Technical Quality Residual Expert**：优先修 MUSIQ Image Quality，使用 low-noise video-to-video latent residual 或 late-denoise LoRA；LPIPS/DINO identity 与 optical-flow warp 保护语义和时序。
2. **Contact/Geometry Expert**：将双臂 SE(3)、gripper/action token、2D 轨迹 heatmap 和 contact mask 注入 residual cross-attention，补 Interaction/Trajectory。
3. **Metric-Aware Preference Expert**：用完整逐指标标签重建 pairwise utility；motion 到 cap 后权重归零，Image/Interaction/JEPA/Trajectory 为主，Instruction/Perspective 和一致性项作 veto。
4. **Best Winner-SFT**：当前 8 个 checkpoint 只参加早筛；只有 fresh task-heldout 过门的单一 checkpoint 才进入候选池。

不默认做 adapter soup；只有 holdout 证明两个专家互补且不发生保护项回撤时才合并。

### 2. 主动预算层

先做零生成 quick win：在 task-disjoint fresh40 上验证“现有 seed1/seed4 的 6 个非 motion 指标保守重选”。理论只替换 114 条即可获得约 `+0.05` 总分，但未经 fresh40 不能覆盖 P0。

再用相同 24–40 个、按 task 分层的 episode 对 8 个 checkpoint 和三类新专家做早筛。第一波最多生成约 160 条；只让通过门槛的分支扩量。

对正式 1000 条预算：

- 约 566 条高置信 P0：不重生成；
- 约 434 条明确低尾：按缺口分配 quality/contact/alignment challenger；
- 首轮定向预算 600 条，只有实测 ROI 达标才使用剩余 400 条预算。

绝不因为“还有 1000 条预算”就平均生成。8 卡按 episode shard 并行，而不是同一 episode 横跨多卡。

### 3. Generated-only 路由层

测试阶段禁止使用隐藏 GT RGB、隐藏 GT 指标或官方逐 episode 得分。路由仅使用：

- instruction/text embedding；
- 动作序列长度、单/双臂、gripper 事件、位移/旋转幅度、动作阶段数；
- P0 seed1/seed4 历史分歧与冻结 selector margin；
- 候选视频自身的 VLM alignment、对象/机器人稳定、光流-动作对齐、背景闪烁、深度/几何一致性、解码与黑帧特征。

训练采用按 task 留组的 OOF pairwise uplift predictor。路由必须能弃权：预测 uplift <0.005 或胜率 <0.8 时保留 P0。

### 4. 条件式画质恢复层

只对背景静态占比高、动态 mask 可靠的 episode 开启：去闪烁、颜色稳定、弱去噪/锐化。机器人、gripper、目标物和高光流区域保持原像素，避免破坏 Interaction、Trajectory、JEPA。

禁止抽帧/FPS 操纵、贴 GT RGB、水印/固定文本、test 上搜索参数和大范围重绘交互区域。

## Go/No-Go 门槛

候选必须同时满足：

- corrected15 ≥ P0 + 0.004；
- non-motion12 ≥ P0 + 0.006；
- Image ≥ +0.03（画质专家）或 Interaction ≥ +0.03、Trajectory ≥ +0.01（接触专家）；
- 40 例严格胜至少 24；
- Trajectory、JEPA 各不低于 P0 - 0.003；
- Image/Aesthetic 不下降；
- black0、640×480、逐 episode 帧数不超过 GT。

任一关键门槛失败，候选不扩量，正式 episode 回退 P0。

## 最快执行顺序

1. 先在 fresh40 验证零生成的非 motion 重选；
2. 对 8 个已训练 checkpoint 做统一 24–40 episode 早筛，不扩全量；
3. 并行训练/验证 quality residual、contact/geometry、metric-aware preference 三类专家；
4. 冻结路由特征与阈值后，只对低尾 episode 使用首轮 600 条主动预算；
5. 通过本地全 15 项与格式合同后，使用唯一一次 episode1–100 官方预测试；
6. 根据预测试做一次全局 P0 / P0++ 选择，不做逐 episode 泄漏式调参；
7. 预留 12–18 小时完成 1000 条解码、帧数、命名、README、ZIP 与邮件复核；
8. 仅在最终模型锁定后发送全队唯一一次正式邮件。

## 官方依据

- 最终提交与截止规则：https://v2.world-arena.ai/WorldArena-2.0/final-update.html
- Motion cap 规则：https://v2.world-arena.ai/WorldArena-2.0/motion-quality-evaluation-update.html
- 当前阶段榜 Space：https://huggingface.co/spaces/WorldArena/WorldArena2.0
- FlowWAM 可运行路线：https://github.com/YixiangChen515/FlowWAM_WorldArena
