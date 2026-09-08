# Wan v7 SE(3) 单卡机制验证报告

日期：2026-08-18  
实验：`v7-se3-single-gpu`  
执行设备：物理 GPU6，`world_size=1`  
状态：训练至 step25，`audit25` 未通过，按合同停止；未运行 step50

## 1. 执行摘要

本轮实验回答的问题是：在冻结 Wan2.2、冻结已有 Action Adapter、只训练三个
SE(3) attention channel gate 的情况下，显式左右臂 SE(3) 几何分支能否产生稳定、
方向正确的 action-conditioned trajectory 信号。

最终结论分为两层：

1. **工程与训练链路成立。** 单卡真实 Wan2.2 forward/backward、三步 optimizer
   smoke、gate-only 参数白名单、显存门禁、checkpoint/resume lineage 均通过。
2. **预注册的机制门禁未通过。** step25 没有形成稳定的
   `correct > reverse/shift/swap` 配对分离，也没有得到正向 trajectory probe 改善，
   因此 step50 被正确阻止。

这不能直接等价为“SE(3) 架构已被证伪”。当前 discovery-8 全部来自同一个
`click_alarmclock` task，并且 3/8 episode 的 position/velocity probe 为
`Infinity`。所以本轮足以决定**不继续烧 step50**，但不足以决定永久放弃所有
SE(3) / geometry-aware attention 路线。

## 2. 实验假设与边界

### 2.1 核心假设

在已有 gated Action Adapter parent 上增加轻量、固定左右臂身份的 SE(3) attention
分支；如果分支真正使用动作几何，短训练后应出现：

- correct condition 的误差稳定低于 swap；
- correct condition 的误差稳定低于 reverse/shift；
- position/velocity probe 朝正确方向改善；
- 不破坏左右 head ownership，不支配 frozen Wan residual。

### 2.2 固定不变的部分

- Backbone：Wan2.2-TI2V-5B，冻结。
- Parent：clean gated step10，冻结。
- Parent SHA256：
  `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`。
- 数据：clean-1000，零 dev-fast20 泄漏合同不变。
- Loss：只使用 weighted flow matching。
- 不训练 T5、VAE、Wan backbone、parent adapter、trajectory probe。
- 不加入 LoRA、Depth、V-JEPA、GAN、DMD 或新 auxiliary network。

### 2.3 唯一可训练参数

三个注入 block：`8 / 16 / 24`。

每个 block 训练一个 `(24, 128)` FP32 gate：

```text
3 × 24 × 128 = 9,216 trainable parameters
```

Head ownership 固定为：

```text
left arm  → heads 0..11
right arm → heads 12..23
```

学习率固定为 `2e-5`，AdamW，weight decay 为 0。

## 3. 单卡 lineage 合同

本轮不是七卡实验的降级冒充，而是独立机制验证 lineage：

| 项目 | 值 |
|---|---|
| Contract | `wan-action-v7-se3-single-gpu/1` |
| Checkpoint contract | `wan-action-v7-se3-single-gpu-checkpoint/1` |
| World size | 1 |
| Physical rank mapping | `[6]` |
| Max steps | 50 |
| Checkpoints | 10 / 25 / 50 |
| GPU visibility | 只允许 GPU6 |
| Persistent root | `/data/di/worldarena2_track1_20260815/runs/v7-se3-single-gpu/` |

Replay 从可信七卡 v6 replay 中逐 step 选择物理 rank6 的 record，保留原始
sample/noise/timestep，不允许与七卡 checkpoint 交叉 resume。

### 3.1 固定 provenance

| Artifact | SHA256 |
|---|---|
| Source clean-1000 manifest | `fc54f851099ea213431efcb64a2fcbfd5c01335e18404f685768f00ece89b289` |
| V7 replay logical digest | `98203b3217b17302dd47e286604630b408b0cbc6c5bc5b3372ad525b17b56738` |
| V7 replay file | `58c3221cc52f3d9513a69e96e9d41d142b9ae396913986183f30f70dddbac956` |
| V7 cached manifest file | `10529db27c4a5c3c41ea37045604510bf1ef444cf81d0026ed58327907aba4e8` |
| Cache logical binding | `64ca793f09ca71de9c2bec4710e373d6aa74f45a07b4ca1363cc000c2405bf94` |
| Runtime source closure | `d3f41158f1452d5e8bc8e41b697e1e4294225692b7fc8359048211d758aad25e` |

最终重新执行 source closure validator，结果仍为同一
`d3f411...ad25e`，与 preflight 和两个 checkpoint 内记录一致。

## 4. 启动前修复记录

本节非常重要：这些错误均发生在正式 checkpoint 产生之前。没有任何失败尝试
被当成训练结果，也没有污染 step10/step25。

### 4.1 数据与 provenance 接线

1. Leakage receipt 的正式 split key 是 `dev-fast-20`，cache/trainer 曾使用旧别名
   `dev-fast20`。两处已统一为 receipt 原始键名。
2. 旧 v6 replay 绑定 cached-manifest hash，而 v7 pin 绑定 source-manifest hash。
   已证明两者生成的 100 个 step、共 700 个 rank record 完全一致，再保留旧文件
   备份并原子生成 canonical source-bound replay。
3. 第一版 v7 merged manifest 错以 source manifest 为基底，丢失 latent/context/
   action raster/pose statistics 等训练缓存字段。已改为以
   `clean-1000.cached.jsonl` 为基底，只按 sample identity 合并 SE(3) sidecar。

### 4.2 官方 Wan2.2 runtime 兼容

1. `rope_apply` 的真实定义在 `wan.modules.model`，不在
   `wan.modules.attention`。
2. 官方 RoPE API 是分别调用 `rope_apply(q, grid_sizes, freqs)` 与
   `rope_apply(k, grid_sizes, freqs)`。
3. 官方 output projection 带 bias。Geometry residual 只复用 frozen output weight，
   不重复添加 bias，从而保持 zero-gate 与原 Wan bitwise 等价。
4. 官方 QK RMSNorm 在 3072 维 projection 后、head reshape 前执行；wrapper 已与其
   顺序一致。
5. 官方 Wan backbone 返回 `list[Tensor]`；wrapper 现在保持该结构，不强制改为裸
   Tensor。

### 4.3 单卡 runtime 兼容

1. world size 1 下 FSDP 自动切换 `NO_SHARD`，会尝试 flatten frozen BF16 权重与
   FP32 gate，产生 mixed-dtype 错误。单卡路径改为 unsharded BF16 backbone；七卡
   FSDP 路径未改变。
2. torchrun logical rank 是 0，而 replay 保留 physical rank 6。单卡入口显式映射
   `logical 0 → physical 6`。
3. 非反向传播的 reverse/shift/swap preflight 使用 `torch.no_grad()`，避免无意义
   autograd graph 持有 checkpoint condition lease。
4. 训练 backward 完成后显式释放残留 checkpoint lease；未完成 backward 的第二次
   forward 仍会 fail closed。

### 4.4 针对性回归验证

- SE(3) attention focused Torch suite：15/15 通过。
- Attention + Wan integration focused suite：最终 27/27 通过。
- Trainer/launcher focused suite：8/8 通过。
- Python compile、diff check、source closure validator 均通过。

限制：最终 runtime 修复后没有重新执行整个项目全量 pytest；报告只声明上述
focused suites，不把它扩大表述为 full-suite green。

## 5. Preflight 结果

Artifact：
`runs/v7-se3-single-gpu/mechanism/preflight-gradient-audit.json`

SHA256：
`85be6788959e27f6e95e5dea189f0c7d0496089f0df5e0692df6368e37290fec`

### 5.1 硬门

| 项目 | 结果 |
|---|---:|
| Passed | True |
| Gate gradient finite | True |
| Gate gradient non-zero | True |
| Original parameter gradients | 0 |

Gate gradient L2：

| Block | Gradient L2 |
|---|---:|
| 8 | 0.00422618 |
| 16 | 0.00274941 |
| 24 | 0.00111491 |

说明优化信号确实能到达三个 gate，且 block8 初始梯度最强。

### 5.2 零 gate counterfactual FM loss

| Condition | FM loss |
|---|---:|
| Correct | 0.16461086 |
| Reverse | 0.16518331 |
| Shift | 0.16570401 |
| Swap | 0.16391936 |

preflight 只证明梯度存在，不证明语义已正确。尤其 zero-gate 下 swap loss 甚至略低于
correct；四者差异都很小，不能将其解读为已有 causal separation。

## 6. Production smoke

Artifact：
`runs/v7-se3-single-gpu/smoke/production-smoke.v7.json`

SHA256：
`194dd563bfe044e6c14ea18a17942a62ab9d6b41de07c41aa8a8f6aa5eb569d5`

连续执行三次：

```text
forward → backward → optimizer.step → zero_grad
```

| 指标 | 结果 |
|---|---:|
| Passed | True |
| Peak allocated | 14,118,126,080 bytes，约 13.15 GiB |
| Peak reserved | 14,612,955,136 bytes，约 13.61 GiB |
| Hard limit | 22 GiB |
| Step time | 2.648 / 2.275 / 2.276 s |
| Mean step time | 2.400 s |
| Gate gradient seen | True |
| Original gradient seen | False |

因此单张 4090 24GB 足以执行本轮 gate-only v7 训练，显存余量约 8.4 GiB
（按 reserved 对 22 GiB 门禁计算）。

## 7. 训练 checkpoint

### 7.1 Step10

- 文件：`step-000010.pt`
- SHA256：`d77977f975f634eeb4e621c05c15647504afbf589de46ad5cbbb940ca1237bdb`
- 大小：117,373 bytes

Gate L2：

| Block | L2 |
|---|---:|
| 8 | 0.00482527 |
| 16 | 0.00535825 |
| 24 | 0.00550932 |

### 7.2 Step25

- 文件：`step-000025.pt`
- SHA256：`760a866ded9cb659d0cef9bc07c279bd4d60fc2dd6e4c7121357ca3b1200b47d`
- 大小：117,437 bytes

Gate L2：

| Block | L2 | 相对 step10 |
|---|---:|---:|
| 8 | 0.00810799 | +68.0% |
| 16 | 0.01047516 | +95.5% |
| 24 | 0.01098639 | +99.4% |

Gate 确实持续更新，没有 zero-gradient 或 branch collapse。问题不是“参数没训练”，
而是更新后的方向没有形成预期的反事实语义分离。

## 8. Audit25 正式结果

Artifact：
`runs/v7-se3-single-gpu/mechanism/audit-step-000025.json`

SHA256：
`45d83ccc2ac08dcc365898afba10417e0767033978ad49ac1e52b8d385710ebf`

### 8.1 预注册 gate

```json
{
  "pass": false,
  "reasons": [
    "no_paired_counterfactual_separation",
    "no_positive_probe_direction"
  ],
  "step": 25
}
```

| 指标 | 结果 |
|---|---:|
| Routing retention | 1.0 |
| Head ownership violation | False |
| Residual domination | False |
| Absent-arm output | False |
| Non-finite model output | False |
| Aggregate wins | 1 |
| Correct vs reverse wins | 2/8 |
| Correct vs shift wins | 1/8 |
| Correct vs swap wins | 2/8 |
| Position improvement | NaN |
| Velocity improvement | NaN |

结构性安全项全部正常：左右 head ownership 没破坏、routing 保留、residual 没接管
backbone。失败集中在真正需要的语义项：correct condition 没有稳定优于反事实。

### 8.2 Probe coverage

8 个 discovery episode 中：

- 5 条有有限 position/velocity probe；
- 3 条为 `Infinity`；
- 8 条全部来自 `click_alarmclock` task。

无效 probe episode：

- `episode_000007`
- `episode_000011`
- `episode_000015`

因此总体 `position_improvement` 与 `velocity_improvement` 成为 NaN。JSON 中
`non_finite=false` 表示模型计算本身没有 NaN/Inf，并不表示 trajectory probe 每条
都有有效观测。

### 8.3 有效 5 条的补充统计

这些统计是事后诊断，不替代预注册 gate。

| Metric | Correct | Reverse | Shift | Swap |
|---|---:|---:|---:|---:|
| Mean FM loss | 0.243628 | 0.244393 | 0.243874 | 0.243729 |
| Median FM loss | 0.161534 | 0.163609 | 0.161836 | 0.161781 |
| Mean position error | 1.009206 | 1.001773 | 0.985019 | 1.010029 |
| Median position error | 0.886199 | 0.816948 | 0.882407 | 0.888717 |
| Mean velocity error | 0.877454 | 0.891985 | 0.871839 | 0.873232 |
| Median velocity error | 0.727278 | 0.731982 | 0.733630 | 0.724117 |

Position paired wins（correct error 更低）：

| 对比 | Wins |
|---|---:|
| Correct vs reverse | 2/5 |
| Correct vs shift | 1/5 |
| Correct vs swap | 2/5 |

Velocity paired wins：

| 对比 | Wins |
|---|---:|
| Correct vs reverse | 5/5 |
| Correct vs shift | 2/5 |
| Correct vs swap | 1/5 |

唯一一致信号是 correct velocity 对 reverse 为 5/5；但 correct 对 shift/swap 不成立，
position 也没有一致优势，所以不足以晋级。

## 9. 结果解释

### 9.1 可以确定的事实

1. SE(3) cache、官方 Wan attention integration 和 gate-only backward 已真实运行。
2. 三个 gate 都获得有限、非零梯度，并在 25 step 内持续增长。
3. 训练没有破坏 frozen parameter whitelist、head ownership 或 residual budget。
4. correct condition 没有在 discovery audit 上稳定优于 swap/shift/reverse。
5. 按既定合同停止 step50 是正确行为。

### 9.2 不能从本轮推出的结论

1. 不能说“所有 SE(3) conditioning 都无效”。只测了 gate-only、3 blocks、25 step。
2. 不能说“视频 Trajectory 一定下降”。本轮没有生成 matched video，也没有运行
   SAM3/官方 Trajectory。
3. 不能说“七卡正式实验会得到同一数值”。单卡 lineage 明确禁止冒充七卡 lineage。
4. 不能根据 8 条同 task、其中 3 条 probe 无效的数据判断总体任务分布。

### 9.3 当前最可能的技术解释

Gate 能学、结构约束也正常，但 weighted flow matching 对 9,216 个 channel gate 提供的
优化方向不足以把“正确 SE(3) command”与“swap/shift/reverse command”稳定拉开。
模型更像学到一个小的几何 residual 调整，而不是 command-specific trajectory policy。

这与此前项目结论一致：内部 action sensitivity 或结构 routing 正确，不自动等于最终
image-space gripper trajectory 正确。当前缺口仍更接近最终视觉轨迹监督，而不是继续
增加训练步数。

## 10. 评审建议与下一步

### 10.1 立即决策

- **不运行 step50。** audit25 已明确 fail。
- **不将 step10/step25 作为视频候选或 Stage2 parent。** 它们只保留为机制证据。
- **不立即扩成六层架构。** 三层 gate-only 尚未证明 command separation，六层只会扩大
  未验证机制。
- 保留所有 checkpoint、receipt 和 audit，不删除，供后续归因。

### 10.2 最小补证方案

若希望在完全放弃 v7 前再花一次很小成本，建议只做审计修复，不再训练：

1. 从 clean-1000 中版本化选择 20 条 probe-observable discovery 样本；覆盖至少多个
   task，而不是 lexicographic-first-8。
2. 对 step10、step25 和 zero-gate parent 运行同一套 no-grad audit。
3. 只看 correct-vs-swap/shift 的 paired separation 和 probe coverage。
4. 如果 step10/25 仍无稳定正信号，正式结束 v7 gate-only 路线。

该补证只需要模型 forward，不需要 optimizer.step；成本远低于继续训练。

### 10.3 若直接转下一架构

优先级应是：

```text
直接约束最终 image-space gripper trajectory
    > 继续扩大 SE(3) channel gate
    > 继续扫 LR / step / block 数
```

也就是保留已经验证正确的数据、相机几何、左右臂身份与时序合同，但把主要训练信号
对准最终可观测 gripper path，而不是只依赖全局 flow matching 间接塑造 gate。

## 11. Artifact 索引

远端根目录：

```text
/data/di/worldarena2_track1_20260815/runs/v7-se3-single-gpu/
```

关键文件：

```text
mechanism/preflight-gradient-audit.json
smoke/production-smoke.v7.json
mechanism/step-000010.pt
mechanism/step-000025.pt
mechanism/audit-step-000025.json
mechanism/source-sync-closure.final.json
mechanism/preflight.log
mechanism/train10.log
mechanism/train25.log
mechanism/audit25.log
```

截至报告生成时：

- GPU6：6 MiB，0% utilization；
- 无 v7 训练进程；
- `/data`：约 3.5 TB 总量，2.2 TB 已用，1.2 TB 可用，65%。

## 12. 最终判定

本轮实验的准确表述是：

> v7 三层 gate-only SE(3) 机制已完成真实单卡工程验证与 25-step 训练；运行稳定、
> 显存合格、梯度与结构安全门均通过，但预注册 discovery audit 未形成稳定的
> command-specific counterfactual separation，因此按合同停止于 step25。由于 audit
> 样本单一且 3/8 probe 无效，本轮足以否决继续 step50，却不足以否决所有 SE(3)
> architecture。下一步应先做一次低成本、observable、多 task 的审计补证；若仍失败，
> 转向直接视觉 gripper trajectory supervision。
