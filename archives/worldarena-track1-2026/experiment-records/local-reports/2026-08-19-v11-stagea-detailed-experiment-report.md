# v11 WorldArena-Balanced Bimanual World Controller

## Stage-A 单卡机制实验详细报告

| 项目 | 内容 |
|---|---|
| 实验日期 | 2026-08-19，Asia/Shanghai |
| 赛道 | WorldArena 2.0 Track 1 / `video_quality_ood` |
| 实验代号 | `v11-worldarena-balanced-bimanual-world-controller` |
| 实验阶段 | Stage A：左右臂 persistent closed-loop controller 机制验证 |
| 运行服务器 | `huazhi@183.147.142.110:9000` |
| 训练 GPU | 物理 GPU6，单卡 RTX 4090 24 GiB |
| world size | 1 |
| 模型基础 | Wan2.2-TI2V-5B |
| parent | frozen `clean-gated-step10` |
| 正式运行根 | `/data/di/worldarena2_track1_20260815/runs/v11-worldarena-balanced-bimanual-world-controller` |
| 有效 checkpoint | `gpu6/step-000100.pt` |
| 最终状态 | **STOP at step100** |

---

## 1. 执行摘要

本轮实验完整执行了以下链路：

```text
架构与合同实现
    -> 远端 Torch 回归
    -> source closure / lineage 校验
    -> production preflight
    -> GPU6 production-shape smoke
    -> bounded step100 训练
    -> 固定 audit-20 observed mechanism audit
    -> STOP / 不进入 step500
```

工程链路全部健康：

- 远端最终回归测试 `57/57 passed`；
- 可训练参数 `40,244,742`，符合 `[40M, 80M]` 合同；
- frozen Wan 和 frozen parent 无梯度；
- production smoke 峰值 allocated `14.379 GiB`、reserved `15.977 GiB`；
- 100 个 optimizer step 全部完成，无 OOM、NaN、Inf；
- 平均 step time `5.634 s`；
- tokenizer、state updater、visual read、visual write、gate、EEF head 六类梯度均持续非零；
- EEF 可观测样本损失从前 10 step 平均 `0.69298` 降至最后 10 step 平均 `0.06993`。

但核心时间机制门失败：

| 指标 | observed audit-20 | 门槛 | 判定 |
|---|---:|---:|---|
| phase `+1` paired wins | 13/20 | >=16/20 | FAIL |
| phase `-1` paired wins | 8/20 | >=16/20 | FAIL |
| phase `+1` mean margin | +0.001255 | 稳定为正 | 边缘正向 |
| phase `-1` mean margin | -0.000875 | 稳定为正 | FAIL |

更关键的是逐样本结构：

```text
同时通过 +1 和 -1： 1/20
只通过 +1：         12/20
只通过 -1：          7/20
两边都不通过：       0/20
```

这不像随机崩坏，更像每个 episode 的预测轨迹存在明确的单方向相位偏置：模型常常更接近当前目标的一侧邻帧，而不是同时优于前后两个邻帧。

因此，本轮结论是：

> v11 的 persistent bimanual controller 具备可训练性、左右臂隔离、局部视觉读写和 EEF 表征学习能力；但当前 action/EEF/objective 组合没有建立可靠的双向 latent-time binding。按照事先定义的 health gate，实验在 step100 停止，不允许用“再多训 400 step”掩盖机制失败。

本轮没有解码 RGB，没有运行 official WorldArena、WorldArena_VLM 或 WorldArena_JEPA。因此本报告**不声称** v11 改善了最终 Trajectory Accuracy、JEPA Similarity、画质指标或 EWMScore。

---

## 2. 实验背景与研究问题

### 2.1 前序实验提供的证据

此前 frozen-backbone 小模块路线已经反复暴露同一矛盾：

- action 模块有参数、有梯度；
- 一些 action audit 或 EEF 中间指标可以改善；
- 但最终视频中的 arm identity、temporal role 和 gripper trajectory 不稳定；
- 单纯增加 Adapter、LoRA、SE(3) gate 或训练步数不能可靠转化为 Trajectory 改善。

v10 进一步表明：

- phase `+1/-1` 可以在某些中间机制 audit 上达到 `19/20`、`20/20`；
- reverse/swap separation 仍弱；
- relation-enabled path 没有稳定优于 forced-zero path；
- hidden EEF 可以收敛，但不能证明最终 arm binding 成立。

因此 v11 不再继续堆一个 additive residual，而是引入真正分离的左右臂 persistent state。

### 2.2 本轮唯一核心假设

Stage-A 只回答一个问题：

> 当左右臂拥有永久隔离、能读取当前视觉 hidden、再局部写回 Wan visual stream 的 persistent controller state 时，模型能否在不破坏 frozen Wan/parent 分布的前提下，稳定学习每只臂的动作绑定和时间相位？

### 2.3 本轮不回答的问题

Stage-A 明确不研究：

- object slot；
- contact / grasp state；
- coordination slot；
- V-JEPA；
- depth branch；
- GAN / DMD；
- text/instruction router；
- native Wan attention 解冻；
- 更多 action-video 数据扩容；
- 1000 条 official test 视频生成。

这些边界用于保持实验可归因：若 Stage-A 失败，应知道失败发生在最基础的 arm-state / visual-read / temporal-binding 机制，而不是被更多模块掩盖。

---

## 3. 模型架构

### 3.1 总体数据流

```text
cached noisy latent + cached text embedding
                    |
                    v
         frozen Wan2.2-TI2V-5B
                    |
      frozen clean-gated raster parent
                    |
     +--------------+--------------+
     |                             |
left action content          right action content
     |                             |
left tokenizer               right tokenizer
     |                             |
C_left: 21x4x384          C_right: 21x4x384
     |                             |
     +---- controller stage 6 -----+
     +---- controller stage 16 ----+
     +---- controller stage 24 ----+
                    |
          local support writeback
                    |
              frozen Wan head
                    |
             flow prediction
```

### 3.2 每臂 persistent state

每只臂拥有完全独立的状态：

```text
C_left, C_right: (B, 21, 4, 384)
```

四个 modality slots：

1. translation：`dx, dy, dz, ||dxyz||`；
2. rotation：`SO(3) log-map + angle`；
3. image motion：`uv_start, uv_end, duv`；
4. gripper：`opening_start, opening_end, delta`。

每臂还有独立 anchor MLP，对相机坐标中的初始 SE(3) anchor 编码。左右 tokenizer 不共享参数。

时间合同：

```text
slot[:, 0] = exact zero
slot[:, t] = action interval t-1, t=1...20
```

Stage-A 没有跨时间 slot self-attention。每个 latent time 的 slot 只与同一 latent time 的视觉 token 交互。

### 3.3 Controller stage

controller 插在 frozen Wan block：

```text
6 / 16 / 24
```

每个 stage、每只臂执行：

```text
local_visual = gather(X, arm_support, K<=48)
visual_read  = CrossAttn(query=arm_slots, key/value=local_visual)
slots_next   = slots + Update(slots, visual_read)
local_write  = CrossAttn(query=local_visual, key/value=slots_next)
X_next       = X + Scatter(local_write * support)
```

设计约束：

- 每臂最多 gather 48 个 support tokens；
- 选择是确定性的；
- 左右 support 可以重叠；
- 重叠区域两路 residual 相加，不做 winner-take-all；
- absent arm 的 slot/read/write 必须精确为零；
- visual write 投影与 FP32 gate 从零开始；
- controller state 不保存在模块全局属性中，避免跨 batch 泄漏。

### 3.4 EEF head

EEF head 只读取 `visual_read`，不直接读取完整 action-bearing slot。目标是避免 head 仅从 action 指令“抄答案”，要求它从视觉 hidden 中读取 gripper evidence。

输出：

```text
stage 6/16/24
x left/right
x 21 latent frames
x 15x20 heatmap
```

监督只在 RGB-observable 的 arm/time 标签上启用。

### 3.5 可训练参数分布

| 参数族 | 参数量 | 占总可训练参数 | tensor 数 |
|---|---:|---:|---:|
| visual read Q/K/V/O | 15,925,248 | 39.571% | 24 |
| visual write Q/K/V/O | 15,925,248 | 39.571% | 24 |
| state updater | 5,315,328 | 13.208% | 24 |
| EEF head | 1,575,936 | 3.916% | 12 |
| tokenizer + anchor | 1,502,976 | 3.735% | 42 |
| FP32 gates | 6 | <0.001% | 6 |
| **总计** | **40,244,742** | **100%** | **132** |

Wan backbone、native attention、FFN、norm、text cross-attention 和 frozen raster parent 全部不在 optimizer whitelist。

---

## 4. 数据集、划分与零泄漏合同

### 4.1 数据规模

本轮复用 v10 的 action-video 数据合同：

| 集合 | 数量 | 用途 |
|---|---:|---|
| source pool | 2,100 | 全部候选 action-video 样本 |
| optimizer pool | 2,060 | 正式训练 replay |
| audit-20 | 20 | checkpoint 机制门禁 |
| dev-20 | 20 | 后续视频选择，Stage-A 未使用 |
| official test | unavailable | 训练、审计、选择均不可访问 |

数据来自 42 个 RoboTwin clean tasks，每个 source task 50 episodes。optimizer、audit、dev 按 episode identity 严格互斥。

### 4.2 optimizer strata

| stratum | optimizer 数量 | 比例 |
|---|---:|---:|
| single_dominant | 905 | 43.93% |
| bimanual_heavy | 496 | 24.08% |
| mixed | 406 | 19.71% |
| quiet | 253 | 12.28% |
| **合计** | **2,060** | **100%** |

### 4.3 audit-20 构成

audit-20 覆盖 20 个不同 tasks：

| stratum | 数量 |
|---|---:|
| single_dominant | 11 |
| bimanual_heavy | 3 |
| mixed | 3 |
| quiet | 3 |

audit tags 包含：

- `single_arm`；
- `bimanual`；
- `sequential`；
- `crossing_or_overlap`。

### 4.4 零泄漏证据

- optimizer 样本数：2,060，sample id 唯一；
- audit 样本数：20，20 个不同 tasks；
- 实测 optimizer/audit sample overlap：`0`；
- dev 不进入训练、calibration 或 observed audit；
- official test 状态固定为 `unavailable`；
- replay、manifest、cache、source receipt 均用 SHA256 绑定；
- 所有持久产物只写入 `/data/di/worldarena2_track1_20260815`。

---

## 5. 条件输入与 counterfactual 设计

### 5.1 Correct condition

每只臂包含：

- 相机坐标 anchor SE(3)；
- 20 个 interval translation；
- 20 个 interval rotation log-map；
- 20 个 interval UV start/end/delta；
- 20 个 interval gripper start/end/delta；
- 21 帧 arm presence；
- 20 段 motion active；
- 21 帧 `15x20` support tube。

### 5.2 训练 negative

正式 replay 固定循环：

```text
wrong-left -> wrong-right -> active-arm-null -> repeat
```

语义：

- `wrong-left`：保持 left identity、anchor、presence、support、destination time，只把 left motion content 换成 right motion content；
- `wrong-right`：对称操作；
- `active-arm-null`：只有样本恰好单臂 active 时才 eligible，把 active arm motion content 置零；
- frozen raster parent 始终读取 correct raster，不跟随 negative 改动；
- counterfactual 只测试新 controller 是否使用正确的 arm action content。

100 step 的实际分配：

| negative | step 数 | eligible arm 次数 | weighted wrong-half mean |
|---|---:|---:|---:|
| wrong-left | 34 | 34 | 0.027254 |
| wrong-right | 33 | 33 | 0.027286 |
| active-arm-null | 33 | 16 | 0.013252 |

`active-arm-null` 有 17/33 step 因样本不是严格 XOR 单臂 active 而不产生 ranking 梯度。这是合同预期行为，不是运行错误，但意味着 100 step 中有效 counterfactual arm comparisons 总计只有 83 次。

---

## 6. 训练目标与梯度路径

### 6.1 Correct path

每 step 的 correct path：

```text
L_correct = L_weighted_FM + lambda_eef * L_visual_EEF
```

FM 使用 cached latent、cached text embedding、correct raster/support 和原有 flow-matching target。

### 6.2 Binding path

每 step 还运行一个 wrong condition。为控制单卡显存，pairwise binding 被拆成两个 sequential halves：

```text
1. correct-ref forward, no grad
2. wrong forward, correct energy detached
3. correct forward, wrong energy detached
```

区域能量只在对应 arm support 内计算。平滑 pairwise ranking 目标要求：

```text
E_correct < E_wrong
```

总梯度等价于完整 pairwise objective，但不同时保留 correct/wrong 两张大图的反向图。

### 6.3 梯度标定

固定 calibration batch 上测得：

| 项目 | gradient norm |
|---|---:|
| FM | 0.00726648 |
| binding | 0.04609367 |
| EEF | 0.02952238 |

目标梯度比：

```text
binding / FM = 0.5
EEF / FM     = 0.2
```

因此冻结：

| lambda | 数值 |
|---|---:|
| `lambda_binding` | 0.07882299 |
| `lambda_eef` | 0.04922694 |

训练期间不重新标定，避免 checkpoint 间目标漂移。

### 6.4 本轮没有的 loss

Stage-A 没有：

- phase ranking loss；
- final position loss；
- final velocity loss；
- object / contact / coordination loss；
- V-JEPA / depth / adversarial loss。

这对结果解释非常重要：phase 是 step100 health gate，但不是显式训练项；它只能通过 per-time EEF supervision 和 Wan/controller 的时间绑定间接形成。

---

## 7. Optimizer、LR 与 checkpoint 合同

Optimizer：AdamW，`betas=(0.9, 0.999)`，`eps=1e-8`，global clip norm `1.0`。

| 参数族 | base LR | step100 LR | weight decay |
|---|---:|---:|---:|
| tokenizer | 1.0e-4 | 9.9878e-5 | 0.01 |
| updater | 5.0e-5 | 4.9939e-5 | 0.01 |
| read | 5.0e-5 | 4.9939e-5 | 0.01 |
| write | 5.0e-5 | 4.9939e-5 | 0.01 |
| gate | 1.0e-3 | 9.9878e-4 | 0.00 |
| EEF | 1.0e-4 | 9.9878e-5 | 0.01 |

LR schedule：

```text
step 0-50: linear warmup
step 50-2060: cosine decay
minimum factor: 0.2
```

checkpoint 只允许：`100 / 500 / 2060`。

step100 checkpoint 记录：

- `completed_step=100`；
- `samples_seen=100`；
- `CUDA_VISIBLE_DEVICES=6`；
- `world_size=1`；
- 132 个 trainable tensors；
- 完整 AdamW moments；
- scheduler `last_epoch=100`；
- parent/source/replay/data/audit/cache/calibration lineage；
- runtime maximum gradient-family telemetry。

---

## 8. Source closure 与复现性

训练 source closure 包含 57 个直接和递归运行依赖文件。

关键 lineage：

| 项目 | SHA256 |
|---|---|
| training source closure | `3f84c4b1883eb2f2683f7c021d4f8dea09cd02a4cd642699ddda3b6bb4429d1a` |
| parent checkpoint | `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2` |
| replay | `239f78fc9cc85156188f6803a00bb29eeb060798bec8477d4a1c7a18cc0448b0` |
| optimizer manifest | `720d1719a2f8393d0d03da171a985421a910baf5c3b40d95fe779e3ffd9c6124` |
| audit manifest | `17dbc1223811cd147d82641937b78523efd6c5380eec6f74c32cc497019e2d06` |
| relation cache normalization | `869ff9718602c6fa23445cd3399283e256cead359476af81024969761f2a0058` |
| loss calibration | `b7c7146c769e7deab1889bc289e5fcaf281edb46ac617bdd8589792f665cc673` |

唯一 legacy byte-pin exception：

```text
src/worldarena_baseline/skeleton.py
SHA256 93cb2aa340a725e7b0a7acd37450334bf46c6483c7baf3646bd14ca3e09a8143
```

其余 closure 文件要求 Git tracked、index clean、worktree clean。远端每次启动在 CUDA/NCCL 前重新计算 closure 并与正式 receipt 比较。

---

## 9. 实施过程与问题闭环

### 9.1 第一轮 preflight：参数量不足

初始 trainable count：

```text
39,943,686
```

低于 `[40M, 80M]` 硬门。检查发现 action anchor 只存在于数据结构，未通过独立 per-arm MLP 进入四个 slots。

修复：

- 左右臂分别增加 `6 -> 384 -> 384` anchor MLP；
- anchor 注入每个 interval slot；
- latent0 仍保持 exact zero；
- 新增“anchor 改变 slots、但不改变 latent0”回归。

修复后 trainable count：

```text
40,244,742
```

### 9.2 第二轮 preflight：BF16 time embedding dtype 错误

错误：

```text
RuntimeError: mat1 and mat2 must have the same dtype,
but got Float and BFloat16
```

根因：v11 显式 Wan block loop 禁用了 outer BF16 autocast，和官方 Wan time-embedding 路径不一致。

修复：

- CUDA time embedding 使用官方 Wan 的 FP32 autocast boundary；
- CPU test 保留 outer BF16 autocast 行为；
- CUDA 路径显式验证 time embedding / time projection 为 FP32；
- 远端 model-focused Torch 测试通过。

### 9.3 audit100 占位逻辑

原 launcher audit100 曾输出：

```text
phase_plus_wins = 18
phase_minus_wins = 18
```

代码审查确认这是固定占位值，不是 20 条真实前向结果。占位逻辑曾错误生成 `step-000100-gated.pt`。

处理：

- 不使用该结果继续 step500；
- 新增独立、只读、hash-recorded observed health audit；
- 重新加载有效 `step-000100.pt`；
- 对固定 audit-20 逐条真实前向；
- 实测 phase、slot、write、FM 和 repeated-forward state；
- 在正式 run root 写入 `STOPPED.json`；
- 将旧 `audit-0100.json` 与 `step-000100-gated.pt` 标记为无效产物。

这次纠错是本轮最重要的流程发现：如果只看 launcher 的占位 gate，模型会被错误放行到 step500。

---

## 10. 测试结果

远端正式环境最终测试：

```text
57 passed / 0 failed
```

覆盖：

- action state shape/presence/counterfactual；
- independent arm tokenizer 与 anchor；
- sparse support gather/scatter；
- overlap additivity；
- left/right controller parameter isolation；
- visual read/write/EEF shape；
- absent arm exact zero；
- explicit Wan block loop；
- BF16 autocast；
- frozen parent/backbone；
- binding objective 与 sequential-gradient equivalence；
- optimizer whitelist / LR / checkpoint / resume；
- step gate evaluator；
- launcher GPU6-only / source-before-CUDA；
- observed audit soft-argmax、formal-root 与 temporal mask。

---

## 11. Production preflight

Preflight 加载真实 Wan2.2-TI2V-5B 权重、真实 parent、真实 cached latent/context/action 条件。

结果：

- trainable count：`40,244,742`；
- trainable tensors：132；
- parent SHA：通过；
- replay：通过；
- optimizer/audit manifest：通过；
- cache normalization：通过；
- source receipt：通过；
- loss calibration：通过；
- T5/VAE 未进入训练热路径。

---

## 12. GPU6 Production smoke

真实 shape 下执行：

```text
warmup
forward -> backward -> optimizer.step -> zero_grad
x 3
```

### 12.1 显存

| 指标 | bytes | GiB | 门槛 |
|---|---:|---:|---:|
| peak allocated | 15,439,620,608 | 14.379 | <22 |
| peak reserved | 17,154,703,360 | 15.977 | <22 |

### 12.2 三步详情

| iteration | negative | eligible | FM | EEF | weighted wrong half | grad norm | step time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | wrong-right | 1 | 0.192872 | 0.693708 | 0.027318 | 0.011686 | 5.717 s |
| 2 | active-arm-null | 0 | 0.191611 | 0.693368 | 0.000000 | 0.006363 | 5.608 s |
| 3 | wrong-left | 1 | 0.328441 | 0.693588 | 0.027318 | 0.004736 | 5.605 s |

所有六个 trainable families 在 smoke 窗口内至少一次出现非零梯度。

Smoke 决策：**PASS**。

---

## 13. step100 训练结果

### 13.1 运行效率

| 指标 | 结果 |
|---|---:|
| 完成 step | 100/100 |
| 平均 step time | 5.6336 s |
| median step time | 5.6198 s |
| P90 step time | 5.6316 s |
| 最慢 step | 6.1310 s |
| 纯 optimizer wall time | 563.36 s / 9.39 min |
| OOM | 0 |
| NaN / Inf | 0 |

按当前速度外推，额外训练 step100->500 约需 37.6 分钟；完整 2060 step 约需 3.22 小时。该估计不包含 checkpoint/audit/权重加载，且本轮因 gate 失败未继续。

### 13.2 分段曲线

| steps | FM mean | observable EEF mean | EEF eligible steps | wrong-half mean | grad-norm mean | step-time mean |
|---|---:|---:|---:|---:|---:|---:|
| 1-10 | 0.244750 | 0.692979 | 10 | 0.019110 | 0.009615 | 5.6771 s |
| 11-25 | 0.470069 | 0.688696 | 11 | 0.023665 | 0.014298 | 5.6365 s |
| 26-50 | 0.342423 | 0.660198 | 18 | 0.026227 | 0.026704 | 5.6343 s |
| 51-75 | 0.310179 | 0.441048 | 17 | 0.019635 | 0.040001 | 5.6244 s |
| 76-100 | 0.298227 | 0.097680 | 22 | 0.022871 | 0.023806 | 5.6230 s |

注意：不同 step 使用不同 replay 样本，因此 FM 分段均值不能直接当作 paired improvement。它只能证明数值没有发散。

### 13.3 全程分布

| 指标 | mean | median | min | P90 | max |
|---|---:|---:|---:|---:|---:|
| FM | 0.332693 | 0.244211 | 0.144042 | 0.651625 | 1.055683 |
| grad norm | 0.025734 | 0.018529 | 0.004292 | 0.046272 | 0.247315 |
| step time | 5.633617 | 5.619839 | 5.604722 | 5.631576 | 6.131045 |

### 13.4 EEF 学习信号

可观测 EEF loss：

```text
steps 1-10:   0.69298
steps 26-50:  0.66020
steps 51-75:  0.44105
steps 76-100: 0.09768
last-10:      0.06993
```

这说明 EEF head 与 visual-read state 的训练信号非常强，且在 step50 后快速收敛。

但它只证明中间 head 能拟合当前标签，不证明：

- correct action 优于 counterfactual；
- latent phase 正确；
- flow prediction 中机械臂轨迹改善；
- 解码 RGB 中 gripper trajectory 改善。

### 13.5 Binding signal

eligible wrong-half 的窗口均值：

| 窗口 | eligible comparisons | weighted wrong-half mean |
|---|---:|---:|
| 1-20 | 16 | 0.027301 |
| 41-60 | 16 | 0.027322 |
| 81-100 | 16 | 0.027225 |

该项没有呈现像 EEF 那样的明确下降。由于窗口样本不同、日志没有同时保存 correct/wrong raw energy margins，这不能严格证明 binding 完全没学到；但它是一个明确警告：当前最明显的训练进展来自 EEF head，而不是 counterfactual action separation。

### 13.6 梯度族变化

| family | steps 1-10 mean | steps 91-100 mean | 变化解读 |
|---|---:|---:|---|
| tokenizer | 1.84e-5 | 5.39e-4 | action encoding 梯度增强 |
| updater | 3.49e-6 | 6.26e-4 | persistent state 更新增强 |
| read | 1.05e-3 | 4.46e-4 | 仍稳定非零 |
| write | 5.82e-3 | 1.48e-2 | 最主要梯度路径 |
| gate | 3.72e-6 | 3.66e-4 | 从近零逐渐打开 |
| EEF | 3.69e-4 | 2.43e-3 | 与 EEF 快速拟合一致 |

模型并非只有 EEF head 在更新；write/update/tokenizer/gate 也得到明显梯度。但最终是否形成正确 trajectory 仍需机制和视频证据。

---

## 14. checkpoint 完整性

| 项目 | 结果 |
|---|---|
| 文件 | `gpu6/step-000100.pt` |
| 大小 | 483,096,825 bytes，约 461 MiB |
| SHA256 | `ee9e201cca5717650ed8e651c1575b550992fce686918d17eae538cd68127783` |
| completed step | 100 |
| samples seen | 100 |
| topology | GPU6 / world size 1 |
| optimizer state | 六组完整 AdamW moments |
| scheduler | last_epoch 100 |
| global clip | 1.0 |

---

## 15. observed audit-20 方法

### 15.1 为什么重新实现 audit

原 audit100 的 phase 结果是固定占位值，不能作为实验数据。因此 observed audit 作为独立只读程序：

- 不修改 checkpoint；
- 不修改训练 source closure；
- 独立记录 audit script SHA；
- 验证 checkpoint 原始 training source lineage；
- 逐条加载固定 audit-20；
- 使用与训练一致的 cached latent/text/action 输入；
- 对同一 sample 使用确定性 noise/timestep；
- 实际前向 stage24 EEF logits；
- 计算 current target 与相邻 target 的距离 margin。

### 15.2 Phase margin 定义

对预测位置 `p_t` 与 GT：

```text
d_current = ||p_t - gt_t||^2
d_plus    = ||p_t - gt_(t+1)||^2
d_minus   = ||p_t - gt_(t-1)||^2
```

margin：

```text
margin_plus  = d_plus  - d_current
margin_minus = d_minus - d_current
```

正 margin 表示预测更接近正确时间点，而不是相邻时间点。

### 15.3 其他观测项

- final left/right slot RMS；
- max direct-write RMS；
- audit FM finite/mean；
- frozen parameter gradient absence；
- 相同输入重复 forward 的 bitwise equality；
- smoke memory；
- six gradient families runtime evidence。

---

## 16. observed audit-20 汇总

| 指标 | 结果 | 判定 |
|---|---:|---|
| gradient families | 6/6 nonzero | PASS |
| frozen gradients absent | true | PASS |
| FM finite | true | PASS |
| audit FM mean | 0.356166 | 仅记录 |
| left slot RMS mean | 0.807345 | 无整体塌缩 |
| right slot RMS mean | 0.410579 | 无整体塌缩 |
| max direct-write RMS | 0.028169 | 未爆炸 |
| repeated-forward state leak | false | PASS |
| phase +1 wins | 13/20 | FAIL |
| phase +1 mean margin | +0.001255 | 边缘正向 |
| phase -1 wins | 8/20 | FAIL |
| phase -1 mean margin | -0.000875 | FAIL |
| final decision | STOP | 不进入 step500 |

说明：产物字段沿用了 `controller_to_parent_rms` 名称，但 observed audit 实际记录的是 **max absolute direct-write RMS**，没有除以 parent hidden RMS。它可用于检查 residual 是否爆炸，但不是严格的 controller/parent ratio。这是当前 audit 的一项已知限制。

---

## 17. 逐 episode audit 结果

`W/L` 表示该方向 margin 正/非正。

| # | Task / episode | Stratum | +1 margin | -1 margin | +1/-1 | L slot | R slot | FM | Write RMS |
|---:|---|---|---:|---:|:---:|---:|---:|---:|---:|
| 1 | `click_bell` / `episode_000009` | quiet | +0.000104 | +0.001333 | W/W | 0.0000 | 0.6503 | 0.4991 | 0.00707 |
| 2 | `dump_bin_bigbin` / `episode_000029` | quiet | +0.016287 | -0.005470 | W/L | 0.8957 | 0.0000 | 0.2812 | 0.01984 |
| 3 | `handover_mic` / `episode_000041` | mixed | +0.000606 | -0.000375 | W/L | 1.2196 | 0.4619 | 0.6094 | 0.02817 |
| 4 | `hanging_mug` / `episode_000012` | mixed | +0.010929 | -0.011148 | W/L | 0.8345 | 0.5869 | 0.5244 | 0.01800 |
| 5 | `lift_pot` / `episode_000038` | quiet | +0.006178 | -0.006239 | W/L | 1.0611 | 0.6035 | 0.2394 | 0.02456 |
| 6 | `move_pillbottle_pad` / `episode_000000` | single_dominant | -0.008622 | +0.008567 | L/W | 1.1177 | 0.0000 | 0.1776 | 0.02653 |
| 7 | `move_playingcard_away` / `episode_000049` | single_dominant | +0.006542 | -0.006738 | W/L | 0.0000 | 0.6129 | 0.1679 | 0.00673 |
| 8 | `pick_diverse_bottles` / `episode_000028` | bimanual_heavy | -0.004990 | +0.002677 | L/W | 1.2029 | 0.7056 | 1.0374 | 0.02605 |
| 9 | `pick_dual_bottles` / `episode_000022` | bimanual_heavy | -0.017394 | +0.017396 | L/W | 1.1693 | 0.6953 | 0.2908 | 0.02638 |
| 10 | `place_bread_basket` / `episode_000028` | single_dominant | +0.001798 | -0.001332 | W/L | 1.1844 | 0.0000 | 0.4167 | 0.02626 |
| 11 | `place_burger_fries` / `episode_000009` | bimanual_heavy | +0.003846 | -0.007369 | W/L | 0.9764 | 0.7471 | 0.3301 | 0.02171 |
| 12 | `place_can_basket` / `episode_000011` | single_dominant | +0.000603 | -0.002192 | W/L | 0.9498 | 0.4796 | 0.3602 | 0.02096 |
| 13 | `place_empty_cup` / `episode_000026` | single_dominant | -0.011902 | +0.011932 | L/W | 1.0542 | 0.0000 | 0.1550 | 0.02476 |
| 14 | `place_fan` / `episode_000023` | single_dominant | +0.004860 | -0.004875 | W/L | 1.1819 | 0.0000 | 0.2675 | 0.02723 |
| 15 | `place_object_scale` / `episode_000042` | single_dominant | +0.002184 | -0.002177 | W/L | 1.0659 | 0.0000 | 0.3652 | 0.02572 |
| 16 | `place_phone_stand` / `episode_000043` | single_dominant | +0.007992 | -0.008545 | W/L | 0.0000 | 0.7230 | 0.3616 | 0.00796 |
| 17 | `shake_bottle_horizontally` / `episode_000041` | single_dominant | +0.011247 | -0.008204 | W/L | 1.2724 | 0.0000 | 0.3302 | 0.02697 |
| 18 | `stack_blocks_three` / `episode_000019` | mixed | -0.001027 | +0.001221 | L/W | 0.9610 | 0.4668 | 0.2986 | 0.02096 |
| 19 | `stack_blocks_two` / `episode_000046` | single_dominant | -0.002736 | +0.002649 | L/W | 0.0000 | 0.7539 | 0.1933 | 0.00809 |
| 20 | `stamp_seal` / `episode_000034` | single_dominant | -0.001406 | +0.001398 | L/W | 0.0000 | 0.7249 | 0.2177 | 0.00795 |

样本中某一侧 slot RMS 为零，通常对应该 episode 中该臂不存在或不参与，并不等价于全局 branch collapse。aggregate left/right RMS 均为正，且 bimanual samples 两侧均非零。

---

## 18. 分层 audit 结果

| stratum | N | +1 wins | -1 wins | +1 mean margin | -1 mean margin |
|---|---:|---:|---:|---:|---:|
| single_dominant | 11 | 7 | 4 | +0.000960 | -0.000865 |
| bimanual_heavy | 3 | 1 | 2 | -0.006179 | +0.004235 |
| mixed | 3 | 2 | 1 | +0.003503 | -0.003434 |
| quiet | 3 | 3 | 1 | +0.007523 | -0.003459 |

每个非 single-dominant stratum 只有 3 条，不能据此做可靠总体排名。但至少可以看出：失败不是只发生在 crossing/bimanual，也存在于 single-dominant 和 quiet tasks。

---

## 19. 结果解释

### 19.1 已被证实的部分

1. **实现可训练。** 六类参数都有梯度，优化器和 checkpoint 完整。
2. **显存可控。** 单卡 production-shape 只使用约 14.4/16.0 GiB allocated/reserved。
3. **左右 persistent state 没有整体塌缩。** aggregate slot RMS 均为正；bimanual cases 两侧均能形成状态。
4. **局部 visual read 能拟合 EEF 标签。** EEF loss 在 100 step 内显著下降。
5. **controller write 没有爆炸。** max direct-write RMS 约 0.028。
6. **没有跨 forward 状态泄漏。** 同输入重复前向 bitwise equal。

### 19.2 未被证实的部分

1. correct action 没有被证明稳定优于 wrong-left/wrong-right/null；
2. `+1/-1` 双向 phase binding 没建立；
3. EEF head 下降没有被证明传递到最终 flow prediction；
4. controller-enabled 没有和 force-zero controller 做完整 paired audit；
5. 没有 RGB trajectory proxy；
6. 没有官方 SAM3 Trajectory Accuracy；
7. 没有 WorldArena/VLM/JEPA no-regression 结果。

### 19.3 最强失败信号：单方向相位偏置

19/20 样本只在 `+1` 或 `-1` 的一侧获胜，只有 1/20 同时胜过前后邻帧。

这更符合以下模式：

```text
target(t-1) ---- predicted(t) ---- target(t) ---- target(t+1)
```

或其反方向，而不是：

```text
predicted(t) ~= target(t)
```

也就是说，EEF head 很可能学会了轨迹形状与 arm identity，却存在 episode-level 的提前/滞后偏移。

### 19.4 可能原因，按优先级排列

以下均是待验证假设，不是已证明结论：

1. **EEF label 与 slot interval 的时间合同仍存在有效相位偏差。** 数据合同形式上是 `latent t <- interval t-1`，但 observable EEF 标签来自 81->21 causal packing；两者在真实抓取/停顿段可能并非同一物理时刻。
2. **EEF head 只从 visual read 读图，但 current visual hidden 的时间语义可能已经由 noisy latent/timestep 混合，导致局部 head 偏向邻近状态。**
3. **没有显式 phase loss。** EEF BCE 可以快速拟合热点分布，却未强制当前预测同时远离 `t-1/t+1`。
4. **Counterfactual 目标只改变 arm content，不包含 phase-shift negative。** 它训练 which-arm / null dependence，但没有直接训练 when。
5. **100 samples seen 太少。** 但根据预先 gate，样本不足不能自动授权继续，因为当前 `-1` 方向已经是负均值，而非单纯低置信度。
6. **15x20 EEF head 的低分辨率量化。** 可能压小 margin，但难以单独解释稳定的单方向符号模式。

---

## 20. 为什么不继续 step500

继续训练的唯一正当依据应是：

- step100 工程健康；
- phase 没有灾难性退化；
- action binding 开始形成；
- 没有明显 objective shortcut。

本轮只满足第一项。具体反证：

- phase `-1` 只有 8/20；
- phase `-1` mean margin 为负；
- 只有 1/20 同时通过两个方向；
- wrong-half 窗口均值基本持平；
- 最明显的改善集中在 EEF head，而不是已证实的 action separation。

如果继续到 500，即使 EEF loss 继续下降，也可能只是让偏相位 head 更自信。这样会浪费约 38 分钟 GPU 时间，并降低实验可归因性。

---

## 21. 与 WorldArena 目标的关系

WorldArena Track 1 最终关心的是解码视频。v11 Stage-A 只完成了内部机制验证的第一层。

| WorldArena 关注项 | v11 当前证据 |
|---|---|
| Trajectory Accuracy | 未评测；phase gate 失败，不能晋级 RGB trajectory |
| JEPA Similarity | 未评测 |
| Subject Consistency | 未评测 |
| Motion Smoothness | 未评测 |
| Image Quality / IQA | 未评测 |
| VLM functional metrics | 未评测 |
| 黑帧/坏视频 | 未解码，因此未知 |
| action identity | 中间状态存在，但 final binding 未证明 |
| temporal alignment | observed audit 明确失败 |

所以本轮价值在于机制筛选，而不是 leaderboard 分数。

---

## 22. 有效与无效产物

### 22.1 有效

| 产物 | 用途 |
|---|---|
| `gpu6/preflight.json` | 参数、lineage、calibration 证据 |
| `gpu6/production-smoke.json` | 显存、三步训练、梯度证据 |
| `gpu6/training.jsonl` | 100 step 完整日志 |
| `gpu6/step-000100.pt` | 有效可复核 checkpoint |
| `gpu6/audit-0100-observed.json` | 唯一有效 step100 observed gate |
| `STOPPED.json` | 禁止续训的 fail-closed 标记 |
| `RUN_REPORT.md` | 运行摘要 |

### 22.2 无效或禁止使用

| 产物 | 原因 |
|---|---|
| `gpu6/audit-0100.json` | phase 值是占位常数 |
| `gpu6/step-000100-gated.pt` | 由无效占位 gate 生成，不得作为 step500 parent |

---

## 23. 下一轮最小、可归因实验建议

本报告不自动授权下一轮，但从当前证据出发，最小决策顺序应是：

### 23.1 先做 no-train 时间对齐诊断

对固定 audit-20：

```text
lag = -3, -2, -1, 0, +1, +2, +3 latent steps
```

分别比较：

- action interval endpoint UV vs observable EEF；
- stage24 EEF prediction vs target；
- clean parent / fresh v11 step0 / v11 step100。

必须回答：

1. 最佳 lag 是否稳定为 0？
2. 偏移是数据/packing 已存在，还是训练后产生？
3. 不同 episode 的偏移方向是否与动作阶段、停顿、抓取时刻相关？

### 23.2 若 step0 已偏移

优先修：

- 81->21 label packing；
- interval `t-1 -> latent t` 合同；
- observability 的帧索引；
- action endpoint 与 RGB gripper timestamp。

不应先加新网络。

### 23.3 若 step0 正确、step100 才偏移

保留 v11 架构，只做一个新变量：

```text
L = L_FM + L_binding + L_EEF + lambda_phase * L_phase
```

phase loss 必须同时比较 `t-1` 和 `t+1`，且只在 motion-discriminative / observable token 上启用。先进行梯度范数校准，再跑最多 100 step。不要同时加 object、coordination、V-JEPA 或更多 block。

### 23.4 必须补的 baseline

本轮缺少同一 observed audit 对：

- clean-gated-step10 parent；
- fresh v11 step0；
- v11 step100。

没有这三个 paired baseline，不能判断 step100 的 13/20、8/20 是改善、持平还是恶化。下一轮设计必须先补齐。

### 23.5 晋级条件

只有在固定 audit-20 同时满足：

```text
phase +1 >= 16/20
phase -1 >= 16/20
两方向 mean margin > 0
correct binding 开始优于 wrong arm/null
FM 与 parent 无明显退化
```

才值得解码 4-8 条 RGB，再决定是否跑 matched fast20。

---

## 24. 实验决策

### 24.1 本轮最终判断

```text
工程实现：PASS
单卡可训练性：PASS
显存：PASS
左右状态存在：PASS
EEF 中间监督：PASS
双向 temporal phase：FAIL
RGB trajectory：NOT RUN
官方综合评测：NOT RUN
继续 step500：NO
```

### 24.2 一句话总结

> v11 第一次证明了“左右臂永久隔离的视觉闭环 controller”能在单卡 4090 上稳定训练并快速学会 EEF 中间表示；但它仍把动作时间绑定到相邻 latent 的一侧，20 条中只有 1 条同时分开前后相位，因此当前 objective 尚不足以支撑 WorldArena Trajectory 晋级。

---

## 25. 复核入口

本地报告：

```text
/Users/disaster/workspace/code/worldArena2_video_eval/baseline/
reports/2026-08-19-v11-stagea-detailed-experiment-report.md
```

远端正式运行目录：

```text
/data/di/worldarena2_track1_20260815/runs/
v11-worldarena-balanced-bimanual-world-controller
```

关键文件：

```text
gpu6/preflight.json
gpu6/production-smoke.json
gpu6/training.jsonl
gpu6/step-000100.pt
gpu6/audit-0100-observed.json
STOPPED.json
RUN_REPORT.md
```

