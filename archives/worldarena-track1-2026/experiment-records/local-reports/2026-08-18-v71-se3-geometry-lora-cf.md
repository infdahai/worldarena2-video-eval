# Wan-Action v7.1 SE(3) Geometry LoRA-CF 机制实验报告

日期：2026-08-18

实验状态：**step25 硬门失败，已停止；未运行 step50，未解码 RGB，未运行 matched fast20**

## 1. 最终结论

本轮只修改了训练目标，没有扩大模型结构：在冻结的 Wan2.2 与 frozen clean-gated parent 之上，保留 block `8/16/24` 的 rank16 geometry-only Q/K/V/O LoRA 与 channel gate，加入 geometry-only counterfactual softplus ranking loss。

实验结果否定了本轮核心假设：

> 明确监督 correct geometry action 优于 reverse/shift/swap，仍未让冻结 Wan 上的小型 SE(3) branch 在 25 steps 内形成稳定 action semantics。

step25 的固定 audit-20：

| negative | paired wins | 要求 | average ranking margin | position separation | velocity separation |
|---|---:|---:|---:|---:|---:|
| reverse | 7/20 | ≥12/20 | +0.0000331 | -0.000452 | -0.001480 |
| shift（±1 hard negative） | 8/20 | ≥12/20 | -0.0000343 | -0.001227 | -0.001172 |
| swap | 8/20 | ≥12/20 | +0.0000033 | -0.001539 | -0.001445 |

三类均未达到 12/20；shift ranking margin 仍为负；三类 position/velocity separation 全部为负。step25 gate 首项即报：

```text
step25 reverse wins gate failed
```

因此严格按批准的决策树执行：

- 不运行 step50。
- 不解码 4–8 条 RGB。
- 不运行 S1A125 / clean-gated-step10 / v7.1-CF matched fast20。
- 正式停止 small Adapter / geometry gate / frozen-backbone geometry LoRA 这条小参数路线。
- 下一阶段应转为 partial Wan attention unfreeze + SE(3) geometry branch + counterfactual objective + 更大 action-video 数据。

## 2. 本轮唯一变量

### 2.1 完全冻结

- Base：Wan2.2-TI2V-5B。
- Parent：`clean-gated-step10.pt`。
- Parent SHA256：`105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`。
- 原 Wan Q/K/V/O、native RoPE、native attention path：冻结。
- clean-gated parent raster adapter：冻结。
- block：只使用 `8/16/24`。
- geometry LoRA：Q/K/V/O rank16。
- 可训练张量：27。
- 可训练参数：1,188,864。
- LR：`1e-4`。
- 初始化 seed：`20260818`。
- replay、noise、timestep、audit-20：固定。

没有加入 trajectory loss、Pose FiLM、Depth、JEPA、gripper bias、coordination branch 或更多 block。

### 2.2 唯一新增项

```text
L = weighted_FM(correct) + lambda_cf * softplus((E_correct - E_wrong) / tau)
```

其中：

```text
E(action) = robot/action support 区域上的 unreduced weighted FM
```

- `condition_support` 提供 action support，不新增 detector 标签。
- 原 `loss_weight` 与 `valid_mask` 继续参与 energy 计算。
- 全局静态背景不进入 ranking energy 的分母。
- `tau=0.1`。

## 3. Counterfactual 合同

### 3.1 只扰动 geometry branch

所有 correct/wrong forward 均固定：

- `action_raster`：correct，且保持同一 tensor。
- `condition_support`：correct，且保持同一 tensor。
- `action_present`：correct，且保持同一 tensor。
- frozen parent adapter：始终接收 correct action。

只改变：

- `se3_arm_transform`。
- `se3_arm_present`。

因此 correct-vs-wrong energy difference 不会混入 frozen raster Adapter 已有的响应。

### 3.2 三类 wrong action

- reverse：保留 anchor，反转 anchor 后的 geometry 时序。
- shift：训练时交替 `+1/-1 latent step`。
- swap：交换两臂相对运动，但保留左右臂各自初始姿态；不会直接交换绝对 SE(3) anchor。

训练负样本固定循环：

```text
reverse -> shift+1 -> swap -> reverse -> shift-1 -> swap -> ...
```

25 steps 的实际 exposure：reverse 9 次、shift 8 次、swap 8 次。

audit 中 shift 同时计算 `+1` 与 `-1`，每条 episode 取更难的低误差 negative，因此 shift wins 表示 correct 同时经受两种时移方向。

## 4. 两-forward 精确梯度实现

由于 Wan activation checkpoint 的 condition lease 不允许同时保留两张未 backward 的不同 action graph，本轮没有同时驻留 correct/wrong graph。

每 step 使用两次带梯度 forward：

1. wrong forward：计算 `E_wrong` 与 `∂E_wrong/∂θ`，随后释放 checkpoint graph。
2. correct forward：计算 weighted FM、`E_correct` 与 softplus correct-side gradient。
3. 用解析系数补入 wrong-side gradient：

```text
dL/dE_wrong = -sigmoid((E_correct-E_wrong)/tau) / tau
```

该实现与直接同时保留两张 graph 的 pairwise softplus 梯度数学等价，但峰值显存维持单 forward 级别。

## 5. 数据、replay 与零泄露

- 训练集：pinned clean-1000。
- source manifest SHA256：`fc54f851099ea213431efcb64a2fcbfd5c01335e18404f685768f00ece89b289`。
- cache SHA256：`64ca793f09ca71de9c2bec4710e373d6aa74f45a07b4ca1363cc000c2405bf94`。
- replay SHA256：`98203b3217b17302dd47e286604630b408b0cbc6c5bc5b3372ad525b17b56738`。
- dev-fast20：仅作为 zero-leakage exclusion boundary，没有进入训练或 audit-20。
- official test：不可用且未访问。

固定 audit-20：

- 20 个 episode。
- 20 个不同 task。
- 7 left-observable、7 right-observable、6 both-observable。
- 20/20 都有有效 position 与 velocity probe target。
- 与上一轮 v7.1 使用同一 deterministic selection。

## 6. Step0 gate-gradient 标定

preflight contract：`wan-action-v71-cf-preflight/1`。

标定结果：

| 项目 | 数值 |
|---|---:|
| FM channel-gate gradient L2 | 0.00411794 |
| 未缩放 CF channel-gate gradient L2 | 0.03466029 |
| 自动标定 `lambda_cf` | 0.11880849 |
| `tau` | 0.1 |
| 缩放目标 | 1:1 |

step0：

- reverse/shift+1/swap margin 均为 0。
- 三类 ranking loss 均为 `0.69314718 = ln(2)`。
- channel gate 梯度非零。
- Q/K/V/O LoRA 因 zero gate 在第一个 backward 为零，符合初始化合同。
- 27/27 trainable tensors graph-connected 且 finite。
- 原 Wan/parent gradient：0。

## 7. Production smoke

contract：`wan-action-v71-geometry-lora-cf-single-gpu-production-smoke/1`。

- 物理 GPU6，world size 1。
- 三次完整 `forward -> backward -> optimizer.step -> zero_grad`：通过。
- max allocated：14,469,733,376 bytes，约 13.48 GiB。
- max reserved：15,177,089,024 bytes，约 14.13 GiB。
- step time：5.065 / 4.666 / 4.668 秒。
- Q/K/V/O/channel gate 五类均出现非零梯度。
- 原 Wan/parent gradient：0。
- 低于 22 GiB 显存硬门。

## 8. 训练曲线

### 8.1 Ranking loss 与 sampled margin

| negative | exposure | first margin | last margin | first ranking loss | last ranking loss | mean margin |
|---|---:|---:|---:|---:|---:|---:|
| reverse | 9 | 0.000000 | +0.0000734 | 0.693147 | 0.692780 | +0.0000395 |
| shift | 8 | +0.0004326 | +0.0000260 | 0.690986 | 0.693017 | +0.0000626 |
| swap | 8 | -0.0002160 | +0.0000514 | 0.694228 | 0.692890 | -0.0000544 |

局部 sampled training pairs 上确实出现极小的 margin 学习，但量级只有 `1e-5~1e-4`，且未泛化到固定 audit-20。shift ranking loss 后期反而回到接近 `ln(2)`；swap 全程 mean margin 仍为负。

### 8.2 Correct rollout probe 相对 fresh parent

| checkpoint | position error | relative improvement | velocity error | relative improvement | global FM | FM regression |
|---|---:|---:|---:|---:|---:|---:|
| parent/step0 | 0.958399 | — | 0.815522 | — | 0.3703673 | — |
| step10 | 0.960682 | -0.238% | 0.817730 | -0.271% | 0.3703597 | -0.0021% |
| step25 | 0.960036 | -0.171% | 0.817681 | -0.265% | 0.3703544 | -0.0035% |

FM 没有退化，甚至下降约 0.0035%；但 position/velocity 都变差。这再次证明较低 FM 不等于更正确的 action-conditioned trajectory。

## 9. 固定 audit-20

### 9.1 Step10（仅观察）

| negative | wins | position separation | velocity separation | average energy margin |
|---|---:|---:|---:|---:|
| reverse | 6/20 | -0.001797 | -0.001035 | +0.0000090 |
| shift | 5/20 | -0.001562 | -0.001533 | -0.0000564 |
| swap | 5/20 | -0.002261 | -0.001469 | +0.0000086 |

- routing retention：1.0。
- 数值 finite。
- 按合同不在 step10 淘汰，继续至 step25。

### 9.2 Step25（硬门）

| negative | wins | step25 要求 | position separation | velocity separation | average energy margin |
|---|---:|---:|---:|---:|---:|
| reverse | 7/20 | ≥12/20 | -0.000452 | -0.001480 | +0.0000331 |
| shift | 8/20 | ≥12/20 | -0.001227 | -0.001172 | -0.0000343 |
| swap | 8/20 | ≥12/20 | -0.001539 | -0.001445 | +0.0000033 |

- routing retention：1.0，通过。
- correct FM regression：-0.0035%，通过 ≤2% 门。
- 三类 wins：全部失败。
- 三类 average margin：shift 失败。
- parent-relative position/velocity：均负，失败。

结论：step25 硬门失败，停止。

## 10. Checkpoint 与 provenance

| checkpoint | SHA256 | bytes |
|---|---|---:|
| step10 | `5de45c74f0abac6f05c36dd18432d37736ef22bfe5bcf83290c0c3661505e5e6` | 14,300,037 |
| step25 | `51b155c1b5b3e472f21ead4535cfaa82f3508c7cbc5c96cf62d52d8f74580f8f` | 14,300,485 |

- 远端 source commit：`a5a72e940285feafba95ac3eac14b57609fc63b1`。
- preflight source-code closure SHA256：`c7febd5c9692c0d607a3fc725223cb39d6b75475a78d3bdb883fe0f6141e7997`。
- 远端最终 CF + 既有 v7.1 broader regression suite：33 passed。
- GPU6 停止后：6 MiB、0% utilization。
- GPU0–5 与 GPU7 的其他任务没有被停止、重启或占用。

远端产物目录：

```text
/data/di/worldarena2_track1_20260815/runs/v71-se3-geometry-lora-cf-single-gpu/mechanism
/data/di/worldarena2_track1_20260815/runs/v71-se3-geometry-lora-cf-single-gpu/smoke
```

说明：outer audit contract 已是 `wan-action-v71-cf-mechanism-audit-report/1`；内部 metrics 聚合器继续复用旧 retirement contract label。这是字段命名遗留，不影响样本或数值，本轮训练 lineage 不再修改。

## 11. 失败归因

本轮已排除：

- geometry branch 无容量：排除，1.19M 参数均可更新。
- 梯度未连接：排除，27/27 connected。
- LoRA 没更新：排除，Q/K/V/O/gate 在 smoke 中均非零。
- raster counterfactual 混入：排除，raster/support/action_present 始终 correct。
- CF signal 太弱：按 channel-gate gradient 1:1 自动标定，已排除明显量级不足。
- negative 不均衡：reverse/shift/swap 为 9/8/8。
- shift 单方向捷径：训练交替 ±1，audit 同时测试 ±1。
- FM 崩坏：排除，FM regression 为负且远低于 2%。

剩余证据最支持：

> 在原 Wan attention 完全冻结时，这个 geometry side branch 即使获得显式 counterfactual objective，也只能在 sampled energy 上学到极小局部 margin，无法稳定改变最终视觉 trajectory representation。

这意味着当前瓶颈不再适合通过继续增加小 gate、LoRA rank、层数或短步数来解决。

## 12. 下一步建议

按预先批准的停止规则，下一阶段不再做 v7.2 小补丁。应设计真正的 action-conditioned fine-tuning：

```text
partial Wan attention unfreeze
+ SE(3) geometry branch
+ counterfactual objective
+ larger action-video dataset
```

下一阶段应先决定：

1. 解冻哪些 Wan attention blocks，以及 q/k/v/o 的最小白名单。
2. 如何在 8×24GB 4090（或当前可用拓扑）上满足 <22 GiB/rank。
3. clean-1000 与更大 action-video 数据的 exposure、零泄露与 balanced sampling 合同。
4. counterfactual 与 weighted FM 的梯度预算是否继续按 1:1，或改为分阶段调度。

不建议从本轮 step25 warm-start。新的实验应从同一个 clean-gated parent 建立独立 lineage，避免把失败 branch 的局部 energy bias 带入真正的 backbone fine-tuning。
