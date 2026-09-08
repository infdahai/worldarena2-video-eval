# Wan-Action v7.1 SE(3) Geometry LoRA 机制实验报告

日期：2026-08-18
实验状态：**已按门禁结束于 step25；未运行 step50/100；未解码 RGB**

## 1. 结论

本轮验证了两个不同问题：

1. **工程与容量假设成立。** rank16 geometry-only Q/K/V/O LoRA 可以在单张 RTX 4090 上稳定训练；原 Wan path 与 clean-gated parent 保持冻结；Q/K/V/O LoRA 和 channel gate 均获得非零梯度；3-step production smoke 的峰值 reserved 为 14.13 GiB，低于 22 GiB 硬门。
2. **weighted FM 的监督假设不成立。** 到 step25，correct action 没有稳定优于 reverse/shift/swap。特别是 swap paired win 从 step10 的 8/20 降至 step25 的 7/20，reverse 的 position/velocity improvement 为负。模型在降低极少量 FM loss，但没有把新增 geometry representation 学成正确的 action dependency。

因此：

- v7 gate-only 已正式淘汰；不跑 step50。
- v7.1 geometry LoRA weighted-FM-only 也在 step25 停止；不跑 step50/100。
- 下一主实验应保持同一 v7.1 架构与参数规模，只加入 counterfactual action ranking loss；不增加 block、rank、gripper bias、trajectory loss、Pose FiLM、Depth、JEPA 或 coordination branch。

## 2. 实验目标

本轮只回答一个问题：

> 可学习的 arm-grouped SE(3) geometry Q/K/V/O representation，在只有 weighted flow-matching 监督时，能否产生稳定的 correct-vs-counterfactual action separation？

它不是画质训练、视频生成或 leaderboard 评测。

## 3. 冻结架构合同

### 3.1 冻结部分

- Base：Wan2.2-TI2V-5B。
- Parent：`clean-gated-step10.pt`，SHA256：`105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`。
- 原 Wan Q/K/V/O、QK norm、3D RoPE 与 native attention path：全部冻结。
- 原 clean-gated action raster parent：全部冻结。
- T5/VAE：不进入训练热路径；输入只使用 cached latent、cached text context、action raster/support 与 SE(3) sidecar。

### 3.2 可训练部分

仅 block `8/16/24` 的 geometry branch：

```text
frozen raw Wan Q/K/V
  + geometry-only Q/K/V LoRA rank16
  -> arm-grouped SE(3) attention
  -> frozen Wan O (no duplicated bias)
  + geometry-only O LoRA rank16
  -> zero-init output channel gate
  -> add to frozen native Wan output
```

- 每个 block：Q/K/V/O 四组 rank16 LoRA + 一个 3072-channel gate。
- 总可训练张量：27。
- 总可训练参数：1,188,864。
- Loss：仅 weighted FM。
- LR：`1e-4`。
- 未加入 gripper bias、trajectory loss、counterfactual loss、Pose FiLM、Depth、JEPA 或 coordination branch。

### 3.3 SE(3) 条件

- 沿用 v7 已验证的 81→21 causal packing 与 motion-scale normalization。
- raw Q/K/V 在 Wan 3D RoPE 前分叉；native path 仍按官方顺序执行 RoPE。
- 左右臂固定 head ownership，各 12 heads。
- presence 来自 action/EEF validity，不来自 RGB observability。
- absent arm 的 geometry heads 严格置零；null action 同时移除两组 geometry heads。

## 4. 数据与零泄露合同

- 训练源：pinned clean-1000。
- source manifest SHA256：`fc54f851099ea213431efcb64a2fcbfd5c01335e18404f685768f00ece89b289`。
- cache SHA256：`64ca793f09ca71de9c2bec4710e373d6aa74f45a07b4ca1363cc000c2405bf94`。
- replay SHA256：`98203b3217b17302dd47e286604630b408b0cbc6c5bc5b3372ad525b17b56738`。
- dev-fast20 只作为 leakage receipt exclusion boundary；没有进入训练或本轮 discovery audit。
- official test 仍不可用且未访问。

### 4.1 修正版 audit-20

最终 audit 集合满足：

- 20 条 episode。
- 20 个不同 task。
- left-only observable 7 条、right-only observable 7 条、both observable 6 条。
- 20/20 均有有效 position 与 velocity probe target。
- selection SHA256：`0c476a01987e2b05f3297883669b5152b80e4d7933ce288dbb865a76af90f530`。

覆盖 task：

`click_alarmclock, click_bell, dump_bin_bigbin, handover_mic, lift_pot, hanging_mug, move_can_pot, move_pillbottle_pad, pick_diverse_bottles, move_playingcard_away, move_stapler_pad, pick_dual_bottles, open_laptop, place_a2b_left, place_bread_basket, place_bread_skillet, place_burger_fries, place_can_basket, place_container_plate, place_empty_cup`。

原始选择中的 `move_stapler_pad episode16` 虽有 RGB observability，但没有有效 probe target；修正版确定性替换为 `episode19`。因此最终所有判断均可按 20/20 paired episodes 解释，不存在静默 survivor filtering。

## 5. v7 gate-only 最终 no-train retirement audit

修正版 20/20 结果：

| checkpoint | reverse wins | shift wins | swap wins | 关键方向 | stable separation |
|---|---:|---:|---:|---|---|
| zero-gate | 14/20 | 10/20 | 10/20 | shift/swap 未达到 60% | 否 |
| step10 | 13/20 | 12/20 | 10/20 | 三类 velocity 均未保持正向 | 否 |
| step25 | 14/20 | 10/20 | 9/20 | reverse position/velocity 为负，swap 下降 | 否 |

step25 详细 improvement（counterfactual error - correct error，正数才表示 correct 更好）：

| variant | position improvement | velocity improvement |
|---|---:|---:|
| reverse | -0.000218 | -0.000810 |
| shift | +0.000600 | -0.000469 |
| swap | +0.004649 | -0.000512 |

结论：三个标量 channel gate 没有形成稳定 action semantics，正式淘汰，且 launcher 已移除 gate-only step50。

## 6. v7.1 preflight 与 production smoke

### 6.1 Preflight

- contract：`wan-action-v71-preflight/1`。
- passed：true。
- 27/27 trainable tensors graph-connected、finite。
- channel gate 第一 backward 即非零。
- Q/K/V/O LoRA 首个 backward 为零符合 zero-gate 初始化预期；gate 更新后才产生 LoRA 梯度。
- 原 Wan/parent unexpected gradients：0。
- preflight receipt SHA256：`f92c5c45e66664ec72ef21c838be924b82afa8a34a99df9b20a0dce9e8b8ccf6`。

### 6.2 三步 smoke

- contract：`wan-action-v71-geometry-lora-single-gpu-production-smoke/1`。
- GPU：物理 GPU6，world size 1。
- 3 次完整 `forward -> backward -> optimizer.step -> zero_grad`：通过。
- max allocated：14,443,174,912 bytes，约 13.45 GiB。
- max reserved：15,172,894,720 bytes，约 14.13 GiB。
- step time：2.775 / 2.352 / 2.357 秒。
- channel/Q/K/V/O 五类梯度在三步内均至少一次非零。
- 原 Wan/parent gradient：0。

## 7. v7.1 训练结果

训练按同一 single-GPU replay 执行到 step25。25 optimizer updates 只相当于 clean-1000 的 0.025 nominal epoch；本轮是机制验证，不是数据规模训练。

### 7.1 Correct-vs-counterfactual probe

| checkpoint | reverse | shift | swap | 结论 |
|---|---:|---:|---:|---|
| step10 | 14/20 (70%) | 10/20 (50%) | 8/20 (40%) | 只有 reverse 达标，swap 明显失败 |
| step25 | 13/20 (65%) | 11/20 (55%) | 7/20 (35%) | 没有开始形成稳定三类 separation |

step25 详细 improvement：

| variant | position improvement | velocity improvement |
|---|---:|---:|
| reverse | -0.001099 | -0.002154 |
| shift | +0.001437 | +0.000529 |
| swap | +0.005388 | -0.001539 |

关键观察：

- reverse win rate 尚可，但 correct 的平均 position/velocity 反而更差，不能算有效机制分离。
- shift 只有轻微正向，未达到 60%。
- swap 从 8/20 下降到 7/20，是最明确的失败信号。
- step10→25 没有形成单调改善曲线，因此没有机制依据继续 step50。

### 7.2 Weighted FM

20 条 audit 平均 correct FM：

| checkpoint | mean correct FM | min | max |
|---|---:|---:|---:|
| step10 | 0.3703586 | 0.1464451 | 0.9220972 |
| step25 | 0.3703558 | 0.1464664 | 0.9220974 |

在与 zero-output frozen parent 共有的 19 条 matched episodes 上：

| checkpoint | parent FM | candidate FM | relative regression |
|---|---:|---:|---:|
| step10 | 0.3568895 | 0.3568800 | -0.0027% |
| step25 | 0.3568895 | 0.3568763 | -0.0037% |

FM 没有退化，远低于 2% regression 门；但改善几乎为零，也没有转化成 trajectory/action separation。这正是“weighted FM 对 action dependency 监督太弱”的直接证据。

### 7.3 参数实际更新

所有 block 的 gate 与 Q/K/V/O LoRA 均发生更新。step25 参数 L2：

| block | gate L2 | gate abs max | Q-up L2 | K-up L2 | V-up L2 | O-up L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.04608 | 0.002418 | 0.00577 | 0.00660 | 0.01749 | 0.02076 |
| 16 | 0.05729 | 0.002367 | 0.00630 | 0.01035 | 0.02487 | 0.03149 |
| 24 | 0.05736 | 0.002347 | 0.00433 | 0.00582 | 0.02531 | 0.01992 |

V/O 更新明显大于 Q/K，证明分支不是“没有训练”；失败发生在监督语义，而不是无梯度、参数冻结或容量为零。

## 8. 门禁判定

### step10

- 梯度：通过。
- weighted FM：finite，通过。
- residual/channel gate：bounded，通过。
- frozen Wan/parent whitelist：通过。
- head ownership/absent-arm：operator regression 通过。
- action separation：只作观察，不作为 step10 淘汰门。

### step25

要求开始出现正 separation。实际：

- reverse 13/20，且 position/velocity 为负。
- shift 11/20。
- swap 7/20。

判定：**失败，停止。**

### step50/100

- step50：未运行、无 checkpoint。
- step100：未运行、无 checkpoint。
- routing retention ≥90%、正式 60–70% win gate、4–8 RGB decode：均未进入评估阶段。

## 9. 测试与 provenance

- 远端 focused Torch/contract suite：48 passed。
- remote Git HEAD：`1845adc4919f06fd98790b47d5b7b7476cc4c9c0`。
- source closure SHA256：`371a63917d9e7aeb779111a23e7c626547a10b4c1192d94adbeeac04886104dd`。
- GPU6 结束后：6 MiB、0% utilization。
- GPU0–5 与 GPU7 的同事进程未停止、未重启、未占用。

Checkpoint：

| checkpoint | SHA256 | bytes |
|---|---|---:|
| step10 | `f45febf85a71ced721775b0d2cb2ecb7dffb6ea576b7e9966eea30c1848b5677` | 14,299,781 |
| step25 | `0d3de68e2dedbb43d09bb129a6e7e0ba9d506acac4f50fbb2e6f03b322f479c7` | 14,300,165 |

远端产物目录：

- `/data/di/worldarena2_track1_20260815/runs/v71-se3-geometry-lora-single-gpu/mechanism`
- `/data/di/worldarena2_track1_20260815/runs/v71-se3-geometry-lora-single-gpu/smoke`

说明：v7.1 audit JSON 内层 metrics 复用了 retirement aggregator，因此内层 `contract` 字段仍写作 `wan-action-v7-gate-only-retirement-audit/1`。样本、计算和结果均为 v7.1 checkpoint；这是命名遗留，不影响数值，但下一实验应启用新的 v7.1 counterfactual-audit schema，避免继续复用该标签。

## 10. 下一步唯一建议

启动一个新的、独立 lineage：`v7.1-se3-geometry-lora-cf`。

保持不变：

- 同一 clean-gated-step10 parent。
- 同一 block 8/16/24。
- 同一 rank16 geometry Q/K/V/O LoRA。
- 同一 cached clean-1000、replay、audit-20。
- 原 Wan path 完全冻结。

唯一新增：

```text
L = weighted_FM + lambda_cf * ranking(correct, reverse/shift/swap)
```

第一轮仍只跑 10/25/50；如果 counterfactual loss 仍不能让 correct 稳定优于 swap/shift/reverse，则停止小 Adapter/LoRA 路线，转向部分解冻 Wan attention + 更大 action-video 数据。

不建议从失败的 step25 warm-start 后再宣称“只改变 loss”。为了让因果归因干净，下一轮应从同一个 frozen parent 与相同随机初始化合同重新开始。
