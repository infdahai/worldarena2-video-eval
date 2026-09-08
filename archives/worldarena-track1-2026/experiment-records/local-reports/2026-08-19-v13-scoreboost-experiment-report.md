# v13-ScoreBoost 实验报告

更新时间：2026-08-20（实验完成，RGB8 gate FAIL）

## 1. 实验目的

v1–Final-NativeBand 已充分说明：在当前比赛窗口内继续追逐小型 action controller 的边际收益过低。v13 不再改 action 接口，而是冻结已验证最好的 `clean-gated-step10` action parent，直接优化总分中更可控的机器人/物体完整性、跨帧物体稳定性和前后关系，同时以 trajectory 与 coverage 为不可退化门禁。

本实验只回答：late-block 低秩微调加训练期视觉监督，能否在不破坏已有 action trajectory 的前提下改善物体消失、变形和穿模。

## 2. 架构

```text
clean latent + text + frozen action raster parent
                        │
                        ▼
                 Wan2.2 TI2V-5B
          blocks 0–17          frozen
          blocks 18–25 Q/V/O   LoRA r16/a16
          blocks 26–29         frozen
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         RGB/flow path      block25 hidden
                                  │
                         training-only depth head
```

可训练项严格只有：

- blocks 18–25 `self_attn.q/v/o` LoRA，LR `5e-6`；
- training-only depth head，LR `1e-4`。

Frozen：Wan 原始权重、`clean-gated-step10` parent、T5、VAE、SAM3、DAV2 teacher。SAM3 与 DAV2 只离线生产标签，不进入 trainer hot path。

## 3. 数据与泄漏合同

- 源池：optimizer2060；
- 训练：150 条唯一 episode，一次 exposure/episode，按 task 均衡；
- RGB gate：4 条 single + 4 条 bimanual/crossing；
- 排除：audit20、dev-fast20、dev-clean50、RGB8、可用时 official test；
- dev-clean50 使用固定 revision `506c4e...e2714`、seed `20260815` 的 5-task/50-episode身份；
- dev-clean50 与 official-test 按 task 隔离，不能只做 episode 隔离。

当前已冻结结果：train150=150，RGB8=8，训练/RGB8 与 dev-clean50 task overlap=0；train150 覆盖 41 个 task，每 task 3–4 条。

## 4. 离线标签

21 个 latent 时间点严格对应 RGB `0,4,...,80`：

- SAM3 robot union mask；
- SAM3 manipulated-object mask；
- DAV2-Small relative inverse depth；
- 输出统一 `21×30×40`，绑定 HDF5、SAM3、DAV2、prompt map 与 payload SHA256。

初次抽检发现 `robot arm/end effector/gripper` 在当前 SAM3 video predictor 中被最后一个空 prompt 覆盖，产生 robot 0/21。该批 staging 已整体隔离。单样本 prompt calibration 结果：`robot=18/21`、`robotic arm=19/21`、`mechanical arm=0/21`、`gripper=0/21`。正式实现改为每 prompt 独立 session 后做 union，首个重跑样本恢复为 robot 19/21、object 21/21。

Robot track 采用 fail-closed：有效帧不足或异常大 mask 进入累计 invalid manifest，selector 排除后确定性补位。实际运行中 `turn_switch`、`open_laptop` 的 robot prompt 跨 episode 持续不可观测，因此整 task 从本轮 supervision pool 排除；最终 158/158 样本通过 robot hard gate，训练集仍为 150 条、RGB8 仍为 8 条，覆盖 39 个 task。

Object track 采用显式 eligibility gate：有效帧不足、面积突变或异常大时，`object_mask=0`、`object_valid=0`，只关闭该样本的 object temporal/object-region 增益；robot-region、FM 和 depth 监督继续生效。最终 70/158 条 object-eligible，88 条保守少标；不会把空 mask 当成“背景真值”，也不会伪造物体标签。这与冻结教师在不满足可观测条件时跳过辅助 loss 的原则一致。

## 5. Loss

```text
L = L_fm + 0.5 L_mask + 0.1 L_temporal + 0.1 L_depth
```

- `L_fm`：有效 future latent 上的 flow matching；
- `L_mask`：`1 + 2*robot + 4*object` 的 per-sample 均值归一化 clean-latent SmoothL1；
- `L_temporal`：相邻有效 object union 区域内的 latent delta SmoothL1；
- `L_depth`：scale/shift-invariant relative inverse-depth SmoothL1。

无 trajectory/counterfactual/phase/JEPA/GAN/Depth inference branch。

## 6. 训练与硬门

- topology：physical GPU6，world size 1；
- GPU0–5、GPU7 不触碰；
- warmup 10；最大 exposure 150；checkpoint 100/150；
- smoke：warmup 后重置 peak，连续 3 次 forward/backward/step/zero-grad；
- allocated 与 reserved 均必须 `<22 GiB`；
- Q/V/O LoRA 与 depth head 梯度必须存在、有限、非零；
- 任意 native Wan/parent 参数可训练或收到非零梯度即失败。

## 7. 评测顺序

1. exposure100 只做训练健康检查；无硬故障继续到150。
2. RGB8 与 parent matched：trajectory 不超过2%退化、coverage不降、black=0，且 object disappearance / penetration-deformation 至少一项改善、另一项不恶化。
3. RGB8 通过才跑 dev-fast20；要求 paired trajectory win ≥55%、trajectory mean>0、coverage不降、black=0。
4. fast20 通过后才顺序扫 action scale `1.0/1.25/1.5`，再扫 text CFG `4/5/6`，禁止笛卡尔积。
5. 最终候选再跑 dev-clean50 WorldArena + VLM + JEPA 完整 profile。

## 8. 实际执行结果

- 本地最新 scoped 合同测试：17 passed；
- 远端最终完整 v13 Torch 回归：33 passed；推理命名修复后的相关回归：9 passed；
- SAM3 mask：158/158 最终样本完成，robot hard invalid=0，object-eligible=70；
- DAV2-Small：固定 revision `5426e4...bef99`，官方 LFS SHA256 `3152477c...6b70`，158/158 深度缓存完成；
- data receipt SHA256：`941bc8ba...04b13`；
- supervision receipt SHA256：`670dbf38...c9567`；
- source closure SHA256：`685a0195...1d6be`；
- production smoke：3/3 完整更新通过，peak allocated `13,539,098,112` bytes（12.61 GiB），peak reserved `13,950,255,104` bytes（12.99 GiB），均低于 22 GiB；step time 约 1.67 秒；
- 正式训练：150/150 exposures 完成，无 OOM/NaN/Inf/梯度缺失；54 个 trainable tensors 与 54 个 optimizer states 完整；
- exposure100 SHA256：`f584754f...752af`；
- exposure150 SHA256：`8bada4ef...9b358`；
- RGB8 matched 原子队列：16/16 完成（8 个 parent + 8 个 v13），0 failure；全部为 81 帧、480×640，black=0。

### 8.1 分段训练统计

| Exposure | Total | FM | Mask | Temporal | Depth | Grad norm |
|---|---:|---:|---:|---:|---:|---:|
| 1–25 | 0.4846 | 0.3878 | 0.0633 | 0.0405 | 0.6102 | 0.4700 |
| 26–50 | 0.4029 | 0.3313 | 0.0548 | 0.0217 | 0.4209 | 0.1491 |
| 51–75 | 0.4901 | 0.4160 | 0.0401 | 0.0235 | 0.5167 | 0.2931 |
| 76–100 | 0.5556 | 0.4867 | 0.0500 | 0.0156 | 0.4226 | 0.2165 |
| 101–125 | 0.4243 | 0.3623 | 0.0427 | 0.0330 | 0.3727 | 0.1296 |
| 126–150 | 0.4088 | 0.3451 | 0.0279 | 0.0137 | 0.4836 | 0.2023 |

这些区间覆盖不同 task，不能把 FM 的非单调变化解释成同分布 learning curve；可确认的是所有 loss 有限、mask loss 后段较前段降低、梯度始终存在。

### 8.2 推理集成故障与闭环

首个 RGB8 job 曾被严格 loader 拒绝：训练在 activation-checkpoint wrapper 上安装 LoRA，checkpoint key 为 `blocks.N.block.self_attn.*`；推理在裸 Wan block 安装，目标 key 为 `blocks.N.self_attn.*`。修复仅规范化 blocks18–25 的这一层 wrapper 名称，不放宽 inventory/shape/lineage 校验。新增真实 wrapped-key RED/GREEN 回归，失败记录保留在 queue logs 后重新入队。

## 9. 待回填结果

| Gate | 结果 |
|---|---|
| mask cache valid/refill | PASS: 158/158 robot hard gate；70 object-eligible |
| depth cache receipt | PASS: 158/158；receipt `670dbf38...c9567` |
| preflight | PASS: 3,152,385 trainable parameters；teacher hot path=false |
| GPU6 smoke peak allocated/reserved | PASS: 12.61/12.99 GiB |
| exposure100 loss/gradient | PASS；checkpoint `f584754f...752af` |
| exposure150 loss/gradient | PASS；checkpoint `8bada4ef...9b358` |
| RGB8 matched generation | PASS：16/16，0 failure，0 black |
| RGB8 matched PSNR/SSIM | 微弱正向：PSNR +0.012 dB（5/8 win）；SSIM +0.002864（6/8 win） |
| detector-free trajectory proxy | INVALID：GT calibration median error 30.34 px > 12 px；仅作风险信号，v13 score -5.60%、paired 4/8 |
| official SAM3 trajectory | FAIL：24/24 detector run 完成；共同有效仅1/8；该样本 v13 DTW 0.2003 vs parent 0.0897 |
| fast20 | NOT RUN：RGB8 未通过 |
| dev-clean50 profile | NOT RUN：RGB8 未通过 |

## 10. 决策纪律

v13 失败时回滚 `clean-gated-step10`。不因失败重新打开 phase、SE(3)、PRA、GRU/TCN、action Jacobian 或小 Adapter 搜索；不从不通过 gate 的 checkpoint warm-start。

## 11. RGB8 完成结果与初步审查

RGB8 原子队列最终状态为 `done=16 / failed=0 / claimed=0`。parent 与 v13 使用完全相同的8个 episode、seed、prompt、action condition、50 sampling steps、CFG=5 和 action scale=1.0，因此视频结果可做逐 episode matched 比较。

### 11.1 基础视频与像素质量

所有16条视频均为81帧、480×640，未出现黑视频或解码失败。相对同一 GT 的 FFmpeg matched 指标如下：

| 指标 | parent mean | v13 mean | v13 delta | paired wins |
|---|---:|---:|---:|---:|
| PSNR | 9.0867 dB | 9.0988 dB | +0.0120 dB | 5/8 |
| SSIM | 0.712925 | 0.715789 | +0.002864 | 6/8 |

这只能说明 v13 没有发生全局像素质量崩坏，并有极弱的正向倾向；增量太小，不能表述为明确画质提升。

### 11.2 人工视觉审查

逐 episode contact sheet 使用 frame 0/20/40/60/80，parent 在上、v13 在下。整体运动和场景结构相近，但存在两项需要纳入硬门的风险：

- `place_object_stand`：v13 后段操作物体可见性较弱，存在物体消失/未稳定留在支架上的风险；
- `place_can_basket`：v13 的篮筐/容器颜色与身份一致性出现明显漂移风险。

因此目前不能宣称 object disappearance 或 deformation 得到改善。

### 11.3 detector-free proxy 为何不用于裁决

旧 proxy 在 GT 上的 tracker calibration coverage=1，但 median pixel error 为 `30.3448 px`，超过 `12 px` 门槛，故 `selection_allowed=false`。它给出的 v13 trajectory score 相对 parent 为 `-5.60%`、paired win `4/8`、DTW `108.87 vs 107.87`，只能视为风险信号，不能替代官方 SAM3。

### 11.4 官方 SAM3 结果

固定 WorldArena evaluator commit `7b3feee108427bee3380064bb5154970ed7468b5` 对8条 GT、8条 parent、8条 v13 共24条视频完成检测：`Processed=24 / Skipped=0`。轨迹数组 shape 为 `8×81×2×2`，所有数组均写入 SHA256 receipt。

| 指标 | parent | v13 | 判定 |
|---|---:|---:|---|
| 有效 episode | 2/8 | 2/8 | 都很低，不能用非配对均值选择 |
| mean frame coverage | 0.08488 | 0.06944 | v13 下降18.18%，FAIL |
| 共同有效 episode | 1/8 | 1/8 | 有效 matched 样本不足 |
| 共同有效样本 DTW | 0.08968 | 0.20027 | v13 增加123.30%，FAIL |
| 共同有效样本 inverse-DTW | 11.1502 | 4.9934 | v13 下降55.22%，FAIL |
| paired win | — | 0/1 | FAIL |
| black/broken | 0/8 | 0/8 | PASS |

parent 与 v13 各自恰好有2个有效 episode，但第二个有效 episode 并不相同：parent 在 `place_bread_skillet` 有效，v13 在 `place_object_stand` 有效。因此把 invalid 记0后得到的非配对 aggregate 会受“检测到了不同 episode”支配，不可解释为 trajectory 提升。唯一共同有效的 `click_alarmclock` 上，v13 同时出现 coverage `0.3086→0.1605` 和 DTW `0.0897→0.2003` 的明确退化。

### 11.5 最终裁决

**v13 RGB8 gate FAIL，停止该候选，不运行 fast20、CFG/action-scale sweep 或 dev-clean50。**

失败原因不是训练崩溃：视频全部可解码、无黑帧、PSNR/SSIM略有上升；失败点是更关键的 trajectory/coverage 未被保护，而且人工审查未证明 object disappearance / deformation 得到改善。按预先冻结的决策纪律，incumbent 保持 `clean-gated-step10`，v13 exposure150 只保留为可复现实验 artifact，不作为发布 checkpoint、后续 parent 或 warm-start。
