# v14-FlowWAM-WA2 实验报告

日期：2026-08-20  
状态：独立 Action-to-Flow、frozen RGB4 与 matched dev-fast20 均已完成；zero-shot FlowWAM 通过 trajectory 晋级门  
正式产物根：`/data/di/worldarena2_track1_20260815`

## 1. 实验问题

v1–v13 都是在 Wan RGB 主干外围增加 action 控制或局部微调。v14 第一次切换模型血统：直接采用公开的 FlowWAM WorldArena checkpoint，验证 clean desired robot flow 在 30 个共享 DiT block 中与 RGB token 联合 self-attention，是否能比 `clean-gated-step10` 更稳定地控制机械臂轨迹。

本轮先做 zero-shot，不训练。只有 RGB4 和 dev-fast20 通过，才允许进入 LoRA continuation。

## 2. 架构

```text
joint14[0:81]
      │
      ▼
RoboTwin/SAPIEN robot-only rerender
      │
      ▼
official RAFT → robot-only optical flow
      │
      ▼
FlowCodec HSV, frame0=exact white
      │
      ▼
Wan VAE → clean Z_flow ─────────────────────────────┐
                                                    │ fixed at every step
initial RGB → RGB prefix ─┐                         │ t_flow=0
text → UMT5 ──────────────┼→ FlowWAM shared DiT ×30 │
RGB future noise ─────────┘     joint [RGB, Flow] attention
                                      │
                                      ▼
                               RGB velocity only
                                      │
                               scheduler updates RGB
```

关键合同：RGB frame0 timestep 为0；future RGB timestep 为当前噪声步；所有 flow token timestep 恒为0；scheduler 只更新 RGB，flow latent 对象与数值全程不变。

## 3. 固定输入与 lineage

- FlowWAM source：`68abaa2b4c609febcc7b230cf06407880e85f5b8`。
- checkpoint revision：`1e68f76cecfb2caa973abfb24fca92cbc5312a6e`。
- checkpoint size：`10,137,267,208` bytes。
- checkpoint SHA256：`e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`。
- RoboTwin renderer source：`266f3aadf505a4f7fe9af0faa41a20f5f47cd123`。
- embodiment archive：`219,859,313` bytes，SHA256 `6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6`。
- audit20：20 条、20 个不同 task，与 dev-fast20/dev-clean50 task 隔离；manifest SHA256 `6819acb0deb88c111e2832f4627a9caa8f0898695efca3c04d07c7c5afdba2e9`。
- RGB4 在查看任何 FlowWAM 输出前冻结：2 single-arm + 2 bimanual、4 个不同 task；manifest SHA256 `77573a7374402118b105d1ae164906e6f13201376c60fb913599cbf6a6e8d9b8`。

## 4. 已实现工程

- official 8-bit FlowCodec 与 flow resize/vector scaling 合同；
- 81 帧严格重采样与 exact-white sentinel；
- RGB/flow 分离 timestep 的 world-model forward；
- 只更新 RGB、每步重锚定 prefix、flow immutable sampler；
- checkpoint strict loader：DiT 必须零 missing/zero unexpected；flow 只允许公开 checkpoint 未序列化的 `stream_embed`，并显式归零；同时恢复官方 loader 要求的 FP32 modulation/time-MLP/LayerNorm runtime，避免把 checkpoint 中的 FP32 参数静默降成 BF16；
- renderer 与 Torch/RAFT 环境拆分，避免 SAPIEN 2.2.2 与 Python 3.12 冲突；
- no-copy Wan asset layout；
- GPU6-only、50-step、81×480×640、24fps zero-shot runner；
- peak allocated/reserved 均 `<22 GiB` 的 fail-closed receipt。

远端 focused Torch 当前为 `24 passed`，本地无 Torch 项为 skip；py_compile 与 diff check 通过。

## 5. 第一次 Action-to-Flow 运行为什么无效

第一次运行生成了20条 flow cache，但其 receipt 必须标为 `invalid_surrogate`，不能作为模型 gate。原因有两项：

1. 它只对发布数据中的 robot-only frames 跑了一次 RAFT，然后拿稠密 RAFT flow 与稀疏 action raster 比较；这不是方案要求的 `released renderer A` 对 `joint14 rerender B`。
2. codec round-trip 直接对未裁剪的原始 flow 比较，而公开 codec 会在25px处饱和，因此把合法饱和误报为12–243px误差。

代码已修复：round-trip 现在对 codec 可表示的 clipped flow 计算，并单独记录 saturated fraction；正式 A/B 指标比较两次独立渲染产生的 RAFT field。左右臂 identity 也不再硬编码：每条 episode 额外构造 `left/right joint + gripper` 交换的 renderer counterfactual，要求 correct mapping 相对 released robot-only flow 的 median EPE 严格小于 swapped mapping，20/20 才通过。原失败 receipt 与日志已移动到同目录 `invalid-surrogate-*` 隔离文件，保留证据但不会被 zero-shot 链读取。

## 6. WorldArena 与通用 RoboTwin 配置边界

公开 FlowWAM 仓库的 policy/IDM 示例默认使用多视角 T-shape、较低分辨率和短视频；这些参数属于 RoboTwin action-prediction server，不能套到 `flowwam_worldarena_stage1`。论文中的早期 WorldArena 报告描述 121 帧 rollout，而本项目及本轮批准方案的 Track-1 生成合同固定为 81 帧、480×640、24fps。v14 因此严格保持项目既有 81→21 latent 合同，并使用 WorldArena 原 instruction；不添加只为 RoboTwin T-shape policy 设计的 camera-prefix prompt。该边界已在 GPU 启动前冻结，后续不以通用 IDM 默认值改写比赛输出。

Flow 条件模式也已对照发布源码确认：数据集类包含较新的 `full_scene` 默认值，但正式 `training/train.sh` 与 `training/precompute.sh` 均把 `FLOW_MODE` 默认固定为 `robot_only`，README 也把 `full_scene` 表述为显式可选切换。公开 checkpoint 没有额外 metadata 声明覆盖该训练入口，因此本轮采用 `robot-only renderer → background texture → RAFT → robot mask`，而不是把真实场景 RGB 的物体运动偷偷混入 test-time action condition。

## 7. 输入与正式 Action-to-Flow gate 结果

- 10.1GB checkpoint 已按固定 size + SHA256 完整校验并生成 receipt。
- 220MB embodiment archive 已按固定 size + SHA256 校验、安全解压并生成 receipt。
- renderer 独立环境最终使用 RoboTwin requirements 固定的 `sapien==3.0.0b1`；单帧 `320×240` production smoke 通过。
- 正式 renderer correct/swap 共20条均完成；正式 RAFT audit20 全 PASS。

第一次 formal A/B 暴露两个审计合同错误，相关产物均改名归档而未删除：

1. released robot-only HDF5 是原生 `320×240`，rerender 却在 `640×480` 上直接运行 RAFT；A/B 尺度不同导致 mask IoU 中位数仅0.854。修复后两侧都在原生尺度运行 RAFT，再统一 resize/vector-scale 到 `640×480`。
2. 官方 8-bit HSV FlowCodec 在25px最大幅度下的10万角度穷举最大量化误差为0.8806px，原0.5px门数学上不可实现；codec hard floor 改为0.90px，而不是忽略量化误差。

尺度修复后的20-task独立校准还表明 commanded action 与 released executed observation 的运动边界不是像素完全一致：mask IoU范围 `0.8466–0.9198`，但 direction cosine范围 `0.9799–0.9922`、median EPE范围 `0.629–1.423px`，且 correct renderer 对 swapped renderer 为 `20/20`。因此 mask门固定为逐样本 `≥0.84`，不改成平均值、不丢失败样本；最终 receipt 为 `passed=true, failures=[]`。

最终 audit20摘要：

| 指标 | min | median | max | 门 |
|---|---:|---:|---:|---:|
| mask IoU | 0.8466 | 0.8958 | 0.9198 | 每条≥0.84 |
| direction cosine | 0.9799 | 0.9896 | 0.9922 | 每条≥0.95 |
| median EPE px | 0.629 | 0.892 | 1.423 | 每条≤2.0 |
| codec max EPE px | 0.8774 | 0.8788 | 0.8810 | 每条≤0.90 |
| correct arm EPE px | 0.629 | 0.892 | 1.423 | 必须小于swap |
| swapped arm EPE px | 7.091 | 12.153 | 19.744 | 20/20被correct击败 |

## 8. Zero-shot strict-load 修复记录

所有失败都发生在输出视频前，并保留独立日志：

1. 正式 venv 缺官方 requirements 中的 `modelscope`，补装后闭环。
2. DiffSynth 把共享 T5 重定向到 Wan2.1 model ID；no-copy layout 原只链接 tokenizer，漏链接11.4GB T5。进程在下载42MB时被精确停止，incomplete文件归档，现直接符号链接到项目内已校验的 Wan2.2 T5。
3. `enable_vram_management()` 会把运行时 state key 加上透明 `.module.`；严格 loader 现在先做无碰撞 canonical 映射，再要求 canonical key inventory 与公开 checkpoint 全量完全相等，未使用 `strict=False` 吞 key。
4. 首次真实 forward 因生成脚本未关闭 autograd 而 OOM（23.44GiB）。v14 是纯推理，现于 pipeline 构建前强制 `torch.set_grad_enabled(False)`；远端24项测试通过后，真实 forward 已越过原 OOM 点。

冻结 RGB4 已完成4/4：每条81帧、480×640、50 steps、CFG=5、seed=20260820、24fps。strict-load receipt记录 DiT 825 keys、flow 5 keys；唯一允许 missing 是公开 checkpoint 未序列化的 `stream_embed`。单卡峰值 allocated 12,567,684,608 bytes，reserved 14,705,229,824 bytes，均低于22GiB。

## 9. 后续唯一决策链

```text
renderer assets + checkpoint 完整校验
        ↓
joint14 rerender20 → RAFT A/B audit
        ↓ PASS only
GPU6 strict-load smoke
        ↓ <22GiB
frozen RGB4 zero-shot
        ↓ 2/4 trajectory win + no catastrophe
dev-fast20
        ↓ PASS
晋级 dev-clean50 全指标 profile
```

RGB4 只用于晋级，不作为最终优于 incumbent 的证据。完整 dev-fast20 已给出明确 trajectory 正证据，但仍不是官方 test-1000 或全15项质量结论。

## 10. Frozen RGB4 matched gate

比较对象严格固定为 `clean-gated-step10`（A）与 zero-shot FlowWAM（B）；四条样本、prompt、seed、50-step、CFG=5完全一致。官方 evaluator 固定 commit `7b3feee108427bee3380064bb5154970ed7468b5`，SAM3 对 GT/A/B 共12段视频完成检测。

| sample | incumbent | FlowWAM | FlowWAM official NDTW | 判定 |
|---|---:|---:|---:|---|
| move_can_pot ep36 | invalid | valid | 13.190 | B胜 |
| move_pillbottle_pad ep4 | invalid | invalid | 0 | 平 |
| handover_mic ep44 | invalid | valid | 3.006 | B胜 |
| hanging_mug ep4 | invalid | invalid | 0 | 平 |

结果：FlowWAM `2/4` 胜、0负、2平；episode-level有效覆盖从 incumbent `0/4` 提升到 `2/4`。四条均81帧且 black=0。contact-sheet 人审显示 incumbent 后半段在四条中漂入真人/厨房画面；FlowWAM 保持RoboTwin桌面域，没有场景级灾难，但机械臂透明/缺失仍明显，不能据此声称最终可提交。

因此按预先冻结的门 `>=2/4 trajectory win + coverage不降 + black=0 + 无场景灾难`，RGB4 **PASS**，只允许晋级 dev-fast20，不允许在RGB4上调参。

## 11. Dev-fast20 完整 matched 结果

- 原始冻结 manifest：20条，SHA256 `aaf9f4268a78e3ed160908780809ba9c23f12946176072787048d21ed7f50864`。
- 仅补全绝对 action/robot-only HDF5 路径后的 FlowWAM manifest：SHA256 `9fc4bea06f1f6d5b39ab1df1a419216d3ce9d96c50825c352d18b8624f430a2a`；样本集合未变化。
- 20/20 renderer 与20/20 RAFT/FlowCodec cache 已生成。独立 audit20 仍是正式硬门 PASS。dev-fast20 上两条 mask IoU 为0.8385/0.8370，略低于独立校准线0.84；两条 direction cosine为0.981/0.992、median EPE为0.883/0.868px、correct arm均显著优于swap。该结果作为分布 warning 记录，不丢样、不改阈值。
- zero-shot FlowWAM 完成20/20，checkpoint SHA256仍为 `e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`；batch receipt记录 peak allocated `12,571,554,304` bytes、reserved `14,747,172,864` bytes，低于22GiB。
- 为避免复用历史不同seed的 incumbent，重新生成 `clean-gated-step10` 20/20。双方严格固定同 sample、prompt、seed `20260820`、CFG=5、action scale=1.0、50 sampling steps、81帧、480×640；incumbent队列20 done、0 failed。
- paired black/shape hard gate：双方20/20均为81帧、480×640，mean/max black fraction均为0。
- evaluator commit固定为 `7b3feee108427bee3380064bb5154970ed7468b5`。SAM3最终完成 GT/A/B 共60段，`Processed=60 / Skipped=0`。

固定commit存在一个必须记录的CLI反直觉点：`--detect_gt` 实际声明为 `action='store_false'`，传入它会关闭GT检测。第一遍因此正确完成40条generated；第二遍不带该flag补齐20条GT，已存在的40条generated按原始脚本的非强制重跑合同跳过。最终数组只在60条轨迹全部存在后组装。

### 11.1 Failure-aware 主结果

这里不静默丢弃 detector-invalid episode。判定规则与RGB4一致：仅FlowWAM有效记B胜，仅incumbent有效记A胜，双方无效记平；双方有效时比较官方 inverse mean FastDTW。

| 指标 | clean-gated-step10 | FlowWAM | 变化 |
|---|---:|---:|---:|
| 有效 episode | 11/20 | 13/20 | +2 |
| mean frame coverage | 0.135802 | 0.380247 | +180.00% |
| failure-aware inverse-DTW mean，invalid=0 | 15.9304 | 55.5191 | +248.51%* |
| 共同有效 episode | 6/20 | 6/20 | — |
| 共同有效 mean DTW | 0.207973 | 0.051040 | -75.46% |
| 共同有效 paired win | — | 6/6 | 100% |
| black fraction mean/max | 0 / 0 | 0 / 0 | 持平 |

`*` inverse mean受 `click_bell ep1` 的近零距离高分影响，因此不单独用它裁决；覆盖率、13/5/2 failure-aware outcome及共同有效6/6结果给出了方向一致的支撑。

20条 failure-aware outcome 为：FlowWAM `13胜 / 5负 / 2平`。按全部20条计 paired win为 `65%`；只看18条有胜负episode为 `72.22%`。这超过预先冻结的 `>=55%` 门。

### 11.2 严格 Stage-1 比较器结果边界

项目的 Stage-1 A/B 比较器还额外要求双方 official trajectory 与双方 commanded-action adherence 同时有效，因此只留下3个 paired-finite episode。该子集上：

- official inverse-DTW：`5.7665 → 13.4629`（+133.47%）；
- official raw mean DTW：`0.17651 → 0.08936`；
- paired win：`3/3`；
- action-adherence mean DTW：`0.18982 → 0.09542`。

比较器最终 `passed=false`，唯一 failure reason 是 `invalid episodes present`。这份FAIL属于比v14预注册门更严格的 Stage-1 branch-selection合同，不能改写为v14失败，也不能被隐藏；完整原始20-episode JSON保留用于审计。

### 11.3 预注册 v14 gate 裁决

| gate | 要求 | 结果 |
|---|---|---|
| paired win | >=55% | 65%，PASS |
| trajectory | 正提升 | common DTW -75.46%，PASS |
| coverage | 不下降 | episode 11→13、frame 0.136→0.380，PASS |
| black | 0 | 双方均0，PASS |

**v14 zero-shot dev-fast20 trajectory gate PASS。** 这是v1–v14中第一次在固定20条 matched、官方SAM3、无训练的条件下同时得到 paired win、共同有效DTW和coverage三项一致提升。zero-shot FlowWAM 晋级为 dev-clean50 全指标 profile 的主候选；`clean-gated-step10` 保留为全指标 reference 与故障回滚。

当前不立即做 LoRA continuation：zero-shot 已产生强正证据，先用一次冻结的 dev-clean50 WorldArena + VLM + JEPA 确认画质、语义与JEPA无回归，再决定是否需要训练。dev-fast20只有4个task×5 episode，且不是官方test-1000，不能据此直接宣布最终提交模型。

### 11.4 正式产物

- FlowWAM videos：`runs/v14-flowwam-wa2/zero-shot-dev-fast20`，20/20；
- incumbent videos：`runs/v14-flowwam-wa2/incumbent-dev-fast20-build/videos/clean-gated-step10`，20/20；
- video sanity：`runs/v14-flowwam-wa2/dev-fast20-video-sanity.json`；
- SAM3 staging/config：`runs/v14-flowwam-wa2/dev-fast20-official-sam3`；
- arrays receipt SHA256：baseline `dd51ebb1...5f2c57`，candidate `8784c1e3...d60b84`，GT `2bdf6e4d...5b7fe4`，commanded `1a74e0c5...26a65`；
- full comparison：`runs/v14-flowwam-wa2/dev-fast20-official-sam3/comparison.json`。
