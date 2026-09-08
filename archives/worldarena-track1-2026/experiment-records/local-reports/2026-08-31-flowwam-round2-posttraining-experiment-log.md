# WorldArena 2.0 Track 1 — FlowWAM Round 2 / Post-training 实验总账

> 状态：持续更新（append-only evidence ledger）  
> 首次建立：2026-08-31（CST）  
> 远端：`huazhi@183.147.142.110:9000`  
> Artifact root：`/data/di/worldarena2_track1_20260815`  
> 目标：在 2026-09-04 前，以完整官方 corrected15 为唯一晋级依据，提高 HZ-World Track 1 总分。

## 1. 记录硬合同

从本文件建立起，每个实验必须记录下列证据；缺任一关键证据的实验不得晋级：

1. 实验 ID / work key、目标假设、所属轮次和状态。
2. Official parent、代码、配置、数据 split、checkpoint 和输入/输出 SHA256。
3. GPU physical index + UUID、launcher/compute PID、PPID/PGID、开始/结束时间。
4. 训练曲线、显存、异常、重试及隔离/恢复方式。
5. 视频结构收据：episode、121 帧、640x480、black0、视频 SHA。
6. frozen generated9 selector 的选择结果；不得用 full15 或 GT 倒推 seed。
7. selected candidate 的完整 15 项、nonmotion12、corrected15/EWMScore_P、wins/CI（适用时）。
8. 结论只能是 `PROMOTE`、`CLOSE`、`DIAGNOSTIC_ONLY` 或 `UNKNOWN`，并写明下一步。

本总账记录实验事实和阶段结论；各实验目录中的 JSON receipt 是机器可验证的最终证据。后续阶段跃迁必须同时更新本文件和对应 receipt。

## 2. 冻结评测口径

- P0 dev4 corrected15 基线：`0.6897477279`。
- 候选生成：每个 episode 仅比较 `seed1` / `seed4`。
- 选择顺序：先读取 generated-only 9 项并用冻结 selector 选择；选择完成后才读取 selected full15。
- corrected15：`Dynamic Degree`、`Flow Score`、`Motion Smoothness` 逐 episode 取 `min(candidate, matched-GT)`；其余 12 项保留 candidate 原值。
- 主目标：`corrected15 = mean(15 metrics)`；报告时 `EWMScore_P = 100 * corrected15`。
- 禁止：full15 oracle 选 seed、Official test1000 调参、失败 arm 追加 seed/steps、旧 Wan 链路混入。

## 3. 当前证据边界

### 3.1 已知基线

| 基线 | 样本 | corrected15 | 说明 |
|---|---:|---:|---|
| P0 selector9 fresh40 | 40 selected / 80 raw | `0.671070` | seed1=20、seed4=20；fixed seed1=`0.670021`，fixed seed4=`0.665498`，full15 paired oracle=`0.677411` |
| P0 dev4 | 4 selected / 8 raw | `0.6897477279` | Round 2 n8 晋级的直接配对基线 |
| Official FlowWAM seed1 clean50 | 50 | `0.654107` | expanded dev 诊断，不是最终无偏 holdout；nonmotion12=`0.707392` |

clean50 最低项：Photometric=`0.162124`、Aesthetic=`0.387585`、Image=`0.521135`、Trajectory=`0.629107`、Interaction=`0.664`、Instruction=`0.692`。Motion Smoothness GT-cap 命中率 84%，剩余 headroom 仅 `0.001287`，因此不再作为主攻项。

### 3.2 当前主要归因

- 视觉保真是第一瓶颈：Photometric、Aesthetic、Image；高动作任务的 Photometric 尤其差。
- 动作/语义条件仍有缺口：Instruction、Interaction、Trajectory，但不能以牺牲 JEPA/Image 为代价。
- 普通 FM / q-v-o LoRA 在现有数据上稳定偏向 motion，常损伤视觉保真。
- 因此 Round 2 不再重复普通 LoRA，而验证三类有明确归因的新分支：LRAM、Late ORB-v2、ORC。

## 4. Round 1 — 已关闭实验

| 实验 | 结果 | 关键指标变化 | 结论 |
|---|---:|---|---|
| B-c050 | corrected15=`0.6811378849`，Δ=`-0.0086098430`，wins=0/4 | JEPA `-0.114985`，Trajectory `-0.051868` | `CLOSE` |
| E-mask-d1 | corrected15=`0.6872641946`，Δ=`-0.0024835333`，wins=1/4 | Trajectory `+0.04381`，Photometric `+0.00524`；JEPA `-0.09155`，Image `-0.00682` | `CLOSE` |
| D-p025 | generated9 | Aesthetic `-0.006754`，Image `-0.004034`，Photometric `-0.004935`；仅 Dynamic/Flow 小幅上升 | `CLOSE` |
| D-p050 | generated9 | Aesthetic `-0.005798`，Image `-0.004051`，Photometric `-0.007444`；motion 三项小幅上升 | `CLOSE` |
| BD | corrected15=`0.6847555393`，Δ=`-0.00499219`，wins=2/4 | Semantic `+0.06433`；JEPA `-0.08124`，Trajectory `-0.03360`，Photometric `-0.01285` | `DIAGNOSTIC_ONLY` |
| B-control plain-FM 25-step | generated9 | Photometric `-0.016015`，Aesthetic `-0.003938`，Image `-0.003197`；Dynamic/Motion 上升 | `CLOSE` |
| F | 未继续 | 机制与已有失败路径重复 | `CLOSE` |

Round 1 结论：现有训练数据/普通 FM 目标更容易换取 motion，而不是提高官方总分所需的视觉质量；不得重试普通 q/v/o LoRA、contact-FM、scalar preserve、早期 ORB、B/D/E/BD。

## 5. Round 2 — 当前实验矩阵

### 5.1 A：LRAM（Latent Reference Appearance Memory）

实验根：`/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/round2/lram-v1`

假设：在 RGB-only 的 late blocks 注入 frame0 外观记忆，并保持 action/dynamic residual 严格为零，可提高静态外观与时序视觉一致性，避免普通 FM 对动作分布的偏置。

机制：blocks 29 或 24+29；RGB-only；frame0 memory after FFN；static x0 + temporal delta supervision；Official backbone 冻结。

| Arm | 配置 | step25 checkpoint SHA256 | step25 loss / fm | 训练状态 | n8 视频 | 评分状态 |
|---|---|---|---:|---|---|---|
| A1 | block29 | `18fd95e3c1d4a7e67a279af46024391978f9628926c09d087fd24b2ddeced338` | `0.26123 / 0.24564` | 完成 | seed1/4 共8条，结构全通过 | `CLOSE_AT_CHEAP_GATE` |
| A2 | blocks24+29 | `a39fa4896013142027748463c5d8386e94405e92625f50fbbef2172e141ae1ad` | `0.25078 / 0.23445` | 完成 | seed1/4 共8条，结构全通过 | `CLOSE_AT_CHEAP_GATE` |
| A3 | blocks24+29 variant | `ec01f42a9b15011734560f82009ea57e8d5e895070372ca9b178291f92ac3034` | `0.27549 / 0.24056` | 完成 | seed1/4 共8条，结构全通过 | `CLOSE_AT_CHEAP_GATE` |

部署证据：module SHA prefix=`f43aac5c`；trainer=`65d1e290`；launcher=`bb35e4ec`；tests=`6efe0407`。配置 SHA prefixes：A1=`7f1d05c0`、A2=`87e8a243`、A3=`7bd05df2`。Focused tests 11/11 PASS；推理 tests 16/16 PASS。推理 runner SHA prefix=`b75eea9d67fd`，stage1 SHA prefix=`f35f3410751a`。

n8 视频收据 SHA prefixes：A1 seed1=`1754940659`、seed4=`09ebfd93c2`；A2 seed1=`1379782435`、seed4=`0ff756160e`；A3 seed1=`dc23ae738f`、seed4=`e752ab376d`。共 24 个视频，均 121 帧、640x480、black0=0，文件 SHA 与 receipt 一致。

终态：三臂视觉 target cluster 均为负，全部在 cheap gate 关闭。A1 target cluster=`-0.0039369`（Image=`-0.0055987`、Aesthetic=`-0.0069789`、Photometric=`+0.0007669`）；A2=`-0.0014455`（Image=`-0.0030181`、Aesthetic=`+0.0003846`、Photometric=`-0.0017031`）；A3=`-0.0022179`（Image=`+0.0040301`、Aesthetic=`-0.0061150`、Photometric=`-0.0045688`）。close receipt SHA prefixes：A1=`cac6d146`、A2=`9968f900`、A3=`0991ca50`。均不花 full15、不进入 200-step。

### 5.2 B：Late ORB-v2

实验根：`/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/round2/late-orb-v2`

假设：把 ORB 限制在 block29 或 blocks24+29，并去掉 preserve/contact 辅助项，只保留 pure FM，可减少早期 ORB 对全局视觉表征的破坏。

| Arm | step25 checkpoint | n8 | generated9 | full15 | 状态 |
|---|---|---|---|---|---|
| B1 | SHA prefix=`b5ac7008` | 完成，结构通过 | visual3 Δ=`-0.0001570`，FAIL | 未花费 | `CLOSE_AT_CHEAP_GATE` |
| B2 | `7bf5e7cc1e664ca002377542f95df17b31f3fe7ab33dad87f771036b9d3df46d` | 完成，结构通过 | 通过 cheap gate | corrected15=`0.6847657795`，Δ=`-0.0049819484`，wins=2/4 | `CLOSE` |

B2 frozen selector receipt SHA prefix=`8241be8e`；选择 `[seed4, seed1, seed4, seed4]`。相对 P0 raw9：Instruction=`0`、Interaction=`0`、Perspective=`0`、Image=`-0.001483`、Aesthetic=`+0.000944`、Photometric=`+0.005401`、Dynamic=`+0.000168`、Flow=`-0.000635`、Motion Smoothness=`+0.005023`；视觉三项均值 `+0.001621`，9项均值 `+0.001047`，无单项低于 `-0.02`。

异常与恢复：B2 首次 base 因 stale summary 缺 `click_bell` GT trajectory (`traj.npy`) 失败；所有派生 partial 已隔离，模型、视频和 selector 结果未改。base 在 GPU3 用新的 launcher/compute 恢复；JEPA 在 GPU4 完成并释放。

B2 selected full15 已闭环：corrected15=`0.6847657795`，相对 P0 Δ=`-0.0049819484`；nonmotion12=`0.7635289365`，Δ=`-0.0060362173`；wins=2/4。主要逐项变化：JEPA=`-0.0844820`、Depth=`-0.0066110`、Dynamic(corrected)=`-0.0026487`、Trajectory=`-0.0020500`、Image=`-0.0014826`；Photometric=`+0.0054009`、Semantic=`+0.0161133`、Aesthetic=`+0.0009444`。B2 失败共同 corrected15 门及视觉保护门，禁止进入 200-step。

评分收据 SHA256：base=`8b63da15797b59b5deb5f64e5145e335455b92111b6754d00a84525c2ee06bdc`；VLM=`f362b2558cba30a3641b5bbbb8be019669440e15e680fe5580c824ac8247887d`；JEPA=`c10b60565427887fe6bab220513ee71cc63035ded4b841add84751d683500c98`；aggregate=`f2f18e67ec03ccd1944ad4514e4844f9da795adb54772c70c4c4f74281cef32c`；corrected15=`324220983dc60a3332887a84a08b13d9eb7f22529bdf0f5ebd7fd19fc28d9fc8`；corrected CSV=`e1057d073a5bb1c79d08362f3593d46e1057030817190e31b46bbef1433ddb95`。

终态 `CLOSE` receipt：`late-orb-v2/b2-block24-29-s31-r1/terminal-decision.closed.complete.json`，SHA256=`2d8876a771abe3126f7c16a36b986fc0e6d856cc06aa42100631879cf58f0d80`。

推理部署：`late-orb-v2/deploy/late-infer-20260831-v1`；runner SHA prefix=`a777d58d`；52 tests PASS。

B1 frozen selector 选择 seed1=1、seed4=3；visual3 Δ=`-0.000157025`，overall9 Δ=`+0.000990980`，Aesthetic 最低 Δ=`-0.002787642`。逐项：Instruction=`0`、Interaction=`0`、Perspectivity=`0`、Image=`-0.001460511`、Aesthetic=`-0.002787642`、Photometric=`+0.003777077`、Dynamic=`-0.002486326`、Flow=`-0.000766397`、Motion Smoothness=`+0.012642621`。由于视觉目标簇未为正，cheap gate 失败，不进入 full15。selection receipt SHA=`33f319a86976eed303851aa560d0884dd41a7b73aa629012dddd0786c2a19ef9`；gate SHA=`85aa3986d2927c22ff1a37decb1203dc8c8fa6dcb9ce7d728a9be0a5267eb219`；terminal close SHA=`671d7c7c0c6c00ea07c1c9d05e6e7c1f8f40ec3af6c05fb3b8aa7305ecf8416a`；远端 Markdown SHA=`8bdcca2b6b694e29bc6c64a95205d0dcd194e2fbe67c6d51bd10b9b12d459c91`。

### 5.3 C：ORC（Object-Response Contrastive）

实验根：`/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/round2/orc-v1`

假设：在 block29 仅对 contact/object-response 区域做 task-disjoint wrong-prompt 二元 InfoNCE，使 late representation 更区分正确指令与错误指令，提高 Instruction / Interaction，同时不直接修改 Official 主干。

机制：block29-only；contact mask=`5x15x20`；task-disjoint wrong prompt；binary InfoNCE；Official parent/native frozen；strict checkpoint load。

| Arm | 权重 | step25 checkpoint SHA256 | 训练观察 | n8 状态 | 评分状态 |
|---|---:|---|---|---|---|
| C1 | 0.05 | `994b707c3dfd4915dec453c1ba1eb99e5498fcd38fe876d07cf3110c682555bb` | final NCE=`0.1675`，last5 mean=`1.2877`，振荡明显 | 完成 | `CLOSE_AT_CHEAP_GATE` |
| C2 | 0.10 | `7957f27c8bb6777a48ddfbbfb3b9163c86698387cfe6645fe40de866f3dd3c3d` | final NCE=`1.0820`，last5 mean=`0.6880`，振荡 | 完成 | `CLOSE_AT_CHEAP_GATE` |

训练 receipt SHA=`c33938d243f31f176165e2206d5efb51889c437839097e8744c14451c24802ff`。3-step smoke 两臂均 PASS，InfoNCE 从约 `0.6931` 降至 `0.3778`，峰值显存约 11.75 GiB。Focused tests 110/110 PASS；inference tests 86 PASS。

推理部署：`orc-v1/deploy/orc-infer-hotload-v1`。初版 runner SHA prefix=`d4a7df`；修复动态 T31 mask 后 runner SHA=`a0baf0d0ca7d9151d0f089914f87f2b85d21b2cd6f39c876558148b019097c`；ORC module SHA prefix=`6566f85`。

当前结论：C1/C2 均为 `CLOSE_AT_CHEAP_GATE`。训练 NCE 振荡不足以转化成 generated9 的 Instruction/Interaction 提升。

C1 frozen selector 选择 seed1=1、seed4=3。相对 P0 raw9：Instruction=`0`、Interaction=`0`、Image=`-0.00071`、Aesthetic=`-0.00394`、Photometric=`+0.00501`、Perspectivity=`0`、Dynamic=`+0.00252`、Flow=`-0.00093`、Motion Smoothness=`+0.00799`。由于主攻 Instruction/Interaction 完全未提升，C1 在 cheap gate 关闭，不花 full15，也不进入 200-step；C2 接续验证。

C2 frozen selector 选择 seed1=2、seed4=2。相对 P0 raw9：Instruction=`0`、Interaction=`0`、Image=`-0.00050`、Aesthetic=`-0.00295`、Photometric=`-0.00954`、Perspectivity=`0`、Dynamic=`-0.00095`、Flow=`-0.00065`、Motion Smoothness=`+0.00379`。同样无目标收益，C2 关闭。C1/C2 gate receipt SHA：`071286dbc9e54721cf2c847c0633f7ecec2afe21d8ac91a4ac53701ac26f1d10` / `3d75dd58d9529b35eb78f9f4e5fad5c3cb29ea869751d231ee68514c27dfd45f`。

## 6. 当前资源与运行态

最近一次已知分配：

| GPU | 工作 | 状态 |
|---:|---|---|
| 0–2 | LRAM A1/A2/A3 | 全部关闭并释放 |
| 3–4 | Late B1/B2 | 全部关闭并释放 |
| 5–6 | ORC C1/C2 | 全部关闭并释放 |
| 7 | 外部 `/data/fjy/worldarena` | 禁止触碰、抢占或发信号 |

所有 GPU 状态必须在每次启动前重新按 physical index + UUID + `/proc` 核验；本表只记录阶段证据，不替代实时资源检查。

## 7. 晋级与后训练计划

用户已批准对真正通过 full15 的赢家开展 stagewise post-training；不对失败 arm 延长训练或增加 seed。

1. Round 2 先完成 25-step + paired n8。
2. 每类机制只保留 generated9 通过且 selected corrected15 确认提升的最佳 arm。
3. 赢家进入 200-step post-training，并评 n24。
4. 只有 n24 corrected15、保护项和趋势持续通过者，才进入 500-step / n64 task-disjoint。
5. 只有持续赢家才进入 1000-step、n128 / randomized task-disjoint holdout。
6. Official backbone 保持冻结；仅训练新增 branch。任何扩大训练都必须保留 15 项门控，防止用 motion 换视觉退化。
7. 最终锁模后才生成 test1000、ZIP 和邮件；未经用户确认不得打包上传或发送邮件。

## 8. 待完成清单

- [x] A1/A2/A3 generated9 与 raw9 gate：三臂均关闭，无 full15 通过臂。
- [x] B2 base retry 完成，聚合 VLM/JEPA/matched-GT，corrected15=`0.6847657795`，已 `CLOSE`。
- [x] B1 generated9：visual3 Δ<0，cheap gate 关闭。
- [x] C1/C2 paired n8 完成并严格核验。
- [x] C1/C2 frozen generated9：Instruction/Interaction 均无提升，两臂关闭。
- [x] Round2 每类 close/promote 决策：全部 `CLOSE`，无 200-step 赢家。
- [ ] 通过 arm 的 200-step post-training + n24。
- [ ] 后续 500/1000-step 与 task-disjoint holdout（仅持续赢家）。

## 9. 变更日志

### 2026-08-31 — 建立总账

- 汇总 Round 1 已关闭结论、Round 2 三类机制的代码/训练/checkpoint/视频/评分证据。
- 将“实验文档 + receipt”设为晋级硬合同。
- 明确未知项：当前尚无 Round 2 arm 完成 selected corrected15，不得误报已提分。
- 后续每次实际启动、阶段完成、错误恢复、close/promote 或 post-training 扩量都在此节追加记录。

### 2026-08-31 10:40 CST — LRAM paired n8 完成，generated9 ready

- work key：`round2/lram-v1/{A1,A2,A3}/n8-dev4/{seed1,seed4}`。
- parent checkpoint SHA：`e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`。
- inference runner SHA：`b75eea9d67fda56b5e204d78a86777836fbdd850463fc59ebf8a8bc179d5c855`；stage1 runner SHA：`f35f3410751ac2383a3588143f61b9c7d2ac3cc31917ed9cd7a0151df88776f0`；tests SHA：`abe7b59bd17ece8491b858f4290e5bbcba171bae7907181fe60077945a0e8c76`。
- GPU：A1=GPU0 `GPU-66b5e...`，A2=GPU1 `GPU-a3ba...`，A3=GPU2 `GPU-af789...`；对应 launcher/compute PIDs 已退出。
- 结构：24/24 PASS；均 121 decoded frames、640x480、black0=0，实际 SHA 与 receipt 一致。
- 异常恢复：首次 shell launch 因 `$out` 展开导致 no-op；修复 quoting 后重启。失败尝试未产生进程、GPU 或数据变更，不构成重复 work。
- 结论：`UNKNOWN`；下一步是 A1/A2/A3 分别完成 seed1→seed4 base→VLM→merge→frozen selector。

### 2026-08-31 10:40 CST — ORC paired n8 接近完成

- C1/C2 seed1 均为 4/4 receipt 完成；seed4 均为 3/4。
- GPU5 compute PID=`3584437`、GPU6 compute PID=`3584502` 正在生成最后一条，util=100%；未重复启动。
- generated9 chain 正在冻结复用精确的 121→81 staging / score-chain 命令，避免和 Late B2 的评分口径漂移。
- 结论：`UNKNOWN`；待 8/8 视频结构核验后进入 frozen generated9。

### 2026-08-31 10:40 CST — Late ORB-v2 B2 full15 阶段跃迁

- B2 JEPA 已完成，GPU4 已释放。
- GPU3 base retry 仍运行；base receipt 出现后立即在 CPU 汇总 base + VLM + JEPA + matched-GT cap。
- 结论：`UNKNOWN`；完整 corrected15 尚未产生。

### 2026-08-31 10:48 CST — LRAM A1/A2/A3 generated9 并行启动

- work key：`round2/lram-v1/{A1,A2,A3}/n8-dev4/generated9`。
- scorer SHA=`eee3559ab65234917daca7c0a0c4826c611ed58aeb0e8fa6fa3022234d66e585`；chain SHA=`96c5e39fb88949d3fe653bcfde9be8c5efbcbddff4923798f577979b42ae43d6`；remap builder SHA=`4f6c0ca264b86a529e00f26d4b1ad91c63198a2d5a00a18505a6c3f84b29080c`。
- frozen selector model SHA=`4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa`；policy SHA=`aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`。
- GPU/PID：A1 GPU0 PGID=`3617414` child=`3620394`；A2 GPU1 PGID=`3617422` child=`3620458`；A3 GPU2 PGID=`3617430` child=`3620522`；base phase 约 4724 MiB。
- 恢复：初次使用 raw dataset path，在 GPU 工作开始前因缺 episode0 remap fail-closed；改为冻结 D-p025 dev4 scoring dataset。发现与已有 C1 scoring 重复后，仅精确终止后启动的重复 PGID `3617449/3617454`，保留原合法 C1 进程。
- 结论：`UNKNOWN`；base/VLM/merge 完成后冻结 selector。

### 2026-08-31 10:51 CST — ORC C1 generated9 并行启动

- work key：`round2/orc-v1/c1-s31`；InfoNCE dose=`0.05`。
- scorer SHA=`eee3559ab65234917daca7c0a0c4826c611ed58aeb0e8fa6fa3022234d66e585`；checkpoint SHA=`994b707c3dfd4915dec453c1ba1eb99e5498fcd38fe876d07cf3110c682555bb`。
- GPU/PID：seed1 GPU5 UUID=`GPU-1770b48c-3b2b-6d7a-30c7-7a9b8b4e53a4`，PGID=`3615537`，compute=`3616069`；seed4 GPU6 UUID=`GPU-b6761c44-aff9-92ed-40e9-fa5731af5cbf`，PGID=`3615612`，compute=`3616133`。
- generation receipt SHA：seed1=`50d9773e4b7b917016670d6e638a2ca27ee26ca887ad0e5b175b2d954907039f`；seed4=`84eccfcb33b18a7716d286ce6881dda609661eeea30c3d9c587055544201a448`。
- remap receipt SHA：seed1=`1a1cbd8888040c3af11708931f7989d304e23ca428e620198a891b6a58d6f796`；seed4=`2899abb96220cc5de29df3a47a6caa881f267839d0f2aa0d1805c75ddbf3efa6`。
- package receipt SHA：seed1=`00d3aa43f7e795149fa9935817ed35e67a5bb56c4a1761a6c224ec30cfeba3e4`；seed4=`98768b55ea6f8ceb3e6b0465211f31ce6f451dc827d7b27ee341b65173d5dadd`。
- 恢复：原收据零补齐命名和错误 dataset path 两次 fail-closed，均未启动 GPU、未留 partial；改为精确复用 Late B2 remap/dev4 dataset 后闭合。
- 结论：`UNKNOWN`；raw9 仅作 cheap gate。最终仍要求 full15 mean(Instruction,Interaction) 至少 `+0.02` 且 corrected15/保护项共同过门。

### 2026-08-31 10:42 CST — Late ORB-v2 B2 full15 完成并关闭

- corrected15=`0.6847657795`，P0=`0.6897477279`，Δ=`-0.0049819484`；nonmotion12 Δ=`-0.0060362173`；wins=2/4。
- 尽管 generated9 cheap gate 为正，full15 暴露 JEPA=`-0.0844820` 的主要退化，证明 cheap gate 不能替代完整15项。
- 结论：`CLOSE`；B2 不进入 200-step post-training。
- 终态机器收据 SHA256=`2d8876a771abe3126f7c16a36b986fc0e6d856cc06aa42100631879cf58f0d80`；GPU3/4 已释放。

### 2026-08-31 10:57–11:07 CST — generated9 七卡异步推进

- LRAM：A1/A2/A3 的 seed1 generated9 均已完成 base+VLM+merge，4/4 VLM rows 完整；同一 driver 已自动进入 seed4，无 barrier 或重复启动。A1/A2/A3 分别使用 GPU0/1/2，PGID=`3617414/3617422/3617430`。
- LRAM cheap gate：`write_generated9_cheap_gate.py` SHA=`07b32dcd...`，仅读取 frozen generated9；要求 mean(Image,Aesthetic,Photometric) Δ>0，Instruction/Interaction/Perspectivity 各≥-0.05，且任何 raw9 Δ不得<-0.10。corrected15/GT 未参与。
- ORC C1：seed1/4 base 已完成并进入 VLM；GPU5/6 PGID=`3615537/3615612`。raw9 gate SHA=`1b873c859b6d711610d8689d17d2de0958cf7143ebe51cb76631d9b6ab0a9cc9`，明确标记为 `cheap-only`；C2 已 staged，C1 释放后原子接续。
- Late B1：完成 receipt-bound 121→81 staging；seed1 GPU3 PGID=`3649049`、compute=`3649574`，seed4 GPU4 PGID=`3649051`、compute=`3649638`；package receipt SHA prefixes=`84582d47/c9f2ad1c`，wrapper SHA prefix=`0f98c412`。首次 ffmpeg/lib 路径与嵌套引号错误均在 GPU 启动前 fail-closed；改用已验证 ffmpeg 和 hash-pinned wrapper 后闭合。
- 11:07 实时核验：GPU0–6 均有本项目 compute PID，显存约 4.7 GiB、util 79–98%；GPU7 仍为外部 PID=`2659785`，未触碰。
- 当前没有 arm 产生新 selection/gate 终态；结论均为 `UNKNOWN`。

### 2026-08-31 11:06 CST — ORC C1 cheap gate 关闭，C2 接续

- C1 selection：seed1=1、seed4=3；selection receipt SHA prefix=`e71e2a58`，gate receipt SHA prefix=`071286db`；seed1/4 CSV SHA prefixes=`b09171ca/4224b017`。
- 主攻项 Instruction=`0`、Interaction=`0`，未产生任何提升；视觉保护虽通过，但不值得继续花 full15。
- 结论：`CLOSE_AT_CHEAP_GATE`；C1 禁止 full15 和 200-step。
- C2 已立即接续：seed1 GPU5 PID/PGID=`3648181/3648112`，seed4 GPU6 PID/PGID=`3653063/3653057`，运行 generated9。

### 2026-08-31 11:15–11:17 CST — 七卡进入 seed4 VLM

- LRAM A1/A2/A3：seed4 base receipts 均终态完成，同一 PGID 已自动进入 seed4 VLM；seed1 CSV 保持终态，selector/gate watcher 健康。
- Late B1 与 ORC C2 同期运行 VLM。11:17 实时核验 GPU0–6 均有本项目 compute PID，单卡显存约 19.4 GiB；GPU7 仍为外部 PID=`2659785`，未触碰。
- 尚无新的 selection/gate receipt；A1/A2/A3、B1、C2 结论仍为 `UNKNOWN`。

### 2026-08-31 11:24–11:25 CST — Round2 全部终态关闭

- LRAM A1/A2/A3：视觉 target cluster 分别 `-0.0039369/-0.0014455/-0.0022179`，全部 `CLOSE_AT_CHEAP_GATE`。
- Late B1：visual3 Δ=`-0.000157025`，`CLOSE_AT_CHEAP_GATE`；Late B2 已在 full15 关闭。
- ORC C2：Instruction/Interaction 均 `0`，`CLOSE_AT_CHEAP_GATE`；ORC C1 同结论。
- Round2 七个 arm（A1/A2/A3/B1/B2/C1/C2）均已形成终态收据；没有可进入 200-step post-training 的 arm。
- GPU0–6 已释放；GPU7 仍为外部任务，未触碰。

### 2026-08-31 11:55–12:00 CST — Round3 获批并开始并行准备

- 用户明确批准 Round3 全部方案；自动化已解除设计审批等待，禁止继续空等。
- 并行路径：Object-Response Flow Completion、GT-anchored preference post-training、原生 640 full-video 视觉蒸馏数据审计。
- 数据根因已确认：现有 `clean1785/dataset/640` 的逐 episode HDF5 指向 `FlowWAM_RoboTwin_extracted`，`observation/head_camera/rgb` 实际 JPEG 解码为 320×240，再在内存 bicubic 到 640×480；因此不得把它作为原生 640 高频监督证据。
- GT-RPO 首批 8 个非 dev/final-holdout 训练样本已冻结：本地 manifest SHA256=`15d962245daf2e723421f6190cb5ae992c34ba8614a4cfed2eaf008f1e6dc4d3`，远端匿名回读同 SHA；每条均绑定 released GT HDF5、robot-only HDF5、固定 seed1 P0 negative video 及文件 SHA。
- 12:00 前实时核验 GPU0–6 无 compute PID；GPU7 仍为外部 `/data/fjy/worldarena` PID `2659785`，未触碰。
- 当前结论：`ROUND3_PREPARING`。任一路最小实现通过 focused tests 且输入合同齐全后，在首个合法 UUID 立即启动 10-step smoke，不等待其他路径。

### 2026-08-31 12:03–12:15 CST — Round3 最小实现与原生 640 数据裁决

- ORFC 最小机制及输入构建器已通过主控复跑：4/4 focused tests PASS、`py_compile` PASS；零初始化与 parent robot-flow 等价，并可从 scene/robot HDF5 经双向 RAFT 构造 residual/contact/occlusion/persistent-state 缓存。真实 symlink 输入边界正在补最后一项合法-root测试，未冒充 GPU-ready。
- GT-RPO 最小 loss/API 已通过主控复跑：8/8 focused tests PASS；共享 noise/timestep、GT preference、parent distillation、flow freeze 与 native late allowlist 均闭环。真实 FlowWAM dataset/model adapter 正在补齐，未冒充已训练模型。
- 原生 640 审计结论为 `NO_GO_CURRENT_DATA`：现有 RoboTwin/clean1785/randomized500 JPEG 均实际解码 320×240；`FlowWAM_WorldArena-v17` 尚不存在。
- 已启动官方固定 revision `a3c1c335f90958258a68e26f674e90ade24ce0ae` 的 `beat_block_hammer` 320/640 双包下载，PID=`3718921/3718922`，经 Clash `127.0.0.1:7890`；该任务不与 dev4/final holdout 重合。下载完成后必须验证 ZIP、安全解压、scene/robot-only配对及全帧真实640尺寸，才能启动视觉蒸馏。
- 12:11 实时二轮空闲核验：GPU0–6 无 compute PID；GPU7 外部 PID `2659785`，未触碰。GPU 启动条件已满足，哪个真实入口先 ready 即先上卡。

### 2026-08-31 12:16–12:19 CST — GT-RPO 四路真实 FlowWAM smoke 启动

- ORFC 输入边界补齐：允许位于 `artifact_root` 内的逐 episode symlink，但 resolved target 必须仍在同一根内且为 regular file；focused tests `6/6 PASS`。
- GT-RPO 真实入口完成：非 LoRA 加载 Official P0，released GT 为 positive、同条件固定 seed1 P0 rollout 为 negative，共享 noise/timestep，loss 排除 latent frame0；仅 RGB head + block29 或 blocks28–29 可训练，flow 和其余 parent fail-closed。主控复跑 GT-RPO + adjacent preference 共 `21/21 PASS`，`py_compile PASS`。
- 部署哈希：远端 `gt_rpo.py=cf01be798ae818549f0cec783decc4d2550de2d695bef088ffd4c539602b1879`；`train_flowwam_gt_rpo_real.py=c2c66e0d3db514d4c1cb8f74add989c3642e07bb49bd90eaedac44ba99c34f83`；train8 manifest 仍为 `15d962245daf2e723421f6190cb5ae992c34ba8614a4cfed2eaf008f1e6dc4d3`。
- GPU3–6 在本轮按 UUID/compute PID 精确核验为空后启动四个隔离 1-step 真机 smoke；launcher/PGID=`3726754/3726755/3726756/3726757`。配置分别为 block29/LR1e-7/beta1、blocks28–29/LR1e-7/beta1、block29/LR3e-7/beta0.5、blocks28–29/LR3e-7/beta0.5，训练 seed 独立。
- 这些是实际 full model forward/backward/optimizer smoke，不是提分证据。只有形成 `smoke.complete.json` 且通过冻结边界、显存、finite loss/gradient 后才允许扩为短训练；最终仍须 n8 generated9-first + corrected15 才能判断提分。
- GPU7 外部 `/data/fjy/worldarena` PID `2659785` 未触碰；GPU0–2 留给先完成的 ORFC 真实 RAFT cache work。

### 2026-08-31 12:20–12:25 CST — GT-RPO 真机门通过；ORFC 监督可学习性门通过

- GT-RPO 四个 1-step 真机 smoke 全部形成 `smoke.complete.json`：flow 冻结前后均通过、无 frozen parameter 异常梯度、shared noise/timestep 一致、loss/gradient finite。
- block29 两臂 peak reserved=`15.8145 GiB`，blocks28–29 两臂=`16.4219 GiB`，均低于 22GiB 门槛；gradient norm 分别约 `0.2285/0.1436/0.1357/0.0591`。四臂初始 preference logit 均为0符合 parent-equivalent 起点，不是提分结论。
- 四个 step1 checkpoint SHA：block29-LR1e-7=`0affd658...`；blocks28–29-LR1e-7=`22a4512b...`；block29-LR3e-7-beta0.5=`8420cad0...`；blocks28–29-LR3e-7-beta0.5=`0b05823a...`。下一步是固定初始 parent reference 的 train8×1 短训练，而不是直接评估单步权重。
- ORFC train8 共8个双向 RAFT cache 已完成，张量结构/hash/finite/manifest/resolved-HDF5 边界全通过。contact fraction 分布为 `0, 0, .0013, .0153, .0193, .1340, .1420, .1447`；保留零-contact 样本作为 false-response 保护，正监督优先后三类高contact样本。
- 7个隔离 ORFC-only 10-step smoke 完成，全部 loss 下降；示例 move-stapler width32/LR1e-3 从 `.21348` 降至 `.18071`，hanging width16/LR1e-3 从 `.28304` 降至 `.27030`。这仅证明分支可优化，不代表已影响 parent 输出；真实 FlowWAM conditioning adapter 正在实现。
- 本阶段 GPU0–6 均实际使用后释放，GPU7 外部任务仍未触碰。

### 2026-08-31 12:31–12:34 CST — ORFC 接入 Official FlowWAM 的真实门通过

- 新增真实桥接：Official P0 与 FlowStream/shared DiT 全冻结；Official RGB-coded flow 经 VAE 得到 parent flow latent；ORFC 物理流特征经 zero-init 3D projection 形成 native flow-latent delta，frame0 delta 强制为0，再进入实际 RGB FM forward/backward。
- focused tests：ORFC core/input/adapter 共 `10/10 PASS`；adapter SHA=`07a3bcce1450e7a4ec2d6af70133fce3fb20fdfdcddebe2d3199868d228b12d2`，trainer SHA=`a4a40cdbfa5d2706e6630444895f8f32ced0bd63d09aee3ff6222101a0e361a5`。
- GPU0–6 七个独立 1-step full-model smoke 全部 `passed=true`，均产生非零 ORFC gradient（约 `.0606–.0960`），peak reserved=`16.4023 GiB`，且各自输出独立 adapter checkpoint SHA。
- 结论：`MECHANISM_PASS_ONLY`。ORFC 已实际影响 FlowWAM conditioning 图，但当前只证明可训练性；在 adapter 推理路径和 paired n8 corrected15 完成前，不得声称提分。

### 2026-08-31 12:36–12:39 CST — GT-RPO 七路 train8 并行启动

- 主控复跑 GT-RPO、preference 与 native checkpoint merge tests 共 `36/36 PASS`，`py_compile` 与 `git diff --check` 通过。GT-RPO module SHA=`d38edec194ade01626cfdd26b9cce0da39f8482c34ea090365bd0e15c9efade0`，8-step trainer SHA=`7814a9768618162e0c4e03aa03a8470a9ae4af00ed245b25626e6923b403e433`。
- 8-step 固定初始 parent reference：先计算 frozen GT/P0 pair，再恢复当前 policy 建图；train8 按冻结 manifest 8 行各一次，禁止 reference 随 optimizer 漂移。
- 首次 launcher 把 lock/log 放入 output，触发 output-not-empty fail-closed；没有 GPU work/checkpoint。保留失败目录作为证据，改为外置 lock/log，使用全新 `-r2` output missing-only 重启。
- GPU0–6 已启动七个配置：blocks=`1/2`，LR=`3e-8/1e-7/3e-7`，beta=`0.5/1.0`；launcher PIDs=`3751533–3751539`，children=`3751541–3751547`。12:38 实时核验七个 child 均在并行加载 5B（每路约18–25GiB RAM、约9 CPU cores）；尚未产生 GPU compute PID/step receipt，故当前状态为 `TRAIN8_LOADING` 而不是完成。
- GPU7 仍为外部 `/data/fjy/worldarena` PID=`2659785`，未触碰。

### 2026-08-31 12:40–12:54 CST — GT-RPO 七路 train8 全部完成

- 7/7 隔离候选均完成 `8/8` 步，`train-step8.complete.json` 皆 `passed=true`，并分别生成 `native-step-8.safetensors`。
- 参数组合覆盖 block29 / blocks28–29、LR=`3e-8/1e-7/3e-7`、beta=`0.5/1.0`；block29 peak reserved 约 `16.707 GiB`，blocks28–29 约 `17.850 GiB`。
- preference logits 仍只有约 `1e-6` 量级、preference loss 约 `.69314`；这说明短训边界稳定，但更新很小，是否提分必须由 paired dev4 corrected15 判断。
- flow 参数保持冻结，梯度与权重均 finite；未使用 dev4/full15 标签训练。

### 2026-08-31 12:45–12:59 CST — native checkpoint 合并两次 fail-closed 后 7/7 闭环

- 首次合并因 merger 只接受 smoke receipt 而拒绝 train8 receipt；第二次因 partial 中 BF16 可训 tensor 与 parent F32 storage 不一致而拒绝。两次均未发布不完整 checkpoint。
- 经 TDD 修复：只接受精确 `8/8` 且 metrics SHA 完整的 train8 receipt；仅在 shape 一致且 parent F32 nbytes 恰为 partial BF16 两倍时，执行流式 BF16→F32 原样转换，其他 dtype/shape 漂移全部拒绝。
- focused merge tests `13/13 PASS`；部署模块 SHA=`cfb8e6b9432df16251aedf13ff74bb98a00789077640e49bedfda1647575f7cb`。
- 7/7 完整 checkpoint 及 `flowwam-native-merged-checkpoint/1` receipt 已产生；block29 每个覆写 30 tensors，blocks28–29 每个覆写 57 tensors。

### 2026-08-31 12:20–12:58 CST — Official native640 数据完整性验证 GO

- 固定 official revision=`a3c1c335f90958258a68e26f674e90ade24ce0ae`，task=`beat_block_hammer`。
- 640 包 SHA=`1513b6933c0f7d346e11efaa7487137bf53ee74ea4b3fc412576667bed91e44a`；320 对照包 SHA=`da66c032b6716dd46cad630bf25905b85a33c1229b1773699a5bca23740a979b`。
- ZIP CRC/安全路径、scene/robot HDF5 ep0–49、逐 episode 帧数配对、JPEG 解码尺寸全部通过；640 路共 `5732+5732` 帧均为真实 `640×480`，非 320 插值证据。
- validation receipt SHA=`818f60787a7da31f1fd695e15f1ace432a88d1a84df6fdf37767752c4e78c284`。可用于不读取 dev4/full15 标签的视觉保真后训练 fastpath。

### 2026-08-31 13:00 CST — GT-RPO paired dev4 异步生成启动

- GPU4/5/6 已各启动一个自动队列，queue PID=`3776673/3776674/3776675`，compute children=`3776677/3776678/3776679`；每个候选严格 seed1 完成后自动接 seed4，同一候选输出隔离。
- GPU4 队列 3 候选，GPU5/6 各2候选，共覆盖 7候选×2 seeds×4 episodes；不等待全局 barrier。
- GPU0–3 预留给 ORFC dev4 双向 RAFT cache；GPU7 外部 PID=`2659785` 未触碰。
- 当前结论：`EVALUATING`；Stage1 结构收据通过后才进 generated9 选择，选完后才读 corrected15/GT-cap。

### 2026-08-31 13:01–13:04 CST — GT-RPO 完整 checkpoint 独立字节审计 PASS

- 7/7 merged checkpoint 均为 `10,137,267,208` bytes、`830` tensors，路径均为非 symlink 且未越界。
- 独立重算 merged/partial/source/metrics SHA；block29 严格 30 个覆写、4 个 BF16→F32，blocks28–29 严格 57 个覆写、7 个 BF16→F32。
- 全部 `261` 个声明覆写 tensor 均已逐字节对照 partial→merged（含独立 BF16→F32 重建）一致。
- 审计报告 SHA=`03e9d74d5de77d961a5b474aa534978eb33d674e12cbc9c414cdba22b835073d`。

### 2026-08-31 13:00–13:04 CST — ORFC dev4 双向 RAFT cache 闭环

- GPU0–3 成功任务 PID=`3779897/3779898/3779899/3779900`，均在约10秒内完成并释放；首次误用 baseline `.venv` 的 PID=`3777042–45` 在 import torch 前 fail-closed，未触发 GPU。
- 四行 manifest SHA=`54ef7d937bb303699084b235d30146541a2d5f15f94a3564325bebdd636e0773`；cache-map SHA=`d2e3d66497c2e3e31501ebbfc98038420c663863d783c24cafca078cb85dd4c1`。
- 四个 cache/receipt/scene/robot lineage 逐项验证 PASS；contact fraction 依次为 `.01933/.07467/.00667/0`。

### 2026-08-31 13:05–13:15 CST — ORFC paired dev4 合同修复后 v4 七卡并行

- v1 因缺式 `sample-manifest`、v2 因将训练长 ID 误当 Stage1 episode ID，v3 因原 manifest 未含 `stage1_episode_name` 而均在 GPU 启动前 fail-closed；无视频、无 checkpoint 污染。
- 最小 TDD 修复保留原正整数 episode 合同，并新增严格 `stage1_episode_name=episode+6位数字`；generic dev4 cache map 派生仍强制四行顺序、manifest/cache/receipt/scene/robot SHA 及 parent/adapter lineage。
- focused tests `12/12 PASS`；部署 helper SHA=`00fa7749977badde7de71cb154da90f2a36e1e4adc7b8c44f4d19bb203e0e38b`，runner SHA=`12eb646fd0e4524eb9a7af40d54350dbea6d2af2e50c198a9068bff9fd74a216`。
- Stage1 专用4行顺序 manifest SHA=`1e9197a9670c7e1fb4436966292bcd6440b42f8e428f53fcfe1023eac5505d26`，不修改 generic cache lineage。
- v4 已在 GPU0–3 启动，queue PID=`3798114/3798115/3798116/3798117`，compute=`3798119/3798120/3798122/3798121`；4卡覆盖7 adapters×seed1/4，队列自动接续。
- 13:15 实时核验 GPU0–6 全部为本项目 compute PID；GPU7 外部 PID=`2659785` 未触碰。

### 2026-08-31 13:05–13:15 CST — native640 视觉保真 fastpath 部署就绪

- 方案直接复用 Official FlowWAM Dataset/RAFT/VAE/RGB flow-matching/CPU offload，不建新调度系统。
- 训练 beat_block_hammer released native640 ep0–7，ep40–49 保留；不读 dev4/full15 标签。仅训 rank8 self-attention q/v LoRA（60 pairs）。
- loss=Official RGB FM + 静态区 VAE-x0 L1 + 多尺度低频特征 L1 + 动作区 MSE 保护，直接针对 Photometric/Image/Aesthetic 并避免抹掉动作。
- main verification `15/15 PASS`，包含真实 PyTorch backward；trainer SHA=`72a7e575db0ba6dcdd11d0e522cdfc7733a0d7401a239b2f23af0406782790f6`，fidelity SHA=`d0eb84529af10beaeee8f8a823a4441c4ec2d0a1ad89c289aa85bbea5ca7c850`，v17 contract SHA=`edfd0b78a4bef7195439d07318e95a2a5c6681196797bc7eb98dacf9c5a86e97`，远端 hash+py_compile 通过。
- 当前 GPU0–6 均被 ORFC/GT-RPO 合法工作占用；首张释放卡立即先跑1-step smoke，通过后并行 balanced/static-heavy/action-safe 三个 train8 臂。

### 2026-08-31 13:28–13:40 CST — 首批 paired Stage1 完成并进入 generated9

- ORFC 已完成 4 个 adapter 的 seed1+seed4（`top-hanging/top-scan/top-stack/neg-lift`），共8 receipts/32 videos。
- GT-RPO 已完成 3 个候选的 seed1+seed4（`b2829-lr1e7-b1/b2829-lr3e7-b05/b29-lr3e7-b05`），另有3个seed1完成；当前共9 receipts/36 videos。
- 对现有 17 receipts/68 videos 批量复验：contract、恰4条、declared=decoded=121、640×480、black0、重算视频SHA全部 PASS。
- 七个paired候选的冻结 remap 已完成，14个 remap Stage1 receipts 完整。
- GPU3 已启动冻结 generated9 七候选串行队列 PID=`3842720`；每个候选内部严格 seed1/4 base→VLM→merge，不读 full15。

### 2026-08-31 13:38 CST — native640 smoke 首次在GPU前fail-closed

- GPU3 启动 PID=`3836049`，完成真实640 dataset、Wan/FlowWAM parent 加载后，在 optimizer/GPU training 前被 v17 trainable whitelist 拒绝。
- 根因：真实 PEFT 命名含 `.lora_A.default.weight/.lora_B.default.weight`，本地合同 fixture 只覆盖了无 `.default.` 形式。无 step、无 checkpoint、无污染。
- 正在做一次最小 TDD 修复：仅允许 PEFT default adapter 的 30 blocks×self q/v×A/B=120 tensors，仍拒绝cross/额外adapter。修复后必须用全新 `smoke-s0-r2` 重启。

### 2026-08-31 13:48–13:54 CST — native640 PEFT 合同修复部署；冷却后自动重启已挂起

- 最小修复严格接受真实 `pipe.dit.blocks.*` 与 PEFT wrapper `pipe.dit.base_model.model.blocks.*` 两种单一布局，禁止混用；adapter 仅允许 `.default`，覆盖必须精确等于30 blocks×self q/v×A/B=`120` 个唯一 tensors，继续拒绝 cross、block30、native 权重、缺项与重复。
- 主控与独立复核均为 targeted tests `20/20 PASS`，`py_compile` 与 `git diff --check` PASS；新 `flowwam_v17.py` SHA=`026a353d3ed48ea364e2554d665d1a967289738e3d417d4dcb1beaf784153135`，远端匿名 hash 回读及 pycompile PASS。
- 13:53 实时 GPU 状态：GPU0–2 跑 ORFC 后续 paired Stage1，GPU3 跑第一个 GT-RPO 候选的 generated9 VLM，GPU4 跑最后一个 GT-RPO paired Stage1；GPU7 仍为外部 `/data/fjy/worldarena` PID=`2659785`，未触碰。GPU5/6 已释放，但最近 receipts 分别为13:47:31/13:47:36，尚未满足10分钟冷却。
- 为避免等待下一轮自动化，已挂起 fail-closed delayed launcher PID=`3859759`：冷却610秒后重新按 GPU5 UUID 查询 compute PID、独占 flock、确认新输出 `smoke-s0-r2` 不存在，再启动1-step native640 smoke；任何一项不满足即退出，不抢卡、不覆盖失败证据。
- generated9 队列 PID=`3842720` 正常：第一个候选 seed1 base 已完成、VLM 正在运行且日志无 Traceback/Error/OOM；当前尚无候选同时具备 seed1/4 `generated-only.csv`，因此冻结 selector 尚未启动，也未读取 corrected15。
- 13:53 Stage1 总进度为 `23/28 receipts`（GT `12/14`、ORFC `11/14`），均完成结构/帧数/分辨率/black0/实际视频SHA复验；剩余5份都有明确 active/queued 归属。当前 generated9 有限队列仅覆盖3个早期GT+4个早期ORFC；后就绪的4个GT+3个ORFC必须在 paired/remap 完成后另开不重叠的第二条有限队列，不能因第一条队列结束而停住。

### 2026-08-31 14:00–14:03 CST — native640 实际上卡；第二条 generated9 队列预就绪

- GPU5 自13:47:31起无 compute PID，14:00再次按物理UUID核验为空；主控精确核验 delayed launcher PID=`3859759` 属于 `huazhi` 且 cmdline 含 artifact_root 后仅终止该等待壳，未触碰任何训练/外部进程，并立即用同一冻结命令启动 `smoke-s0-r2`，trainer PID/PGID=`3866438`。
- 新 trainer 已写 `contract.json` 并完成真实640 rollout-count预处理，当前为模型加载阶段；GPU5 尚未出现 compute PID不代表退出，禁止重复启动。GPU7 外部 PID=`2659785` 仍未触碰。
- Stage1 更新为 `24/28`：新增 GT `b29-lr3e8-b1-train8-r2` seed1 receipt 且结构/SHA复验PASS；其seed4与三个ORFC seed4均在对应旧队列 active，剩余4份无失联。
- 第一个 generated9 候选 seed1 VLM 已完成并自动进入 seed4 package/base；当前仍无双seed CSV，故冻结 selector 尚未启动。
- 后四个GT中已有三个完成双seed：`b29-lr1e7-b1-train8-r2`、`b2829-lr3e8-b1-train8-r2`、`b29-lr1e7-b05-train8-r2`。冻结 remap builder SHA=`4f6c0ca264b86a529e00f26d4b1ad91c63198a2d5a00a18505a6c3f84b29080c` 已为它们生成6份 remap receipts。
- 为利用GPU6且不与早期队列重叠，已挂起第二条有限 generated9 queue PID=`3871030`；两轮空闲窗口满足后再次核验 GPU6 UUID/独占flock/6份remap收据，再依次评分上述三个GT候选。runner SHA=`96c5e39fb88949d3fe653bcfde9be8c5efbcbddff4923798f577979b42ae43d6`；任何条件不满足即fail-closed。

### 2026-08-31 14:01–14:12 CST — native640 smoke PASS；三路 train8 与两条后续 generated9 接续

- `smoke-s0-r2` 在56.5秒完成真实1-step forward/backward/optimizer，receipt SHA=`b7e70103d59bec5d7b9eab51e384f4b6a867bd4eae7e2269b06263c0ca7df434`，checkpoint SHA=`84a10d1678063e90fa751a0ebcfa6961cc4ff12ae8bf2ef9e0508605c86a0629`。
- 独立审计 PASS：120个 checkpoint keys 精确覆盖30 blocks×self q/v×A/B、全部 `.default` adapter、无cross；loss=`.15061736`、gradient norm=`.0020752` finite且非零；Flow before/after frozen；peak allocated=`15.91 GiB`、reserved=`16.44 GiB`，低于22GiB门槛。该结果只证明机制/显存/梯度可行，不声称corrected15提分。
- GPU0/1/2 自约14:01起释放；已挂起三个独立 delayed train8 launcher：balanced PID=`3881644`、static-heavy PID=`3881645`、action-safe PID=`3881646`。冷却180秒后分别再次核验UUID/flock/新输出，再以权重 `(static, perceptual, action)=(.25,.10,.20)/(.50,.20,.20)/(.25,.10,.50)` 启动；任一卡被占则对应臂单独fail-closed，不形成全局barrier。
- 第二条 generated9 首次 PID=`3871030` 因冻结 runner 文件无 execute bit而连续三次在GPU前 `Permission denied`，无candidate输出/无GPU工作；保留日志和stale pid/lock普通文件。恢复使用全新 queue root并显式 `bash` 调用同一哈希 runner，PID=`3878708` 已在GPU6启动，compute PID=`3879515`。
- Stage1 当前 `27/28`：三个后续ORFC seed4均完成并通过结构/SHA复验，ORFC队列全部终止；仅GT `b29-lr3e8-b1-train8-r2` seed4仍在GPU4真实运行。
- 为避免最后4个候选完成后GPU5空等，已挂起一次性第三队列 PID=`3882500`：冷却240秒后严格核验GPU5 UUID/flock、最后GT+3个ORFC双seed收据、冻结脚本哈希，再生成8份remap并依次跑4个不重叠generated9候选；缺任一输入即退出，不造假ready。

### 2026-08-31 14:10–14:31 CST — 28/28 Stage1 闭环；首个 frozen selector 与 selected-full15 启动

- Stage1 已完成 `28/28 receipts`、`112/112 videos`；独立重算全部视频 SHA，并核验每份收据恰4条、declared=decoded=`121`、`640×480`、black0=`0`，全部 PASS。最后一份 GT seed4 receipt SHA=`4b30333468f1a4bbf8f162124399e108b424c675bb36931862000ede3bfc5b9b`。
- 三路 native640 train8 已完成并写出 `train-step8.complete.json`；GPU0–2 于14:30无compute PID，等待严格冷却后进入各自paired dev4，不把训练loss冒充corrected15提分。
- 三条 generated9 队列分别在GPU3/5/6持续推进，首个 ready 候选为 `b2829-lr1e7-b1-train8-r2`；双seed generated-only CSV均4行，seed1 SHA=`f78557d84d056435339b881c55578a7d8f6f7ea528775400a27e31e612039e70`、seed4 SHA=`9c39d2560302fb40b36e3e35254842aaeecd16738460530ea5cf12548e2ae567`。
- 冻结 selector model/policy SHA仍为 `4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa` / `aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`；选择收据明确 `corrected15_used_for_selection=false`，4条中 seed1=`1`、seed4=`3`，selection JSONL SHA=`e1dfbd3539577c1cbfa1b36c4776f7c4c107529c2f66e8a20a1664ec52fe1638`。
- 第一次 selected-full15 staging 因远端无 `/usr/bin/ffmpeg` fail-closed；第二次发现 released instruction JSON 使用 `seen/unseen` 而通用打包器要求单一 `instruction`，仍在评分前fail-closed。随后固定使用 `seen[0]` 构建候选隔离的prompt-bound input，并以项目ffmpeg完成81帧ATR打包；selected Stage1 receipt SHA=`e790027db8462c4a232fb30551c3ca27672032346b0e5a840dd799cc89069660`，package receipt SHA=`c15234735a8fddc1d6142c7b40707e81c84df6818c86d868a2bc332283d7ff9e`。
- GPU4自14:10后连续两轮无compute PID；14:30再次核验UUID=`GPU-651bd630-9a1d-c4c2-1089-8501d6ef1051`、无外部进程/claim冲突后启动首个候选 selected-full15 base，launcher PID=`3905706`。GPU7外部 `/data/fjy/worldarena` PID=`2659785` 全程未触碰。
- 14:34独立核验 base compute PID=`3906050`、PGID=`3905706`、cwd/命令/用户/显存及 selection→stage→package SHA 链全部正确，日志无Traceback/OOM；14:39仍在GPU4正常执行。
- native640 三个train8终态均 `8/8 PASS`：balanced/static-heavy/action-safe checkpoint SHA分别为 `9cb311a77e336f18ea894f9ab71dfcb23bbce93728126272cf81554f707a3d8f`、`815fded5138c7e333f26b7366cd66df314b21d410eb460086dadf16d816663ae`、`2e46c1927b80c691637f856aab6b89e63d7a07c37d6ef6a6e66d86e58fb14489`；均为60对self-q/v rank8 LoRA，flow冻结，峰值reserved约16.5GiB。
- 已部署三个独立 delayed paired-dev4 queue：PID=`3916498/3916499/3916500`，分别绑定GPU0/1/2与上述三份receipt/checkpoint SHA。它们在保守等待480秒后重新核验UUID、compute PID、runner/v17 SHA、flock和新输出缺失，再各自顺序生成seed1→seed4；任何条件不满足仅该臂fail-closed，不形成全局barrier。

### 2026-08-31 14:40–15:01 CST — 双候选进入 full15；native640 最小修复后上卡

- 首个 GT-RPO 候选 `b2829-lr1e7-b1-train8-r2` 的 selected-full15 base 完成，receipt SHA=`2ac0d79d28793d36110ae08c2d91c90efe114365813f1f78595d284eaf6493fa`；selection→selected-stage1→package→base SHA 链独立复验 PASS，日志无错误。
- 第二候选 `b29-lr1e7-b1-train8-r2` 完成 frozen generated9 selector：4条全部选择seed4，selection receipt SHA=`3cb521c262642b352f79aeba0967af0c31a88111bfa58336e555974e1aa69e26`、selection JSONL SHA=`48b83f5f55336018ae7695dd3728cb40213445399c85e3ffa459b47722b761a1`，`corrected15_used_for_selection=false`。selected Stage1/package receipt SHA分别为 `d8f6c004ffd2f82844db6d3eef831a47f8331692c92fceea22a7d89f53dc5ab4` / `d0a6b6819f6539485c23a09383f319970875d56ccb37f6fb05cf58abce7aa0d2`。
- 14:55按UUID与双轮空闲证据在GPU0启动首候选VLM（launcher=`3937626`）、GPU1启动第二候选VLM（`3937702`）、GPU2启动第二候选base（`3937778`）；三者真实compute PID、user/cwd/cmdline和显存均核验正确。
- 原 native640 delayed queue 在GPU启动前安全失败，唯一根因是shell漏传 `CUDA_VISIBLE_DEVICES`；六个预期输出均不存在，无视频/receipt/污染。保留v1失败日志，v2只补 `CUDA_VISIBLE_DEVICES="$gpu"` 并使用新queue root。
- GPU4自首候选base完成后满足两轮超过10分钟空闲，14:56启动 native640 balanced paired dev4 queue PID=`3940655`、compute=`3940679`；15:00已进入真实640×480 Stage1生成并产出首条视频，显存约16.6GiB。队列先seed1后seed4，不重复已有输出。
- 15:00 GPU0–6均为本项目合法工作：0/1双VLM、2第二候选base、3/5/6剩余generated9、4 native640；GPU7仍为外部 `/data/fjy/worldarena` PID=`2659785`，未触碰。
- 第三个双seed generated9-ready候选为 `b2829-lr3e7-b05-train8-r2`，seed1/seed4 CSV SHA分别为 `b5f5625acd5ed2d1e218ce21fdd986701a2974eba540f95148d7dd342c3d0833` / `3342aec359445cbc92a165ab9c0fd0f7fe8bc15cccc97466f5573b18b476b67c`；正在CPU执行同一冻结selector与selected-full15 staging，未读取corrected15。
- 15:03–15:05第二候选base、两候选VLM均完成并释放GPU0–2；候选1仅缺JEPA，候选2仅缺JEPA，未提前aggregate。
- 15:09形成GPU0–2首个明确空闲快照；已挂起严格610秒冷却后二次核验的独立队列：GPU0候选1 JEPA PID=`3959431`、GPU1候选2 JEPA PID=`3959432`、GPU2第三候选base PID=`3959433`。最终phase launcher仍逐卡核验UUID、compute PID、package/selection receipt与目标receipt缺失；任一不满足仅该任务fail-closed。
- 第三候选 `b2829-lr3e7-b05-train8-r2` CPU链闭环：两seed各4视频的121帧/640×480/black0/SHA严格PASS；冻结selector选择seed1=`0`、seed4=`4`，selection receipt SHA=`c26bba263ace398da55bab40f9c3097f08cb09a2cbd2a728bc130c19ca7ba251`、selection JSONL SHA=`2f4ca862a6da1407a6f99b88c19610953a90efd4acda7709006fd43df26feb45`，`corrected15_used_for_selection=false`。selected-stage1/package SHA=`662dbd6f7dd0a294a5d6b552a1382be1f07f5cc412a74247a1f65413f1e89be1` / `6a9a20304d795f7605786017b8cd82c5e953f5854765372f4600f52b017adfd4`。
- native640 balanced seed1已于15:08形成Stage1 receipt并自动进入seed4；GPU4 compute=`3957090`持续运行，未出现重复启动。

### 2026-08-31 15:21–16:07 CST — GT-RPO 两个候选 corrected15 关闭；第三候选继续

- `b2829-lr1e7-b1-train8-r2` 与 `b29-lr1e7-b1-train8-r2` 的 selected-full15 JEPA 均于15:21完成；aggregate receipt SHA分别为 `e83213ff3eae0a5b97674143cb52ae9f01adf30e86cb69ba430ae92ab263c14c` / `06ee1dfa428fa7fe8499d1f8189b68813ece44d5f65828adb2faffc653258471`。
- 冻结 GT-cap corrected15 已写入独立CSV/收据，且再次核验 selector 收据 `corrected15_used_for_selection=false`。
- 候选 `b2829-lr1e7-b1-train8-r2`：corrected15=`0.6716396566`，相对P0=`-0.0181080714`，nonmotion12=`0.7466134199`（Δ=`-0.0229517341`），paired wins=`1/4`，decision=`CLOSE`。主要回撤为 Trajectory=`-0.2400866502`、JEPA=`-0.1003770525`；Semantic=`+0.0689695`、Photometric=`+0.0118974`不足以抵消。evaluation receipt SHA=`091862ad2061bb74ebeb0b03b01b941e69565b46094b0e00ee16fe93d8229fbd`。
- 候选 `b29-lr1e7-b1-train8-r2`：corrected15=`0.6853437681`，相对P0=`-0.0044039599`，nonmotion12=`0.7640295425`（Δ=`-0.0055356115`），paired wins=`1/4`，decision=`CLOSE`。主要回撤为 JEPA=`-0.1067712004`、Image=`-0.0072112935`、Trajectory=`-0.0078817254`；Semantic=`+0.06030275`不足以抵消。evaluation receipt SHA=`f4ed47e715e24f362483920fc00bdcfc58b865994e748b176ef3c5ea73242f04`。
- 结论：当前8-step GT-RPO即使提升Semantic，也系统性破坏JEPA；blocks28–29还会显著破坏Trajectory。这两条配置不晋级、不扩量。第三候选 `b2829-lr3e7-b05-train8-r2` 的selected-full15 base已完成，VLM/JEPA/aggregate仍待接续。
- native640 balanced seed1/seed4 Stage1均已完成；进入严格视频/收据复核与generated9-first选择。static-heavy/action-safe尚未生成dev4，等待合法冷卡missing-only启动。

### 2026-08-31 16:07–16:24 CST — C3与native640继续占满合法卡

- 第三GT-RPO候选 `b2829-lr3e7-b05-train8-r2` 的base已完成；16:16按双快照冷卡合同在GPU0启动VLM（launcher=`4034158`）并在GPU1启动JEPA（`4034234`）。JEPA于16:17完成；VLM compute=`4034164`仍运行。VLM收据完成后立即aggregate与GT-cap corrected15。
- native640 balanced 的seed1/seed4共8视频已独立严格复核：每seed4条，declared=decoded=`121`、`640×480`、black0、逐视频SHA全部PASS。
- native generated9两次均在GPU前fail-closed：第一次原生Stage1文件名为 `episode000000.mp4` 而冻结scorer要求 `episode0.mp4`；第二次误指向无episode0素材的raw fresh40输入。两次均无GPU compute、无package终态、无分数污染。
- 已使用SHA绑定的只改名symlink remap修复，不转码、不重生视频；remap工具 SHA=`694ab4dbfd1280d0b0e7cbd4e13c9a8a4562e58df5d48d04d832434db17dad88`。scorer dataset改为Round1已验证的D-p025 dev4 remap dataset；runner SHA=`a6bf9d96b3c3a8b64306c94847dab1389e4947e491b3129e4b72c4ed3068a965`。
- 16:22 balanced seed1/seed4 package均完成并在GPU4/6分别进入generated9 base，launcher=`4041655/4041656`、实际compute=`4043161/4043225`；完成base后同一隔离队列自动接VLM与merge。
- GPU2满足长时冷却且static-heavy输出不存在，16:23启动 `train8-static-heavy-s2` paired dev4，launcher=`4044788`、compute=`4044811`，先seed1后seed4。action-safe等待GPU1自16:17后满足10分钟冷却即missing-only启动，不形成全局barrier。
- 16:24合法GPU使用：GPU0 C3 VLM，GPU2 static-heavy生成，GPU3/5 ORFC generated9，GPU4/6 balanced generated9；GPU1短暂冷却，GPU7外部任务未触碰。

### 2026-08-31 16:26–16:29 CST — 第三个GT-RPO关闭；native三路全进入评测

- C3 `b2829-lr3e7-b05-train8-r2` VLM于16:26完成；CPU aggregate与matched-GT cap随后闭环。aggregate SHA=`4e9225c9a1142cb6f9a85259147e45ce581462d4ccfe403482ad0eb28449fd3f`，evaluation SHA=`3ecbfdb9de4901de12f39a8ed526916a91a38933b89c170aef34d691eb2ba1ce`。
- C3 corrected15=`0.6870465593`，相对P0 Δ=`-0.0027011687`；nonmotion12=`0.7661709931`（Δ=`-0.0033941609`），paired wins=`2/4`，decision=`CLOSE`。Trajectory显著提升 `+0.0895505946`，但JEPA下降 `-0.1299118693`、Image下降 `-0.0069345484`，整体仍未过门。
- GT-RPO本轮3个完整corrected15候选全部关闭：共同失败模式是JEPA显著回撤，说明当前GT-vs-P0偏好目标没有保护表示/时序相似度；不得继续同机制扩量。
- native640 balanced 双seed generated9 base持续在GPU4/6运行；static-heavy paired dev4在GPU2运行。
- GPU1在C3 JEPA结束后满足超过10分钟冷却，且action-safe两seed输出均不存在；16:28启动 `train8-action-safe-s3` paired dev4，launcher=`4049786`，先seed1后seed4。至此native640 balanced/static-heavy/action-safe三路均已进入真实paired dev4/generated9链路。

### 2026-08-31 16:38–16:46 CST — balanced与5个ORFC进入full15；两项GPU评分已启动

- native640 `train8-balanced-s1` 的双seed generated9闭环：seed1/seed4 CSV SHA分别为 `ba2fb32d2e3cb6f3060852fe4a2063252b0de27ef05b30bba0a49195d42f7fa2` / `b806a9000b28386214b75d6a74ef8e6200c89bea14c850a425eeda99d778b83a`；冻结selector选择seed1=`1`、seed4=`3`，selection receipt SHA=`937dae6881b5797adf930e4e7e7fce5679fe6d96f77e10bc16af6bcab0441f91`，明确 `corrected15_used_for_selection=false`。selected-full15 package receipt SHA=`062a927a8828365b726970749bb7325de438043aa3af3d06825402f5e36a0618`。
- 5个已具备双seed generated9的ORFC候选全部完成同一冻结selector与selected-full15 staging，未读取corrected15：`mid-move` 选1/3、其余 `neg-lift/mid-bottles/top-hanging/weak-place` 均选0/4。对应selection receipt SHA依次为 `6decff3f0629e68cf6dc816cbfcec78396434d4c25e0de9f414db29d865a2d6b`、`1da80e568c069468279976effd1ba6e51d1bc31cce7e422c2d150c4b59edb5b6`、`fdadbd36910c003aa3ec9f744e115d9f4e46dadac0935210f0ba971ffa8b0165`、`008e9c42d1bb34f6124f7e0d357854ac70ae254370aff9fc665c739df37d7545`、`e26cc3f0236196f79663bd072601f4c37682e2a904b0cfb2e9a2699f7aa4905f`。
- selected generated9九项均值：balanced=`.561909`；ORFC neg-lift=`.563333`、mid-bottles=`.562544`、weak-place=`.563521`、mid-move=`.562915`、top-hanging=`.563043`。该值仅用于full15执行优先级，不作为晋级证据。
- GPU0在16:37/16:40连续空闲且自16:26无compute PID；按物理UUID、compute PID、候选进程与flock再次核验后，16:42启动balanced selected-full15 base，launcher PID=`4070295`、compute PID=`4070553`。
- GPU5自16:28后满足超过10分钟空闲；按相同合同于16:45启动generated9优先级最高的ORFC `weak-place-w16-lr1e4` selected-full15 base，launcher PID=`4079760`、compute PID=`4080018`。两任务均属huazhi、本项目artifact_root、cwd=`/home/huazhi/nlh/baseline`，日志已进入SAM3 base评分且无Traceback/OOM。
- static-heavy seed1与action-safe seed1 Stage1收据均已完成并严格核验；各自队列自动接seed4，未重复启动。GPU7外部 `/data/fjy/worldarena` PID=`2659785` 未触碰。
- static-heavy seed4于16:46完成；两seed remap receipt SHA=`11b4758949a1e41c24cf46ae220e7b94300d0f41d93d8b1e7b32ee8f86c41c2a` / `febac99260670373ecef10f9cf7ed53a1becd98db0a057c88f371096ad5af37a`，视频SHA与121帧/640×480/black0合同复验PASS。已安排仅在GPU2冷却满10分钟、UUID/compute PID/flock/输出缺失再次核验通过后执行seed1→seed4 generated9的延迟队列 PID=`4089061`；此前创建的PID=`4088341`因awk引号审计发现会fail-closed，按user/cmdline/artifact_root精确核验后仅终止该等待壳，未触碰GPU或其他任务。
- 16:56 balanced与ORFC weak-place的base均完成，receipt SHA分别为 `2cbfabf35a5f2fafb8e69dc11e9d337ae2cda61822c97153e54ec1ab2fd0e72b` / `e71d582f814da1a3fe8ab95adf12e4ac96d0d824b1a53f873cb705c13c7d8620`。GPU4/6自balanced generated9结束后已冷却超过10分钟，立即并行启动balanced VLM/JEPA，launcher=`4090289/4090365`；JEPA已完成，receipt SHA=`1fff491f7569f219f16c6f4c3d1c6e2a4ae86d995067c05d4fed167fbf0865f5`，VLM继续在GPU4。
- action-safe seed4于16:56完成，Stage1 receipt SHA=`7455488ec9242ec66092cd283016b3b17cec68bf7e63f16dd33aeb1d0564ad26`；双seed remap receipt SHA=`dc7529d6ddd48debc5ecd1548fc7d3339c5d6859b16298a88a44016a0e4e6ee8` / `e2d6ecf02b931da3f8d286e6b816a35e230b5163460bc2e87f75895ea2e8e1bf`，已安排GPU1冷却后严格核验并执行双seed generated9的队列 PID=`4092659`。
- GPU2在static-heavy生成结束后已连续空闲超过10分钟；gate-critical full15优先于cheap score，16:58启动weak-place VLM launcher=`4093338`/compute=`4093339`。static-heavy延迟队列到点若检测GPU2被占会按设计fail-closed，后续改投首张合法冷卡，不会重复或抢卡。

### 2026-08-31 17:01–17:05 CST — full15并行接续；top-scan进入selected评测

- static-heavy GPU2延迟队列 PID=`4089061` 在17:02按预期因GPU2正运行weak-place VLM而fail-closed，未启动scorer、无partial输出；已改投GPU5并安排冷却后再次核验的队列 PID=`4098882`。action-safe GPU1队列 PID=`4092659` 仍在保守冷却，未重复启动。
- weak-place base已完成但VLM/JEPA可相互独立；GPU0 base结束后计划冷却满10分钟即启动weak-place JEPA，严格延迟启动器 PID=`4097466`。weak-place VLM继续在GPU2。
- ORFC top-scan seed4 generated9 CSV于17:03形成，双seed闭环后立即执行冻结selector；4条全部选择seed4，selection receipt SHA=`b555b2e787525c42f6875e7c6e0732fc0a76ce8da71940f70617fc3ad10bea7f`，selection JSONL SHA=`dc07a09a51a7f024be3d547e95bb0003dce910dc88019a20242ac82d35e9ab68`，`corrected15_used_for_selection=false`。selected-full15 package SHA=`b76e0b1467c67bdc2d9c14fcc0399d1722178b13e8000766b33412f2da6b64a9`，已进入ready full15队列。
- GPU3原ORFC有限队列在top-scan闭环后自动转入top-stack seed1 generated9 base，compute PID=`4098637`，无需人工重启。

### 2026-08-31 17:06 CST — native640 balanced corrected15关闭

- balanced的VLM与JEPA齐全后立即CPU aggregate+matched-GT cap；aggregate receipt SHA=`af2eae03566231d3333a38ea9f1309056cdf82355bb7987772dd50216533e8f2`，evaluation receipt SHA=`6130e865485c50350ecbac2106536100771a4e5ad9e26a2fde3cbabe536b7acb`。
- `native640-train8-balanced-s1` corrected15=`0.6742783964`，相对P0 Δ=`-0.0154693316`；nonmotion12=`0.7502644166`，Δ=`-0.0193007374`；paired wins=`1/4`，decision=`CLOSE`。
- 该局部保真loss确实提升了直接主攻项：Photometric=`+0.00922880`、Aesthetic=`+0.00134353`、Background=`+0.00339127`、Semantic=`+0.02563475`；但Image仍下降 `-0.00825788`，JEPA下降 `-0.11295709`，Trajectory下降 `-0.15065232`，完全抵消收益。结论：当前8-step self-q/v局部保真微调仍系统性破坏表征相似度和轨迹，balanced不扩量。
- weak-place JEPA已完成，receipt SHA=`0a05e4708b5e5e3bdfda999b2fbb42d59828bb01951904807fe2742729c5c05a`；VLM仍在GPU2，结束即aggregate/corrected15。

### 2026-08-31 17:08–17:13 CST — ORFC weak-place接近持平但关闭；top-scan进入full15

- weak-place VLM于17:08完成，随后立即CPU aggregate+matched-GT cap。corrected15=`0.6899375995`，相对P0仅 `+0.0001898715`，nonmotion12 Δ=`-0.0000977638`，paired wins=`2/4`，未达到预注册 `+0.002` 门槛，decision=`CLOSE`；corrected15 receipt SHA=`1d1dd1351b90b73fca6789d2004212ed0cfa9a4124d0350ae0210783e9475755`。
- weak-place显著提升Trajectory=`+0.11274294`、Flow=`+0.00251533`、Photometric=`+0.00473756`、Semantic=`+0.00927725`，但JEPA=`-0.10240295`、Image=`-0.00924122`、Background=`-0.01031418`，总体仅近似持平。这说明ORFC能够改变动作/轨迹响应，但仍未解决表示相似度与画质保护矛盾，不扩量。
- GPU1/GPU5分别已启动action-safe/static-heavy generated9 seed1 base；GPU3继续top-stack seed1 VLM。GPU6自balanced JEPA结束后已空闲超过10分钟，17:13核验UUID/PID/flock后启动top-scan selected-full15 base，launcher PID=`4111699`。

### 2026-08-31 17:22–17:27 CST — top-scan三相评分并行；neg-lift接力full15

- top-scan selected-full15 base于17:22完成，receipt SHA=`c2ba92a01e5040766e082308262b3c067b275a4773fe4caefa6f350307a7641f`；GPU0/GPU2在各自已有超过10分钟空闲证据、再次核验物理UUID/compute PID/claim后，分别启动top-scan VLM/JEPA，launcher PID=`4122483/4122559`。
- top-scan JEPA于17:23完成，`jepa.complete.json`已落盘；VLM在17:26仍为真实compute PID=`4122491`，显存约19.4GiB，无Traceback/OOM。VLM终态出现后立即aggregate与matched-GT cap，不提前读corrected15。
- GPU4臧16:45后已长时闲置；17:23核验后启动下一个ORFC `neg-lift-w16-lr1e4` selected-full15 base，launcher PID=`4124114`。17:26真实compute PID=`4124434`正常运行，显存约4.7GiB。
- 同时GPU1/GPU5分别在执行action-safe/static-heavy seed4 generated9，GPU3执行top-stack seed4 generated9；全部为huazhi且cmdline/cwd属于artifact_root项目。GPU6因top-scan base刚于17:22结束处于冷却期；GPU7外部 `/data/fjy/worldarena` PID=`2659785`保持不触碰。

### 2026-08-31 17:31–17:36 CST — top-scan corrected15关闭；neg-lift三相并行

- top-scan VLM于17:31完成；base/VLM/JEPA三相齐全后CPU aggregate与matched-GT cap闭环。第一次aggregate命令因缺 `PYTHONPATH` 在import前fail-closed；补上固定 `/home/huazhi/nlh/baseline/src` 后成功。corrected writer的前两次调用分别因错误模型名CSV路径和8行raw merge输入fail-closed，均未写corrected终态；最终使用经核验为4条识别行的 `csv_results/aggregated_results.csv` 完成。
- `orfc-top-scan-w16-lr3e4` corrected15=`0.6782114177`，相对P0 Δ=`-0.0115363103`；nonmotion12=`0.7552828475`（Δ=`-0.0142823065`），wins=`2/4`，decision=`CLOSE`；corrected receipt SHA=`b8ee046c970db5e5a1dd213713d3d72fec8c9461b4ec10a9df348c5c6fa78eba`。
- top-scan提升Semantic=`+0.03601075`、Background=`+0.00808029`、Photometric=`+0.00801596`，但JEPA=`-0.09494290`、Trajectory=`-0.11885005`、Image=`-0.00770836`，与已关闭候选的核心矛盾一致，禁止扩量。
- neg-lift base于17:32完成，base receipt SHA=`045c56c78fe876bbe27eda7073cb98967385789a9e8cbf344dbaf6cb5a65ba90`。GPU2/GPU6在各自真实最后活动后已满足10分钟，并重新核验UUID/compute PID/receipt/lock；17:36并行启动neg-lift JEPA/VLM，launcher PID=`4139651/4139652`。JEPA于17:37完成，JEDi=`0.7931349277`；VLM真实compute PID=`4139657`在GPU6正常运行。
- action-safe/static-heavy/top-stack的seed4 VLM仍在GPU1/5/3正常运行，完成后由原队列自动merge为generated-only CSV；不重启。GPU7外部PID=`2659785`仍未触碰。

### 2026-08-31 17:41–17:48 CST — neg-lift关闭；三个新候选staging与四路full15 base

- top-stack/static-heavy/action-safe的seed4 VLM及generated-only CSV于17:36/17:41/17:41完成；三者两seed generated9均齐全。使用冻结selector model/policy SHA=`4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa` / `aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7` 并行完成选择与staging，`corrected15_used_for_selection=false`。
- top-stack选seed1/4=`0/4`，selection/package SHA=`7a34933ad703333d3a723a7033854dda36aa9a36968d4a6cedbd51cb88cfe35b` / `d4a2ade3ee4b5ea4b8451f6af3952d0873f5c6d219ea071a4d985ba5ebbd1151`；static-heavy选`2/2`，SHA=`d607fca845d0986c8e993824ca56d541f43d1d88ba69938a10cda60da25bb217` / `8e7740377a9a1bbe0c28c7c852902e3d145ef42cad03d2dfb63474c64ebae4bc`；action-safe选`0/4`，SHA=`72e5f5dcd68d40c68285ea9322d16b64d85304b244232cd2cd66d95ce7240e43` / `7adfd42161588a0efad3acb173bf9b756b259febd6eb3127f4bb007b70c0f5ce`。
- neg-lift VLM于17:44完成并立即aggregate+GT-cap。corrected15=`0.6762555961`，相对P0 Δ=`-0.0134921319`；nonmotion12=`0.7519110199`（Δ=`-0.0176541341`），wins=`0/4`，decision=`CLOSE`；receipt SHA=`d8baf80a9f913d49d1c2aa24262850d678c4461418d57d34e4f64dfcedc46e38`。
- neg-lift提升Dynamic=`+0.00580464`、Flow=`+0.00366299`、Semantic=`+0.01916475`，但JEPA=`-0.08389124`、Trajectory=`-0.13296128`、Image=`-0.00907645`，继续证实ORFC目前无法保护排名关键项，禁止扩量。
- 17:45前后GPU0/GPU4已满足真实双快照冷却，启动static-heavy/action-safe selected-full15 base，launcher=`4152303/4152304`；17:47 GPU2/GPU3的第二个超过10分钟空闲快照成立，启动top-stack/mid-move base，launcher=`4154948/4154949`。四卡均为missing-only、候选隔离输出；GPU7外部任务仍未触碰。

### 2026-08-31 17:53–17:55 CST — 六路selected-full15 base并行

- 17:53实时核验：static-heavy/action-safe/top-stack/mid-move四路base均存活，实际compute PID分别落在GPU0/4/2/3，已产生部分base中间输出，无`base.complete.json`、Traceback/OOM或错误终态，不重启。
- 实际top-hanging目录核定为 `top-hanging-w32-lr1e4`（非w16）；selected-full15 package receipt SHA=`c33e7fb6...c3f93`。mid-bottles package receipt SHA=`6f869975...e752b`，两者均已有冻结selection收据且尚无base终态。
- GPU1/GPU5在17:43后第二个超过10分钟空闲快照成立；17:54再次核验物理UUID、compute PID、package/selection及目标receipt缺失后，启动mid-bottles/top-hanging selected-full15 base，launcher=`4162365/4162366`。
- 至此GPU0–5六路均为可影响corrected15决策的base工作；GPU6只有冷却、暂无独立ready的VLM/JEPA阶段，不做重复或无价值填卡。GPU7外部 `/data/fjy/worldarena` PID=`2659785`未触碰。

### 2026-08-31 17:55–18:04 CST — 六路base全部闭环；GPU6串行JEPA

- 六路selected-full15 base全部成功完成，每路均有4条`generated_results.json`和有效base receipt，无Traceback/OOM/错误终态。receipt SHA：static-heavy=`be55cdec0683e66b2bc46ffd2e91caa41717b865ebd195308ada776f60934d92`，action-safe=`6143225723cdbc559d197e9990769f11ce1ee0068149be5904441989ac4874b1`，top-stack=`f23a0fd7f43bb07d55bdabd8b7db4934a3bcc0cb3857eb38c1eca27fe755e31c`，mid-move=`4b402bc975de51643d9182e159fdfdfbd7ac00a09abe9df8218ddb33a6818e20`，mid-bottles=`61dfc444cda04c9df3573966a727b70caa1905567f8c2a585b4c3de9c186d785`，top-hanging=`c287f61802bf5c20b1aeecad3dfb79ef2303420e74859a5fdbc9636c5cb05062`。
- 18:03时GPU0–5的base均刚结束，仍需从实际结束后建立新的10分钟双快照冷却，禁止立即转VLM。GPU6臧neg-lift VLM结束后已满足冷却且六路JEPA均ready。
- 为避免六个约1分钟JEPA之间空卡，18:04启动GPU6 missing-only串行队列 PID=`4172649`，顺序static-heavy→action-safe→top-stack→mid-move→mid-bottles→top-hanging；任一phase失败即fail-closed，不跳过错误或重算已完成receipt。
- GPU0–5冷却成立后将各自执行一个独立VLM；JEPA/VLM两相齐全者立即CPU aggregate+GT-cap，不等其他候选。GPU7外部任务未触碰。

### 2026-08-31 18:05–18:15 CST — 六路JEPA闭环；六路VLM并行

- GPU6串行JEPA队列PID=`4172649`正常退出，无Traceback/Error/Exception/Failed/Killed/OOM，只有标准torch FutureWarning；日志 SHA=`1f176fe820043beaaa714ad9d9c192995d471985a69230e0d968640a4f99c2cc`。
- 六份JEPA receipt全部有效：static-heavy=`dc48de0cef72d406924cb289bcd8a02b5294d65c836ec4758bb357c45be138a7`，action-safe=`412d7fb6019657f3e9390425625bbd1c406c690c00f8ed662c64861102a58afd`，top-stack=`331e52c3ec3572f886c3debfc34efffbdd40d86505cdcdc8519f7317199c09d0`，mid-move=`63d6a3c24c6df16a1b1967f90244d1323454211b8feaf5e64eb23d5f176544eb`，mid-bottles=`52e2876a5e0b0ee96a3b4011b76d04c7b3b752da2edbbddb659658c9f17a6674`，top-hanging=`23c365e420a8b35694373ee951fa894730c1a5ca2bffc068d7627a163643eae5`。
- 18:14与18:03的物理UUID快照相隔10分32秒，GPU0–5中间无compute PID，所有新receipt的`physical_gpu`均为GPU6；因此GPU0–5全部满足不间断冷却合同。
- 18:14并行启动六路missing-only VLM：GPU0 static-heavy PID=`4184695`，GPU1 mid-bottles=`4184698`，GPU2 top-stack=`4184700`，GPU3 mid-move=`4184703`，GPU4 action-safe=`4184705`，GPU5 top-hanging=`4184710`。18:15已全部出现真实compute PID，各卡显存约17.47GiB，无重复或输出交叉。
- GPU6因刚完成JEPA进入新冷却，且六路只剩各自VLM，暂无可并行的独立GPU阶段；GPU7外部PID=`2659785`继续保留。

### 2026-08-31 18:23–18:27 CST — 六路corrected15全部闭环；同类机制停止扩量

- 六路selected-full15 VLM均正常完成，无Traceback/OOM/错误终态；VLM receipt SHA：static-heavy=`9c106771e685d458d172f54556e4f5a25ac97c7b4435da621a7ac527dece8195`，action-safe=`313e7fd6767b56285bc612026f1c9bf2584771e87c06b525ef97e19ac69c76b5`，top-stack=`85241a393fc2878c0e1c92eb65f2c57977b18cc6d6b486949bea0943ab3062aa`，mid-move=`6c6646b0274d47b531eccf13ed72054cb39b1ff2977dce1ddd1977ed700bd67d`，mid-bottles=`565cd9a4c2b7f2837876b8f9bd31be28c234b97e88287764879e47426d7f9f76`，top-hanging=`50a0f926a49965f48299abe5f81dc80d487f12671bcd570e11820d8194f53edc`。
- 六路随后并行完成CPU aggregate与matched-GT cap corrected15；所有selector收据均明确未使用corrected15做选择，基线为P0 dev4 corrected15=`0.689747728`。
- `native640-static-heavy-s2`：corrected15=`0.6807265815`，Δ=`-0.0090211465`，nonmotion12 Δ=`-0.0114594608`，wins=`2/4`，CLOSE。Semantic=`+0.06665025`，但JEPA=`-0.08004472`、Trajectory=`-0.10986684`；receipt SHA=`e871305ce53644b1082039a6eb7e04606505b1d6c079f3536a89d77ca5a22af5`。
- `native640-action-safe-s3`：corrected15=`0.6912731595`，Δ=`+0.0015254315`，nonmotion12 Δ=`+0.0016579958`，wins=`2/4`，仍低于预注册晋级门槛而CLOSE。Trajectory=`+0.10571180`、Semantic=`+0.03222650`、Background=`+0.00507812`，但JEPA=`-0.10954574`、Image=`-0.01006181`；receipt SHA=`c320c7c9c96d42560e75a80f316448f35a9e988ee14c37d15a9520c8f90c8670`。
- `orfc-top-stack-w16-lr1e4`：corrected15=`0.6896954674`，Δ=`-0.0000522606`，nonmotion12 Δ=`+0.0000369387`，wins=`2/4`，CLOSE。Trajectory=`+0.09061536`、Photometric=`+0.00496474`，但JEPA=`-0.08278426`、Image=`-0.00728485`；receipt SHA=`752a3dcabb90aadf70194fa45a222dabad3dcf2c14b202471e72f805cab421b5`。
- `orfc-mid-move-w16-lr1e4`：corrected15=`0.6774309306`，Δ=`-0.0123167974`，nonmotion12 Δ=`-0.0153974954`，wins=`2/4`，CLOSE。Photometric=`+0.00530748`、Semantic=`+0.03271475`，但JEPA=`-0.09730372`、Trajectory=`-0.10827581`；receipt SHA=`c3daa6534e070bc4eb1056db46f5c86690b93d69ad98605d256ad54f8429e8f5`。
- `orfc-mid-bottles-w16-lr1e4`：corrected15=`0.6856846584`，Δ=`-0.0040630696`，nonmotion12 Δ=`-0.0048813186`，wins=`2/4`，CLOSE。Trajectory=`+0.04720468`，但JEPA=`-0.10593703`、Image=`-0.00777786`；receipt SHA=`f3ca6c5837789292dc9ae73423ebb9f1c6c7d6a8fbc15c75acba1308c6b800d8`。
- `orfc-top-hanging-w32-lr1e4`：corrected15=`0.6701104440`，Δ=`-0.0196372840`，nonmotion12 Δ=`-0.0246638560`，wins=`1/4`，CLOSE。Photometric=`+0.00514723`、Semantic=`+0.02136225`，但JEPA=`-0.09605578`、Trajectory=`-0.20876781`；receipt SHA=`9d95291a04cde93df370678b8ecd5a938d839e7ef2d93be70beff71738486607`。
- 本轮共同结论：native640/ORFC能局部提升Trajectory、Photometric、Semantic或Background，但所有候选JEPA均下降约`0.080–0.110`，Image全部不升；继续同类block/LR/seed扫没有门控价值。最佳只是action-safe的`+0.00153`和top-stack的近持平，均不扩量。下一轮必须先解决表示保持（JEPA/Image）再叠加动作响应，限制为少量机制改变候选而非参数扫。

### 2026-08-31 18:31–18:38 CST — 七卡启动action-safe局部保护因果筛选

- 用户批准使用7张卡；实时复核后可用的是GPU0–6。GPU7仍存在外部 `/data/fjy/worldarena` compute PID=`2659785`、显存约12.7GiB，继续不触碰。
- 训练入口 SHA=`72a7e575db0ba6dcdd11d0e522cdfc7733a0d7401a239b2f23af0406782790f6`；loss module SHA=`d0eb84529af10beaeee8f8a823a4441c4ec2d0a1ad89c289aa85bbea5ca7c850`；validation receipt SHA=`818f60787a7da31f1fd695e15f1ace432a88d1a84df6fdf37767752c4e78c284`；Official parent SHA=`e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`。
- GPU6首个 `action-lr3e7` 试启因漏传`PYTHONPATH`在import前安全失败，未创建checkpoint、未占用CUDA。改用新隔离root `as-lr3e7-r2`，避免混用失败日志。
- 在两轮空闲快照跨过10分钟后，7个train8候选于18:36并行启动并真实进入CUDA：GPU0 `as-action060` compute=`15051`；GPU1 `as-action075`=`14651`；GPU2 `as-static030`=`14833`；GPU3 `as-static040`=`14762`；GPU4 `as-perc015`=`14929`；GPU5 `as-maskthr10`=`14988`；GPU6 `as-lr3e7-r2`=`14758`。GPU0–6均为huazhi、本项目trainer、隔离output root。
- 七路共享原action-safe配置/数据/seed，仅单变量改变：action weight `.60/.75`、static weight `.30/.40`、perceptual weight `.15`、motion threshold `10/255`、LR `3e-7`。目的不是继续旧ORFC/seed扫，而是快速判断JEPA/Image损伤是否来自更新幅度、动作保护区或静态/多尺度保真比重。
- 任一路checkpoint先完成即先进入paired dev4 seed1/seed4→冻结generated9选择→selected full15→matched-GT corrected15，不等待其余候选；首门仍为相对P0至少`+0.002`且Image/JEPA不得显著回撤。

### 2026-08-31 18:44–18:52 CST — 七路训练闭环；冷却感知dev4队列已部署

- 七路train8均成功完成；每路 `passed=true`、`completed_steps=8`、120个self-q/v LoRA tensor、FlowStream与parent训练前后冻结、8行metrics完整且SHA与receipt一致，日志无Traceback/RuntimeError/OOM/Killed。完成时间介于18:44:29–18:45:37。
- checkpoint SHA：action060=`01ddc0579008d0108974cd464588e76e740bb7175568c3af943735486640ec2d`；action075=`3f326c714edeaf387ef4cdf986e6942a293f448fddb4919dc97a4dfa51b6f8a1`；static030=`518ed65bc785b7cc9713bd2df867b51ab65246a5bdc96dc27a77a777495fc59c`；static040=`5e2bb9b36f834f69eab50eefa4f5cd3976e554bd62773b0760600c5d42c88210`；perc015=`791b7af5c86f018e5306a692f51e710dffa95e7ca8ed158ff85a1830dd8d03e3`；maskthr10=`5a938c18ae2e7e9c9fcaf598acc729243bb7e0465a6a7b7c11d8711950295779`；lr3e7-r2=`d926f92ccfe8c7c28e91a485aecf54deeceeeffc2cb612bae3300e37f5da48c7`。
- 审计边界：action/static/perceptual四类单变量直接记录在contract中；`maskthr10`与`lr3e7-r2`的阈值/LR未写入旧contract，只能由已保留的实际启动命令与日志追溯，不能声称仅凭receipt密码学证明参数。
- 为避免等待下一轮心跳，新增最小Round4 dev4队列脚本 `baseline/scripts/run_native640_round4_dev4_queue.sh`，本地/远端SHA=`17ba1386594397c35438d6ae650326712bee17743a8a8b375c3e61920ed8a59a`。它只做receipt/checkpoint/runner/model SHA、variant、UUID、flock、输出缺失和GPU空闲检查，并从训练receipt mtime等待601秒后串行执行本候选seed1→seed4；不包含训练或评分框架扩建。
- 七个队列已隔离启动并存活：action060 PID=`30536`→GPU0；action075=`30418`→GPU1；static030=`30397`→GPU2；static040=`30740`→GPU3；perc015=`30631`→GPU4；maskthr10=`30773`→GPU5；lr3e7-r2=`30611`→GPU6。到达各自not-before后会再次核验物理UUID与compute PID；失败仅影响单臂且不抢卡。GPU7外部PID=`2659785`继续不触碰。

### 2026-08-31 18:55–19:01 CST — 七路seed1生成全部真实上卡

- 七个队列均在各自receipt冷却窗后通过UUID/compute PID/SHA/flock/输出缺失复核并启动seed1，未发生fail-closed或外部占卡。
- 物理GPU与真实compute PID：action060 GPU0=`31443`；action075 GPU1=`33357`；static030 GPU2=`33284`；static040 GPU3=`31718`；perc015 GPU4=`31635`；maskthr10 GPU5=`31180`；lr3e7-r2 GPU6=`31356`。`/proc`逐项确认user=`huazhi`、cwd=`/home/huazhi/nlh/baseline`、PPID指向对应队列、cmdline只含本项目artifact_root与各自checkpoint/output。
- 19:00快照GPU0–6均有真实compute PID且利用率/显存符合Stage1生成；GPU7仍仅外部 `/data/fjy/worldarena` PID=`2659785`。
- 当前七路各仅出现seed1首个MP4，无任何`stage1-only.receipt.json`；因此尚不把partial视频计作完成或做结构验收。队列将在seed1完整收据后自动进入seed4。

### 2026-08-31 19:06–19:14 CST — 七路seed1严格闭环；seed4自动接续

- 七路seed1均形成终态`stage1-only.receipt.json`，每臂4/4视频；逐视频复验声明帧/receipt解码帧/OpenCV实际解码帧均为121，实测640×480，receipt与逐帧重算均black0，实际SHA与receipt逐行一致。
- seed1 receipt SHA：action060=`c10ecd501af911abf9104119e21ca70bc92e4f8550f51f5203931cd81c444fb2`；action075=`8a2f61d8ed6c5fdcea8cf9fbe468a7bea2260d5f02b717ebda606d1a5ab56b33`；static030=`60d68c33a6ccc483a5040c59e9e20b7a646df97f0f9c6f37ed3d6b8e6e151c89`；static040=`5676137ab38c18036fa916043291444d0588fe77c8fc643a7663493603a876b4`；perc015=`1ab36749effc66ae9f8c7a2b33460616c1f28b3d4996074d868f304314a01f7f`；maskthr10=`b6d8872bba0519533de9243deda1abbcc57b7415710cfaf39267b74c66c8b9d4`；lr3e7-r2=`975d3f70214637bbb33a48db3521e68aa5587e44163554e15a8d7fde738516c5`。
- 七个原队列已无人工等待地自动进入seed4；19:14快照每臂完成2/4 partial、尚无seed4终态receipt。GPU0–6均为本项目Stage1 seed4进程、利用率100%、显存约16.55GiB；GPU7仍为外部`/data/fjy/worldarena`进程，未触碰。
- 当前Round4仍无generated9或selected-full15/corrected15终态，因此不得声称提分。准确下一门为：seed4终态结构验收→冻结generated9选择→selected full15→matched-GT corrected15；任一路先完成先评，不等待七路全局barrier。

### 2026-08-31 19:18–19:25 CST — 七路双seed闭环；generated9自动队列就绪

- 七路seed4均形成终态receipt并通过与seed1相同的逐视频严格复验：每臂4/4，声明/receipt/OpenCV实际帧数均121、640×480、receipt与逐帧重算black0、实际视频SHA逐行匹配。seed4 receipt SHA：action060=`65aab2e014b4ef5de48c1430c538290def9ede0df3dc03519db5af7e43b6ef75`；action075=`ff618ae3fd4c52ea1f26fa73c713c88e7effffb2879c7bc66d2f7d12162398b2`；static030=`5c9bb8d1880f7155be6cbefeef9b02e7c1cba5cdb312d2149717d28c9a18ac67`；static040=`d8d1b87e4102a365aafa5aad3068e2a27521700aa06cde02a44d4564769e57fc`；perc015=`69c34b3c4664dda0bd3e5dd9417b30222f1e206cfc032742967d33262b5449ac`；maskthr10=`819c50c15b1c02788d9ec6c3999b9beb0b5fbcd4c9f3392b1d7a7aba5f9d86d7`；lr3e7-r2=`29d4c8babd023756ecb9eeb620fc55d55f96b68c15fea53ff02621a97f4a3937`。
- 14份CPU-only名称remap已使用冻结工具SHA=`694ab4dbfd1280d0b0e7cbd4e13c9a8a4562e58df5d48d04d832434db17dad88`完成；只将`episode000000..3.mp4`映射为scorer所需`episode0..3.mp4`，源视频不转码、不修改，source receipt SHA被写入remap receipt。
- 新增最小generated9冷却队列脚本`baseline/scripts/run_native640_round4_generated9_queue.sh`，本地/远端SHA=`f807a0503cdd66ec2da28f1ae3b5cb3318624e21cb7b1af22de5d42fe3daa23c`；它只做双快照601秒等待、UUID/compute PID、remap receipt、runner SHA与独占flock复核，并依次执行同一候选seed1→seed4的已验证runner SHA=`a6bf9d96b3c3a8b64306c94847dab1389e4947e491b3129e4b72c4ed3068a965`。
- 七个隔离队列已启动：action060 PID=`75803`/GPU0；action075=`75804`/GPU1；static030=`75805`/GPU2；static040=`75806`/GPU3；perc015=`75807`/GPU4；maskthr10=`75808`/GPU5；lr3e7-r2=`75809`/GPU6。GPU7外部任务继续不触碰。冷却后任一UUID非空或漂移则仅该队列fail-closed，不抢卡。

### 2026-08-31 19:36–19:44 CST — Round5 gain-budget启动；VAE ceiling接续已部署

- 用户批准将晋级口径从“所有保护项近乎不降”改为收益预算：有效正收益必须大于全部回撤，最大单项回撤必须小于最大单项提升，最终仍以matched-GT cap后的corrected15净增为准。三项motion超过GT上限的收益不计入有效收益。按此口径，旧action-safe的Trajectory=`+0.10571180`无法抵消JEPA=`-0.10954574`，仍不晋级。
- Round5第一诊断不新增训练框架：直接复用已下载的Wan2.2 VAE与expanded clean50现有GT视频，对4个native `640×480` GT视频执行encode→decode，输出PSNR/SSIM及可继续送官方Image/Aesthetic scorer的视频，用于裁决主要画质瓶颈在VAE decoder还是FlowWAM后段生成。
- 快速脚本 `baseline/scripts/run_flowwam_vae_ceiling_fast.py` 本地/远端SHA=`18f7f91decc910b543a16eed62153689085ab919b0a7852fb5ce0f4ff9227e01`，本地与远端`py_compile`通过；远端隔离root=`runs/flowwam-full15-gap-20260904/round5/gain-budget-v1`。
- 19:36–19:43实时核验Round4七条队列均已从冷却壳转为真实generated9 base compute，GPU0–6各自独立运行，显存约1.49GiB且命令/PPID/候选root正确；GPU7仍只有外部`/data/fjy/worldarena` PID=`2659785`，未触碰。
- 一次性接续脚本 `run_round5_vae_after_round4.sh` SHA=`e6c4b939146c8b3a4e0bd1ac8de3c80bdbbdb8480321d4b63393bb31528bb49b`，远端PID=`99444`。它只等待精确Round4 GPU0队列PID=`75803`退出，再按物理UUID做两次相隔601秒的compute空闲核验，并启动VAE ceiling；任一核验失败即退出，不抢卡、不覆盖输出。
- 当前结论仍为`EVALUATING`：Round4尚无generated9 merge/corrected15结果，Round5尚未产生VAE ceiling收据，不得声称新模型提分。

### 2026-08-31 19:46–19:49 CST — 七路VLM全量运行；Round5自动化切换

- Round4七臂seed1的base phase全部形成终态receipt，当前均进入同一官方VLM judge的`0/4`首条推理阶段；GPU0–6真实compute PID分别为`100465/100189/100743/100188/100390/99964/100045`，每卡显存约`17.47GiB`，cmdline、候选root与物理UUID均匹配，无重复启动。
- seed4 generated9尚未开始；七个原队列会在seed1完整终态后自动继续seed4。当前不能从base中间结果判断corrected15，也不提前读取full15倒推seed。
- Round5 VAE ceiling接续PID=`99444`仍存活，因精确等待GPU0 Round4队列PID=`75803`而尚未占用GPU；receipt仍缺失符合预期。
- 10分钟自动任务已切换为`WorldArena Track1 Round5 gain-budget`：Round4先完成先分析，GPU0释放后自动执行VAE ceiling；后续只依据ceiling结果启动少量VAE/late-RGB/action模型级后训练，不再扩普通q/v、seed、block或loss-weight扫参。

### 2026-08-31 20:23–20:40 CST — VAE ceiling完成；七路full15 base并行

- VAE探针前两次均在CUDA前安全失败：第一次环境缺`easydict`，第二次因`wan/__init__.py`连带导入无关speech栈而缺`librosa`。未安装无关依赖；最小修复仅绕过包级副作用、直接加载`wan.modules.vae2_2`，脚本SHA=`81827665775ec951ffa4fddfb8d85b8d5fd41f24841ac2a0c3402daef2a4ddeb`。
- GPU0进程`211784`完成4个native 640×480、81帧GT视频的Wan2.2 VAE encode→decode。receipt=`round5/gain-budget-v1/vae-ceiling-native4-v1/vae-ceiling.complete.json`：mean PSNR=`36.61042550 dB`，mean SSIM=`0.97998412`，mean MAE约`1.174/255`，峰值allocated/reserved=`8.784/10.574 GiB`。这说明纯VAE重建损失存在但较小，Image/JEPA的大幅缺口不应主要归因于VAE解码器；后续优先检查生成主干/late-RGB表示保持，而不是先重训VAE。
- Round4七臂双seed generated9均完成冻结selector选择和selected-full15 staging。GPU1–6先启动action060/action075/static030/static040/perc015/maskthr10的base；VAE完成释放GPU0后，立即补启动lr3e7-r2 base。20:40快照GPU0–6七路均有真实本项目compute PID，利用率约75–100%、显存约4.73GiB；GPU7外部`/data/fjy/worldarena` PID=`2659785`未触碰。
- 当前MVP边界：等待各base先完成先接VLM/JEPA并聚合corrected15，不建立新的调度框架；尚无Round4 corrected15终态，不声称提分。

### 2026-08-31 21:25–21:48 CST — Round4七路corrected15闭环；低LR净分提高但JEPA仍失败

- 七路selected-full15 base于21:25全部完成；随即用最小尾部链SHA=`a7b9a9075cd1871aada5debccf304d7f14e1a2cca280bb89834aeecfb0adb6b2`在GPU0–6并行执行JEPA→VLM→aggregate。21:46七路四phase均闭环，无Traceback/OOM；随后复用Round3同一GT JSON、P0 baseline与motion-cap脚本计算corrected15。
- `as-action060` corrected15 Δ=`-0.00588917`，JEPA=`-0.08459440`，Trajectory=`-0.03540045`，CLOSE；evaluation SHA=`3c1497fed846d5900d495c7fba82a8acee26a2bc6b703e85d7abc6ca88b9da35`。
- `as-action075` Δ=`-0.00402837`，JEPA=`-0.08777335`，Image=`-0.00727697`，CLOSE；SHA=`e4d645f7fd91ecd84894088ea4fe2edf281af40c798f911533b26e8de66cd1d0`。
- `as-static030` Δ=`+0.00145904`、nonmotion12 Δ=`+0.00168510`，Trajectory=`+0.06959539`、Semantic=`+0.03344725`，但JEPA=`-0.08044937`且Image=`-0.00318020`，未过门；SHA=`afefa559b6985a661cfcb07c92652cd7bdc6ea211bc525eb48615a0a9d1d63f6`。
- `as-static040` Δ=`-0.00770469`，JEPA=`-0.09123471`、Trajectory=`-0.04057130`，CLOSE；SHA=`8b761734afed1ceb5badcbaae6d0af4ff6fc658ad89d2f0f4dcb5995339e97ab`。
- `as-perc015` Δ=`-0.00613342`，JEPA=`-0.10737517`、Image=`-0.00706814`，CLOSE；SHA=`9f14b4620f2a366de087cd1d23ecf2dc9189885fd60ec2ebd99667b0b63cb36a`。
- `as-maskthr10` Δ=`-0.00162091`，Trajectory=`+0.03233465`、Semantic=`+0.05346675`，但JEPA=`-0.09478304`、Image=`-0.00750785`，CLOSE；SHA=`e6f8e7a987a81428867315a141e59ae8d373b311073249b59f5ab278a0deebdb`。
- `as-lr3e7-r2`是唯一净分显著提高者：corrected15=`0.69961803`，Δ=`+0.00987031`，nonmotion12 Δ=`+0.01270911`，Trajectory=`+0.19361629`、Semantic=`+0.03295900`、Photometric=`+0.01316774`、Image=`+0.00074596`；但JEPA仍=`-0.08235624`，paired wins仅`2/4`，触发JEPA硬保护而不直接晋级。evaluation SHA=`2bdd9b09e13a2d642d1ae27b7b92b210b22b81610adbaf53d9b8351cf68a2f1e`。
- 结论：把LR降到`3e-7`确实能把整体收益/损失关系转正，并首次同时得到Image微升和corrected15近`+0.01`；但所有七臂JEPA仍下降`0.080–0.107`，说明仅调action/static/perceptual权重或LR不能解决表示漂移。下一MVP必须保留`lr3e7`的动作/轨迹增益，同时把视觉保持移到独立late-RGB/decoded residual机制；禁止继续同类q/v权重扫。

### 2026-08-31 21:50–22:00 CST — VAE官方base对照重启

- 为把VAE像素重建损失换算成官方base指标，首版`vae-source4/vae-recon4` staging复用了MP4软链接；官方`_flat_mp4_names`明确拒绝symlink，两路在CUDA前安全退出，未产生base receipt或GPU工作。
- 最小修复只把4条generated与4条GT视频物化为普通文件，不改评测器或新增调度系统。staging脚本远端SHA=`87a0f71735e5d8429c360a435f92b2c8c5b74f5356ada55ed93a7d0c9a9ee6d1`；新隔离root为`vae-source4-v2`和`vae-recon4-v2`，共16个regular MP4、零symlink。
- 21:59两路官方base scorer真实上卡：source GPU0主PID=`260075`/compute=`260713`，recon GPU1主PID=`260076`/compute=`260649`；显存各约4.73GiB、利用率约81–95%，日志无旧symlink错误，base receipt尚待完成。GPU7外部PID=`2659785`继续不触碰。
- 当前其余GPU2–6没有现成且合法的late-RGB/decoded-residual训练入口；不以普通q/v、重复seed或无门控重算填卡。两路base一完成就直接比较Image/Aesthetic/Photometric等差值，再决定是否值得实现单臂decoded-residual MVP。

### 2026-08-31 22:03–22:09 CST — VAE JEPA ceiling暴露主要表示损失

- 为减少串行等待，同一source/recon包的VLM分别在GPU2/3并行启动，JEPA分别在GPU4/5并行启动；两路JEPA均已闭环。source receipt SHA=`72a0009b81f03e604d4529b340880b4ca554002c4945be3f64be6037a87d5824`，score=`1.0000008345`；recon receipt SHA=`226a20c51e2fd6e97db968b75c6eb7e889b1709c6db070e175d8a4fc1b5a865d`，score=`0.9058752060`。
- 仅一次Wan2.2 VAE encode→decode就造成JEPA Δ=`-0.0941256285`，与Round4七路候选相对P0的JEPA回撤`-0.080~-0.107`处于同一量级。此前仅凭PSNR=`36.61dB`/SSIM=`.97998`判定VAE误差小，对官方JEPA并不成立；VAE/decoded-RGB表示保持现升级为第一优先模型瓶颈。
- source base标准runner在`candidate==GT`时触发官方Trajectory `1/0`边界错误而退出；这不是模型失败。已在GPU0用同一固定兼容环境missing-only跑除Trajectory外的10个base指标PID=`273839`。recon base、两路VLM继续运行；最终直接比较Image/Aesthetic/Photometric/Background/Subject等，不用无意义的source Trajectory阻塞。
- 下一模型MVP应直接验证decoder-only校准或轻量late-RGB residual后训练，目标是保留`lr3e7-r2`的Trajectory/总分收益，同时回收VAE引入的JEPA/Image损失；普通DiT q/v/LR扫参继续关闭。

### 2026-08-31 22:13–22:16 CST — VAE官方base逐项差值闭环

- recon完整base receipt SHA=`36c6c1f55d963479cdb9aa2ec99cb08618bbf44465a99d44c4e812ba8ea8caa0`。source因`candidate==GT`触发Trajectory除零，没有伪造base receipt；同一固定兼容环境已missing-only完成其余10个base指标，`generated_results.json` SHA=`4c36ddaf571117de962bf77fe096687476814a9f72e0dec2100e4e143fe3e095`；recon JSON SHA=`748599192163fc214384b11191edcbe66592eae8b0e285a2f5cdf827d04b1dce`。
- `recon-source`：Image=`-0.01688527`，Aesthetic=`-0.01352470`，Semantic=`-0.02526825`，JEPA=`-0.09412563`，Subject=`-0.00389163`，Flow=`-0.00865778`，Dynamic=`-0.00348117`；Background近零、Depth零变化。Photometric反而=`+0.03752925`，Motion Smoothness=`+0.00811915`；10项均值净变化=`-0.00260891`。
- 结论：Wan2.2 VAE的平滑确实改善Photometric/Motion，但以明显的JEPA/Image/Aesthetic/Semantic损失为代价。最快有价值的后训练不是继续DiT调权重，而是让decoder/late-RGB分支回收细节与语义，同时保留当前平滑收益；下一步比较decoder末层校准与独立小型RGB residual，先单卡n4/n8淘汰。

### 2026-08-31 22:28–22:40 CST — decoded-RGB residual MVP训练与首个JEPA因果结果

- 8个与dev4任务不重合的Official RoboTwin视频对已物化为81帧、640×480的VAE-recon→GT训练对；pairs receipt SHA=`066bf0271ddb6c03e9e013ccdbc668bf3e0f027e8ad1387851648f27076d2e29`，pairs JSONL SHA=`172ce186015b3e36c2ece037aa5ce89dc3018c584585a1ffbdd1c644c3122c6c`。训练器是独立3帧9ch→中心帧RGB的5层小CNN，输出限制为±`8/255`，不改FlowWAM主干。
- 三个1000-step候选在GPU0–2并行完成，均约134秒：`lr3e-4` final loss=`.00869482`、residual RMS=`.00022697`；`lr1e-3` loss=`.00822966`、RMS=`.00248719`、peak达到`.03137255`；`lr3e-3` loss=`.00869321`、RMS=`.00118033`。checkpoint SHA依次为`271e81b94171d1760df5f5da388aaf7cdb6dcb9d812c756d43b71050c9f40c00`、`3b3ee99897b608fa424700ecf81dbfa9213149585e0cec0cb60323299c46c0a5`、`9f9738b760acbbe971c7897de25f149f5bbd593940506d500557256a43bb694a`。
- 三候选与一个严格identity re-encode control在GPU0–3并行应用到同一dev4 VAE-recon输入；4路均81帧、24fps、640×480、4/4终态。为消除OpenCV二次编码混淆，所有因果比较只对同编码control做配对，不把其绝对分直接与原VAE-recon MP4比较。
- 四路JEPA并行完成：identity=`.88501757`；`lr3e-4=.88501757`（量化后等同identity）；`lr3e-3=.88574719`（Δ=`+.00072962`）；`lr1e-3=.88831270`（Δ=`+.00329512`）。首个有效方向是`lr1e-3`，但增幅仍小，尚未证明corrected15净增。
- 22:38后GPU4–6并行跑`lr1e-3/lr3e-3/identity` official base，GPU0–3并行跑四路VLM；GPU0–6均为本项目真实compute，GPU7外部PID=`2659785`未触碰。base/VLM终态后先做同编码配对增量，只有视觉项净增且Instruction/Interaction不恶化才进入full corrected15；不建设新调度框架。

### 2026-08-31 22:47–22:59 CST — decoded-RGB residual MVP完整15项裁决：CLOSE

- `identity/lr1e-3/lr3e-3`三路official base、VLM、JEPA及aggregate全部闭环，日志无Traceback/OOM；`lr3e-4`量化后与identity等同，未浪费GPU补跑base。三路aggregate receipt已生成。
- corrected15使用与这4个任务对应的expanded-clean50 matched-GT motion JSON（SHA=`ddf3a54a1d14c63bfa612df56a2b29981db895be783945add681154835ef57e0`），逐episode对Dynamic/Flow/Motion Smoothness取`min(candidate, matched-GT)`；同编码identity corrected15=`0.62925896`、nonmotion12=`0.68394140`。
- `lr1e-3`相对identity：corrected15 Δ=`-0.00661303`、nonmotion12 Δ=`-0.00863718`。虽JEPA=`+0.00329512`、Motion Smoothness=`+0.00397702`，但Image=`-0.02638446`、Instruction=`-0.05000000`、Trajectory=`-0.01309541`、Photometric=`-0.00523994`，明确CLOSE。
- `lr3e-3`相对identity：corrected15 Δ=`-0.00018259`、nonmotion12 Δ=`-0.00022896`；JEPA仅`+0.00072962`，Image=`+0.00001178`近零，同时Semantic=`-0.00354000`，明确CLOSE。
- 结论：小型decoded-RGB CNN能回收少量JEPA，但会重绘像素并破坏Image/Instruction，且净分为负；不继续扫CNN学习率/seed。下一MVP改为“冻结action teacher + 表示锚定的late-RGB delta”：动作/contact区域保持teacher，static/appearance token只优化GT视觉与JEPA/Image/Aesthetic，Flow/action主干冻结；另保留flow-warp+occlusion-gated RGB residual作为低风险对照。

### 2026-08-31 23:00–23:12 CST — MELR最小因果实验：JEPA硬门失败，pixel residual路线关闭

- 只实现一个Motion-and-Edge Locked Residual候选：保持原9ch/width32/±`8/255`，运动和强边缘周围7px的残差严格为零。测试先因实现缺失RED，随后4项远端PASS；trainer SHA=`632bae5fe43842ae837f0bf4eee01f37e0c47351ece502bc7e908d77a69afcb5`，apply SHA=`19f5323ca29a412798486f2174d9007144b49e837598698c6d3c6e29de94f5d1`，test SHA=`daf33899f96f40ce3699eee36ef9f47d870fe1c71185902dfbd374c18efe0cd5`。
- 唯一训练`lr=1e-3/1000-step/seed0`在GPU0用`134.725s`完成；checkpoint SHA=`b0e6f10fde68e156c51a8a4aad137ea89dcac2fe63e65e9577e1601e14e8388c`，safe fraction=`.6309565`，locked residual peak=`.00020296`。同dev4应用仅`4.07s`，应用后实际locked residual RMS=`7.33e-6`。
- JEPA先完成：MELR=`.8785510063`，同编码identity=`.8850175738`，Δ=`-.0064665675`，直接触发`JEPA Δ<=0`硬门。无需等待full15，MELR及整条decoded-RGB pixel residual路线CLOSE，不扫mask半径/LR/seed。
- base/VLM虽已与JEPA并行启动，但硬门一出即精确核验本项目PGID=`928332/928646`及子进程user/cmdline/cwd，发送TERM释放GPU1/2；GPU0–6确认无compute PID，GPU7外部`/data/fjy/worldarena` PID=`2659785`始终未触碰。
- 下一唯一MVP转为冻结DINO feature teacher的静态区域表示锚定，仍沿用同一小CNN与pairs，仅改变监督信号和写入区域；若该单臂也不能同时做到JEPA、Image非负，则停止所有后解码器，转真正VAE decoder-tail或生成主干late-RGB feature adapter。

### 2026-08-31 23:24 — DINO feature-anchor单臂已真实上卡

- 唯一候选固定为`200 steps/lr1e-3/seed0/width32/residual±8/255`，不做LR或seed扫描；冻结远端Facebook DINO ViT-B/16，只训练小残差CNN。
- 为避免动作区泄漏，DINO token gate由area pooling改为conservative adaptive-max：一个token范围内任一像素为motion/edge unsafe，则整个token不参与feature anchor。新增hard-token gate与prediction-gradient/teacher-freeze回归测试，远端`6 passed`。
- trainer SHA=`8afd9fd1783fd70781dd8d290b15092bc742dbfc7915c9f4c6e934edbbdaae16`；test SHA=`7b3a8fa17d0daa9e7a1a5b9774b89078cf1e8b00e22b399a9d63b21781cd48bf`。
- 物理GPU0/UUID=`GPU-66b5e202-9179-0f61-c0dd-34d5599ffde2`已启动，PID=PGID=`937901`，启动后显存=`1756 MiB`、util=`35%`；GPU7外部`/data/fjy/worldarena` PID=`2659785`未触碰。
- checkpoint出现后立即apply同一dev4，先跑JEPA硬门；若不超过同编码identity=`.8850175738`则直接CLOSE，过门才补base/VLM与matched-GT corrected15。

### 2026-08-31 23:25–23:29 — DINO feature-anchor单臂完成：JEPA硬门失败

- 训练仅耗时`30.98s`，receipt SHA=`ace39f319ff551caef1a12b9a3ae946028eb4608bf8f600cbfcd1eafce0b023e`；checkpoint SHA=`11a7f16487bdcdd835b3498c9a4616113150c04026fc1afcb2b631175091c9d6`。final static pixel fraction=`.492168`，feature loss=`.718107`。
- 同一dev4应用耗时`3.99s`，4/4视频均81帧、640×480；residual RMS=`9.52e-5`、peak=`1.49e-4`。
- JEPA receipt SHA=`1d5b86bdc56589194f293bf81f2617500d590b87390653f325811677ca7ab9c6`；DINO-arm=`.8785510063`，同编码identity=`.8850175738`，Δ=`-.0064665675`，与MELR结果完全相同，触发硬门。
- 因JEPA失败，不启动base/VLM，不浪费GPU补full15；DINO后解码器路线CLOSE，不扫feature weight/LR/seed。
- 统一结论：普通CNN、MELR、DINO feature-anchor三条post-decode residual路线均无法在保护JEPA/Image的同时获得净分；下一MVP必须前移到真正VAE decoder-tail或FlowWAM late-RGB feature adapter，冻结action/flow主干，后解码像素处理永久关闭。

### 2026-08-31 23:40–23:51 — VAE decoder-tail MVP真实上卡；OOM后缩为head-only

- 真实Wan2.2结构确认：`decoder.conv1 → middle → upsamples[0..3] → head`；原计划只解冻`upsamples[3]+head`约`12,604,172`参数，其余encoder/conv1/conv2/early decoder全冻结。训练输入固定连续因果clip，禁止均匀抽帧。
- 第一版连续9帧、`upsamples[3]+head`在GPU0第一步真实OOM：进程PID/PGID=`1040296`已退出，无checkpoint/receipt；峰值约`23.36GiB`，外部GPU7未触碰。该失败说明高分辨率up3反传不适合24GiB卡的最快MVP。
- 未建设offload框架；直接降为最小可证伪版本：仅解冻`decoder.head.*`约`83,212`参数、连续5帧、200 steps、lr=`5e-6`。本地/远端3 tests PASS；trainer SHA=`3691f602af19c263cb55f5ea215b1212e5776a435a6c57e9b474793693f01454`，test SHA=`8dc80f3c7c81ce345c2abbe760cde938f5d0484fcb4c3395f8922edd2c81be61`。
- head-only已在物理GPU0/UUID=`GPU-66b5e202-9179-0f61-c0dd-34d5599ffde2`真实运行，PID/PGID=`1041917`；23:51快照显存=`9512MiB`、util=`100%`。GPU7外部PID=`2659785`保持不触碰。
- 训练完成后不能对MP4二次encode做因果评估；必须给已有Official Stage1 runner增加最小`--vae-tail-checkpoint`注入，在`pipe.vae.model.decoder.head`严格加载后用同prompt/seed重跑dev4，随后JEPA先行。

### 2026-09-01 08:15–08:21 CST — 四路MVP收敛；Refiner单视频真实上卡

- 后续严格拆成四路：A=`SeedVR2 Refiner`只攻Image/Aesthetic且不训练主模型；B=`Instruction`只做文本CFG与late text-cross小LoRA；C=`Interaction`只做SAM3 object-ROI late LoRA；D=冠军teacher anchor，JEPA/Trajectory只作硬门禁。禁止把六指标混成一锅loss，禁止恢复普通q/v、ORB或全量DiT微调。
- B的零训练Text CFG已完整闭环：scale `1.5/2.0/2.5` corrected15分别为`.67189905/.68537454/.68194545`，全部低于同四条冠军`.69301885`；失败样本的Instruction在三档均保持`.4`。因此Text CFG路线CLOSE，不继续扫scale。
- 旧Instruction late-cross 5-step corrected15=`.68159252`；冠军self+cross合并 corrected15=`.67247608`，Instruction/Interaction无增益且Trajectory=`-.259895`、Photometric=`-.177277`。根因不是selector而是cross更新本身有害，旧cross/checkpoint合并路线CLOSE。
- B最小修复已完成但尚未训练：teacher改为保留冠军120个self q/v LoRA，仅临时关闭24个可训练late-cross q/v，避免旧实现把teacher退回裸Official parent。定向测试`7 passed`。正式训练仍等待非dev、原始instruction可追溯的合法8例manifest，禁止用same-four或评分产物伪造文本标签。
- A已新增SeedVR2 `17-frame window/4-frame overlap` MVP：同一runner只加载一次，121帧拆9窗，线性融合，alpha后恢复首帧数组；本地与远端定向测试均`5 passed`，脚本SHA=`5bfaffd539366019767764fa2cd0a0655515d3bba6e8f23f43e24ca15b2ba991`。
- 冻结冠军四视频选择为seed `4/1/4/4`，输入SHA分别为`07a592e0.../26ff88e4.../6f06ee3a.../3ac45e63...`。08:20单视频alpha=.5 smoke已在物理GPU6/UUID=`GPU-b6761c44-aff9-92ed-40e9-fa5731af5cbf`启动，PID=PGID=`1785372`；08:21显存=`7864MiB`，已完成DiT/VAE加载并进入首窗推理。GPU7外部`/data/fjy/worldarena` PID=`2659785`未触碰。

### 2026-09-01 08:31–09:26 CST — Refiner四条生成闭环；A/B并行MVP启动

- SeedVR2首版在3B DiT加载时OOM，根因是checkpoint默认保留FP32。最小修复只在CPU侧将DiT严格转为BF16并核验所有浮点参数/缓冲区dtype；远端模块SHA=`243c0b3c2d72bf9767957df8bbf4538facd75033d730184ec31fa60b9889fb29`，聚焦测试`2 passed`。没有引入调度或训练框架。
- 单视频smoke随后闭环：121帧、9个17帧窗口、4帧overlap、640×480、首帧数组恢复、零padding；receipt SHA=`c44d900bc75db110fb718c89e2bf66c418d0e7fd9dce258a650c60a9ca616f75`。四条冠军视频完整refine于09:10完成，receipt=`seedvr2-champion4-alpha05.receipt.json`。
- 同一次cached refined pass离线生成alpha `.3/.5/.7`共12条submission视频；reblend receipt SHA=`748f579de99f527658db02da3d32eba26d001356a1f9e693ab520b66cdae2925`。该过程不重复调用SeedVR2、不训练主模型。
- 为消除旧81帧包与新121帧包的混淆，先用同一4/1/4/4输入完成alpha=0严格对照：corrected15=`.6820148312`，raw15=`.6842398931`，corrected receipt SHA=`c263b675f2ea8b31883d7946012cc0eb165f6ff3a8906e702f5ee7803ea93269`。后续只比较`.3/.5/.7 - alpha0`，不与旧`.69961803`直接比较。
- 09:26物理GPU0–2分别启动alpha=.3的base/VLM/JEPA，GPU3–5分别启动alpha=.5的base/VLM/JEPA；GPU6启动Instruction一步smoke。GPU7仍只有外部`/data/fjy/worldarena` PID=`2659785`，未触碰。alpha=.7在第一张评分卡释放后立即接续，无全局barrier。
- B trainer已固定为冠军self teacher + late text-cross student：冠军self LoRA保留，teacher仅关闭待训练cross adapter；部署SHA=`1c7dc8683cea0ce4dad004bb858d251d684acf434df371317062082c89c03fc1`，聚焦测试`1 passed`。数据仅为native640 clean50 ep0–7机制smoke，receipt SHA=`8eece12aac0504324e11663b580be31de31b6d1eafa504df5f350eb33fa94a39`，`promotion_evidence=false`；不得以此声称泛化提分。
- C旧四臂复审确认没有任何Instruction/Interaction增益：safe虽corrected15仅`+.00042094`，但Image/Aesthetic/Trajectory/Photometric均回撤；balanced/strong/lowLR净分为负。还发现旧anchor实际关闭全部LoRA，teacher是裸Official parent而非冠军。下一唯一C候选固定为4-step、lr=`3e-7`、object=`4`、roi=`.5`、anchor=`1`、late self q/v，并先做真冠军teacher最小修复；禁止重复旧臂。

### 2026-09-01 09:27–09:45 CST — Refiner主扫裁决；Instruction/Interaction进入真实MVP

- A四路严格同编码、同121帧、同4/1/4/4对照完成。alpha0 corrected15=`.68201483`；alpha `.3/.5/.7`分别=`.67095976/.65659302/.65267286`，delta=`-.01105508/-.02542182/-.02934198`，全部不能晋级。
- SeedVR2确实提高Image：alpha `.3/.5/.7`分别`+.01633740/+.03625184/+.05523865`；但Aesthetic分别`-.00616559/-.00578194/-.00750206`，JEPA分别`-.03167385/-.05466479/-.06905204`，Trajectory分别`-.03338106/-.10395555/-.24390513`，且Instruction/Interaction均出现回撤。结论是“可增强图像锐度，但当前全帧blend破坏语义与时序”，主扫CLOSE。
- 只补一个最低成本alpha=`.1`安全边界试验；直接复用cached refined帧，未再次运行SeedVR2。GPU0/1/2分别执行base/VLM/JEPA；该点若仍不能同时满足Image正增与JEPA/Trajectory门禁，则A整条全帧Refiner路线关闭，不继续扫alpha。
- B冠军teacher一步smoke成功：1 step=`88.29s`，峰值allocated/reserved=`14.63/15.11 GiB`，120个冠军self张量加载、24个late cross张量可训练，receipt SHA=`729d8258245c92d1fee91629ee124b91f3f9e8e21e20ea1fdd125d7b9df49024`。clean50被合同正确限制为smoke；真实randomized500 click_bell ep0–7已物化并回读，manifest SHA=`1899641457a38ec6108fb543af17ee380c848c982bce17d49654991cf74452c6`，但其原始分辨率为320×240，后续仅作bicubic机制诊断、不冒充native640晋级证据。
- C真冠军teacher 4-step已完成：checkpoint为完整冠军self-q/v 60对，只更新late 12对；elapsed=`173.65s`，peak reserved=`16.49 GiB`，anchor从step2开始非零，full checkpoint SHA=`e0c59816a68d6751971260218c1d66869d5a06aa00aeb7c61adfa533c2b9debb`。seed1/seed4四视频Stage1已分别在GPU4/GPU3并行生成；生成完成后走冻结generated9选择和corrected15，不把训练loss当提分证据。

### 2026-09-01 09:46–10:05 CST — 四路MVP合同锁定；B真实数据首次启动暴露单一分辨率阻塞

- 自动任务已同步为四路MVP：A仅收尾alpha=.1；B仅冠军teacher+late text-cross；C仅object-ROI+late self；D为B/C默认teacher anchor。禁止新增全量DiT、V-JEPA/Trajectory目标或调度框架。
- B数据来源最小修复已部署：`aloha-agilex_randomized_500`只接受`YixiangChen/FlowWAM_RoboTwin@506c4e014f7dbd291e7d7683c79fc685dd3e2714`，clean50仍只接受原WorldArena revision，其他来源全部fail-close。companion SHA=`252b65b5938680e553e0e184799d1b861618272f2528e2acc6bcfbeb4df710e4`；focused test=`1 passed`。
- B 25-step真实训练在GPU5启动后于CUDA训练前退出；不是OOM、不是trainer/teacher错误，而是released randomized500 JPEG原生分辨率为`320×240`，当前子类在基类已有bicubic-to-640能力之前错误强制`native 640`并耗尽8次fallback。失败PID=`1887840`已退出、无checkpoint/terminal receipt、GPU5释放。
- 下一修复严格限定为：只允许固定randomized500来源读取真实`320×240` JPEG并沿现有loader resize到`640×480`训练形状；clean50与其他variant继续维持原native640合同。修复后复用相同25-step命令，不新增trainer功能。
- A alpha=.1 JEPA已完成：`.7722757459`，相对alpha0 `.7952148318`为`-.0229390860`，已明显失败JEPA硬门；base/VLM仍在执行，完成后仍补aggregate/corrected15以形成完整关闭证据。
- C seed1/seed4 Stage1仍在GPU4/GPU3运行；后续现成链已定位为`remap → generated9 seed1/4 → frozen selector → selected full15 → base/VLM/JEPA → aggregate → GT-cap corrected15`，无需写新框架。

### 2026-09-01 10:06–10:25 CST — A路线关闭；B修复重启；C双seed进入generated9

- A alpha=.1完整raw15 aggregate已闭环；用与alpha0已存corrected15逐单元完全复现的同一GT cap公式计算：corrected15=`.6822753829`，相对alpha0=`+.0002605517`。Image=`+.00322322`，但Aesthetic=`-.00824557`、JEPA=`-.02293909`、Photometric=`-.02904395`，明显失败四路计划的目标与硬门禁。因此全帧SeedVR2 Refiner路线正式CLOSE；不再扫更低alpha、窗口或seed。
- B首次真实训练暴露的唯一阻塞已最小修复：randomized500固定来源只允许native `320×240` JPEG；clean50仍只允许`640×480`；未知variant/交叉尺寸fail-close；下游既有`size=(640,480)` bicubic路径保持不变。live trainer SHA从`1c7dc868...`变为`5763e482582f89e0d750c779ad105a35a87e66ef18df00f3a86f570068ecf875`；source-contract与resolution-contract两项focused tests均PASS。
- B v3越过native-size门后在第一forward暴露RGB/flow latent空间不一致：RGB仍为native320而flow已按既有loader变为640，`30×40`与`15×20`赋值失败。最小修复仅在合同校验后把randomized500 RGB PIL用bicubic显式归一到训练形状`640×480`；trainer SHA=`7c0b9afe01f4baeaaf32ed41c58273480c68dc087bda778c400dc4d6c1535928`，正向normalize测试、py_compile与两项合同测试PASS。
- B 25-step真实训练v4使用GPU5、PID=`1911442`，已越过上述第一forward并保持运行；仍是冠军self teacher、仅late blocks24–29 text cross q/v LoRA、rank8、lr=`1e-6`、25 steps。为并行覆盖唯一有价值的更新幅度对照，GPU6另启动同seed、lr=`5e-7`单变量温和臂，PID=`1915202`；不再增加seed/block/架构臂。
- C seed1/seed4 Stage1均4/4终态：121声明/解码/实际帧、640×480、black0、视频SHA全部核验；receipt SHA分别=`d58479c5f06ac398ccb1bdd5645d7a421df2f15afbc3ad590c4cdb53861de9dd`与`c28484ea633254feb3b4466d06bba7f5010690057eb8ee0af4af9412b5d4e70b`。不等待全局barrier，seed1 generated9已在GPU4 PID=`1893498`启动，seed4 generated9已在GPU3 PID=`1901705`启动。

### 2026-09-01 10:26–10:29 CST — C冻结选择完成；full15三阶段异步启动

- C seed1/seed4的generated9均已完整闭环。seed1 base/VLM receipt SHA分别=`1c620d2fbbe000bc1d83e33e6aacb431699f5993912e9b115d733328249a1ae5`/`c5208445cd62d8047665805d77e5748d23f8d377984fe6b07acef8ce8c1df355`，CSV SHA=`a7d1d0edea114d15df15a1b63a3d5944cc85f5daee7a031d59c297d562c99eb6`；seed4对应SHA=`54d3123c41634b0cb6db2503bf7ad3ac3e3655010a56ddb2c5d6935db7940d4c`/`14a13c4f4fc40ecc819b9c3539f62073a049d6d9f41dc9640917a68a27ffd26c`/`2e49693f40d44d5181d6dba281fb6149aaf5907d1cbdf1d82f1ff5cc89a751b6`。
- 冻结P0 generated9 selector精确选出seed1=`1`、seed4=`3`；selection receipt SHA=`477362d5f3dd95e63035ead9b37a626aca0c9aa9c7d8cbe1ef72e841fee531c9`，model/policy SHA仍为冻结值`4e97e87...`/`aa551c7b...`，明确未读corrected15或隐藏GT。
- 首次staging在第1条后因旧instruction JSON仅含`seen/unseen`而fail-close。没有删除合法partial，也没有新建框架；只物化4条`seen[0] → instruction`的12文件normalized input，receipt SHA=`3c27d9302a1505569f094e9b15536a1eaa42e9ac781d96c49a5734c85b4dd19f`，随后同一stager missing-only续跑完成。package receipt SHA=`5f0a1f914ac2f8f5ebdcb8ee29df1aac6a1074785af75b2ee419337758ae2903`，4条均为81帧官方dev包。
- 10:29在精确GPU UUID/compute PID与`/proc`核验后异步启动C selected-full15：base GPU0 PID/PGID=`1933650`，VLM GPU1=`1937024`，JEPA GPU2=`1940080`；三者均为项目路径、互不写同一phase输出。B两条真实Instruction训练继续在GPU5/6运行，GPU7外部`/data/fjy/worldarena` PID=`2659785`保持不触碰。
- C JEPA先完成，score=`.7619388103`，results SHA=`f442e410d16bdc6241d790efa59c22066243cfa358f086e0542f2e51bca2ebec`、receipt SHA=`47397ebc651cc8edae07f0f9e9faa508395d8b9e0e337f5ee41e96982f77722e`。该值已明显不满足JEPA硬门，说明object-ROI late-self候选即便有Interaction收益也不能晋级；仍让已启动的4条base/VLM完成，以取得Interaction及其他15项的因果诊断，不再扩seed/step。

### 2026-09-01 10:30–10:52 CST — C严格配对关闭；B双学习率训练完成并四卡生成

- C selected-full15 的base/VLM/JEPA与aggregate全部闭环；aggregate receipt SHA=`8680cc15504d616f0104adbd2268203161bfd33d7d9180c2e29a5f9620cfb3d8`，CSV SHA=`abf00a0afe3832acdd5e30b7a51168a216631218411eb5e76d42b9b3a22bd564`。用同dev4、同latest-GT cap与严格冠军配对重算：corrected15=`.66511012`，冠军=`.69301885`，delta=`-.02790874`；nonmotion12 delta=`-.04548145`。
- C 的Interaction=`.800000`、Instruction=`.850000`，均相对冠军零增益；Trajectory delta=`-.34365668`、Photometric=`-.19535952`、Image=`-.01063933`、Aesthetic=`-.00842644`，因此唯一C候选明确CLOSE，不扩step/seed。纠正此前绝对值误读：同dev4配对JEPA实际为`+.00499374`，C失败主因是Trajectory/Photometric且Interaction未提升，不是JEPA。
- B两条25-step冠军teacher+late text-cross LoRA训练均完成。lr=`1e-6` checkpoint SHA=`62815f8d34553f00351b08b1ef5e89063c181d9aa88bdfcf46b86330b384928c`、receipt SHA=`2b43ba8f35af051c281b3f481401a42cc15becef5c236b23a3468c65f31ec675`；lr=`5e-7` checkpoint SHA=`36c13e559b265eae9dd3d88756a9d51c1f9cdaabd43a76313f1b850e1d28ce59`，均只更新blocks24–29 cross-attn q/v并加载冻结冠军120个self LoRA tensors。
- 为保持冠军self能力，分别把冠军self与两条cross适配器一次性合并成all-q/v推理checkpoint：lr=`1e-6` SHA=`7cce263f84d1742465166fa6632f3724e01a53bbeadc02ba8f342628130aa146`，lr=`5e-7` SHA=`2d69483484a6e69adc7685d73b9407c56b4b1b9e9a78909905daac346e716fa7`；每份240 tensors、120 loaded pairs，输入key无交集。
- 10:51四个dev4 Stage1任务异步启动：lr=`1e-6` seed1 GPU0 PID=`1961464`、seed4 GPU2 PID=`1961538`；lr=`5e-7` seed1 GPU3 PID=`1961612`、seed4 GPU4 PID=`1961686`。均复用同一官方dev4 input/robot-only、Official parent、121帧640x480与冻结runner SHA=`ba8ee8d36370eb1eb4f7b781ca8d106862f8662fec86911d7105ef302a7f45b1`；GPU7外部PID=`2659785`未触碰。生成完成即走generated9冻结选择与corrected15，两个学习率先完成者先裁决。

### 2026-09-01 11:04 CST — B四路Stage1闭环并接续generated9

- 两个学习率、两个seed共四路Stage1均4/4完成并产生终态receipt；16个视频全部121声明/解码帧、640×480、black0、SHA有效。lr=`1e-6` seed1/4 source receipt SHA分别=`2e6755c4e26f3118e074621c6db646a68af0aa763339a4db844d751ef0b94591`/`8275dc62ff63c2ac43fbbebadef58f6de07daacd66eac71cae0d7a693655fdcc`；lr=`5e-7`分别=`3887302ca66a376998d8e9be2fc0f0787d8155c888eb4eae6959a7fe0845a3df`/`7f0b850e1a19945fc7d9567469e8bec90f5ebc12e520abd8f982c5fe6a6b29c3`。
- 现成remap脚本SHA=`694ab4dbfd1280d0b0e7cbd4e13c9a8a4562e58df5d48d04d832434db17dad88`对四路逐一严格回读并物化generated9输入；没有新增框架或重复生成。
- 11:04四个generated9双seed任务异步接续：lr=`1e-6` seed1 GPU0 launcher PID=`1979064`、seed4 GPU2=`1979068`；lr=`5e-7` seed1 GPU3=`1979073`、seed4 GPU4=`1979084`。四卡已出现真实compute子PID；脚本SHA=`a6bf9d96b3c3a8b64306c94847dab1389e4947e491b3129e4b72c4ed3068a965`。完成后立即CPU冻结selector，并把两个arm的full15三phase跨空卡并行启动。

### 2026-09-01 11:23–11:25 CST — B双arm冻结选择完成；六路full15并行启动

- 两个arm的四份generated9均package/base/VLM/merge闭环，生成侧CSV全部出现。冻结selector严格使用model SHA=`4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa`、policy SHA=`aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`，`corrected15_used_for_selection=false`、row_count=`4`。
- lr=`1e-6`选择计数seed1=`0`、seed4=`4`；lr=`5e-7`选择计数seed1=`2`、seed4=`2`。两路selected package均用已验证normalized dev4 input成功闭环，无隐藏GT参与选择。
- 11:24两个arm的base/VLM/JEPA共六个full15阶段异步上卡：lr=`1e-6`分别GPU0/1/2，launcher PID=`2006327/2006401/2006508`；lr=`5e-7`分别GPU3/4/5，PID=`2006551/2006641/2006764`。GPU7外部任务保持不触碰。三phase收据齐后立即CPU aggregate并按latest-GT cap复算，与strict champion `.6930188548`做配对裁决。

### 2026-09-01 11:43–11:46 CST — B完整15项裁决CLOSE；四路MVP首轮全部收口

- 两个Instruction arm的base/VLM/JEPA与aggregate全部闭环。lr=`1e-6` aggregate receipt SHA=`a81bb075e4a583115d92317660abf6ffa6eb9d1e6d3ef4ab0fe783ddb008f929`、CSV SHA=`7389c1001883d7a6749d35cf6c0f430318055de7516e7591e17aba3bbff20f14`；lr=`5e-7` receipt SHA=`cb171d2c61d29e26e3ef074365cc0eb003b30c545decd551776bf5d565db57b1`、CSV SHA=`18694ab821ca258d34932ef0e5f31320bdc09857abecaec044f6bf863506b001`。
- 使用latest-GT SHA=`79c73e40198125f52d4c010a524892f8483459cde027c50d55fe0da411320548`对Dynamic/Flow/Motion Smoothness逐episode取min，并相对strict champion corrected CSV SHA=`ca78e88aaba3414217288429a1c804a224c28ab0984e4261d52acca075b4b2e5`比较。
- lr=`1e-6` corrected15=`.67492050`，相对冠军`-.01809836`；nonmotion12 delta=`-.03343093`，paired wins=`2/4`。Instruction/Interaction均`0`增益；JEPA=`+.02110058`、Semantic=`+.01306175`，但Trajectory=`-.24342782`、Photometric=`-.17846094`、Image=`-.01420724`、Aesthetic=`-.00773919`，明确CLOSE。
- lr=`5e-7` corrected15=`.67308995`，相对冠军`-.01992890`；nonmotion12 delta=`-.03584516`，paired wins=`1/4`。Instruction/Interaction仍零增益；JEPA=`+.00226289`，但Trajectory=`-.22730944`、Photometric=`-.18132302`、Image=`-.00888912`、Semantic=`-.01330550`，明确CLOSE。
- 结论：冠军teacher成功守住JEPA，却没有把Instruction提高；late text-cross LoRA仍以Trajectory/Photometric显著退化换取运动侧变化。按预注册MVP合同不再扫更多LR、seed或blocks。A全帧Refiner、B Instruction、C object-ROI均完成并关闭，D anchor已随B/C验证；昨晚四路首轮计划全部收口，稳定冠军/P0保持不变。

### 2026-09-01 12:00–12:24 CST — 战略切换到 episode routing；oracle、路由MVP与best-of-4启动

- 按新决策停止继续修改生成模型，关闭新LoRA、ORB、ROI、Refiner、decoder和loss sweep；只允许从多个**完整episode视频**中选择一个，禁止frame-level拼接。
- 对当前冠军与全部已有互补候选做严格dev4 corrected15 oracle。当前冠军=`.6930188548`；全部候选oracle=`.7020720323`，headroom=`+.0090531775`。4条分别由冠军1条、`as-lr3e7-r2` 2条、`ORFC neg-lift` 1条胜出。`as-lr3e7-r2`的leave-one-out贡献=`+.0006973505`，ORFC仅=`+.0001377895`，其余候选为0；因此ORFC不扩量，model-MoE只保留冠军+AS作为后续n20验证。receipt=`runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/oracle/all-existing-dev4-oracle.complete.json`。
- 对P0 fresh40的seed1/seed4做严格corrected15 best-of-2：seed1=`.6700214358`，seed4=`.6654975262`，oracle=`.6774110479`，相对seed1=`+.0073896121`，oracle选择计数22/18。冻结generated9 selector仅达到`.6710700069`（`+.0010485711`），只吃到约14.2%的理论空间。receipt=`.../oracle/p0-fresh40-seed1-seed4-oracle.complete.json`。
- 一次性CPU router MVP只用train160的task先验与冻结generated9模型，在旧holdout40选择融合权重，然后只在零episode重叠的fresh40评测。结果：task prior单独=`.6688008153`；冻结generated9=`.6710700069`；holdout-tuned blend=`.6715479150`，比冻结selector只多`+.0004779080`，仍只吃到约20.7%的oracle headroom，不能部署。receipt SHA=`b3907354fd0cdbed99d151aee2f895c7792499c7cdea6499ade5a213128578c9`，路径=`.../router/seed-router-mvp.complete.json`；一次性脚本SHA=`ec674dfa4bf19e0b13eea4dc79bbf91c485482b4e898ad4aaac7ebbde0f5114c`。
- 因best-of-2仍有明确sampling headroom，立即转向P0 seed2/seed3，而非继续训练模型。12:24前已在GPU0–5启动seed2六个fresh40 shard（共40条），GPU6启动seed3首个7条shard；启动器SHA=`ec36cea27b881849fdceba4023cdcd882782d3bbd2f823ff8a01d11660df51cd`。GPU7外部`/data/fjy/worldarena` PID=`2659785`保持不触碰。seed2先完成的卡立即接续seed3剩余shard，无全局barrier；两seed完整后先跑generated9，再按headroom决定是否投入full15。
## 2026-09-01 14:42 CST — Routing MVP execution update

- Closed generator-weight experiments and switched to episode-level inference-time routing.
- P0 seed2 fresh40 generation completed: six immutable shards (`7/7/7/7/6/6`), 40 videos total. Every Stage-1 receipt reports 121 declared/decoded frames, 640x480, zero black frames, and the frozen Official FlowWAM parent SHA.
- P0 seed3 fresh40 generation is asynchronous: the first 7-row shard and both 6-row tail shards are complete; the remaining three shards continue on physical GPUs 0–2. No existing shard was regenerated.
- Seed2 generated9 scoring uses a 40-row metapackage at `inference-ensemble-mvp-20260901/scoring/seed2/merged40`, assembled from the already resampled 81-frame shard packages. Package receipt SHA256: `01c72d2ed3e1b94ac1ac63522a43826a083e6acde80a0cd6496bca57d4afb356`; hidden GT is not used and GT paths are generated-video self-aliases.
- Generated-only base and resumable VLM phases were launched independently on physical GPUs 5 and 6. The frozen generated9 metrics are the only routing features; full15 remains evaluation-only.
- Disk reserve blocker was removed by deleting two explicitly obsolete and reproducible assets: the old Wan2.2-TI2V-5B model directory and the old `FlowWAM_RoboTwin_clean50` dataset. Stable P0, current extracted fresh40 data, scoring assets, and all historical submission revisions were preserved. Available disk increased to about 173.68 GB; no further cleanup is currently required.
## 2026-09-01 16:40 CST — Seed2 result and seven-GPU seed3 scoring

- Seed3 generation reached 40/40 with six immutable Stage-1 receipts (`7/7/7/7/6/6`). No duplicate generation was launched.
- Seed2 generated9 scoring completed. Task-aware CSV SHA256: `68a127695d1879efe967c720d97a90e6a0d94668bea4f568a896c79361d4e2b0`; receipt SHA256: `3eb1f52b98e65a336d7f89070d25c54540fbf25f5191d2c91e04742d881d7f33`.
- Generated9 means: seed1 `0.5324770657`, seed4 `0.5344968153`, seed2 `0.5303072402`. Although seed2 is weaker on average, it beats the better of seed1/seed4 on 9/40 episodes. Mean9 oracle rises from `0.5400395288` (seed1/4) to `0.5410087941` (seed1/4/2), a marginal `+0.0009692653`.
- Decision: retain seed2 only as a nine-episode diversity candidate. Evaluate those generated9-new-winner rows with true-GT full15; do not spend full15 budget on all 40 seed2 rows.
- Seed3 scoring was parallelized without a global barrier: merged40 generated-only base on physical GPU6, and six resumable VLM shards on physical GPUs0–5. GPU7 remains untouched because it belongs to the external Track2 process.
- Next gate: merge seed3 by canonical task+episode identity, measure its incremental Best-of-N headroom, and full15-score only new generated9 winners.

### 2026-09-01 16:47–17:10 CST — Seed3 VLM闭环；seed2新增赢家进入missing-only full15

- Seed3六个VLM shard均已完成，steps `1106–1111`行数分别为`7/7/7/7/6/6`，合计40条；VLM receipt SHA分别为`68f7c612a2af024fdbda85479bd657414725a4513030ce7f169a8c4fa755a956`、`9515a0daf73f5e69ecdae5f6ef54d57c3e0f3e1074563b05ff15f71169d1de69`、`3df2404f8ab38461b13de0657481b70af0d4483fca050cd00709c8d9ef6dd6d2`、`ad23c9ab54fff7cef9fb546fb99ceab1232dec140a45c3672607f357bdd3d9a7`、`c0379c6b90fe7294d034be072c21d59211b5afb5f7cf163254143b7a17b97dfa`、`9e86d4d9e8e3e3e955ca9ced9af5007974082b3091a32f5448ab9aa03e4d4243`。merged40 base仍在GPU6运行，尚无base receipt，因此Seed3完整generated9均值、新赢家数量与四seed增量headroom仍为`UNKNOWN`。
- Seed2 generated9新增的9个赢家已组成真实GT full15包。首次symlink包被官方validator拒绝后保留为`package.rejected-symlink`；当前包改为18个regular hardlink、无symlink，package receipt SHA=`95131995ff9c77fa164e2f4fb9eb405bab5a00365e67cff5f459d4532040514f`，官方JEPA阶段已通过包校验并完成，score=`0.9135057926`，receipt SHA=`4c1efac7d9f4cad0acf6f62e61b0d51eb83142cc78dfc7dee51ef43613a1e1a2`。
- full15采用missing-only边界：复用已存在的9项generated-only分数，只补`Trajectory / Semantic / Depth / Background / Subject`五项及JEPA。首次missing5启动在GPU4评分前fail-close，原因是命令cwd错误指向不存在的`/home/huazhi/nlh/baseline/video_quality_ood`，未产生终态receipt。最小修复将官方源码/cwd解析回`/home/huazhi/nlh/WorldArena-2.0/video_quality_ood`；修复runner SHA=`cf5c332594c80efe2311975abecd577a438a3b715456253d50dca5e1fdcde0e8`，已在GPU0 missing-only重启并进入SAM3 detection，未重复JEPA或已有generated9。
- 当前不能报告Seed2九条的corrected15：`missing5-base.complete.json`与aggregate receipt尚未形成；必须等待missing5完成、合并15项并对Dynamic/Flow/Motion Smoothness应用matched-GT cap后，才能裁决这9个多seed候选是否真实提高总分。

### 2026-09-01 17:10 CST — Seed2 selected9 corrected15合并完成；仅确认弱正headroom

- Seed2九个generated9新增赢家的missing5与JEPA均已闭环，并在CPU完成15项合并及matched-GT motion cap。v2 completion receipt SHA=`a03b0a9ec0bd9c96e7b59d5b53f48a197ee7e0c198f33c9b762cbf7e0219d019`，summary SHA=`dde8af97f1859d842c83256a9b70f5f7dd616bb206b8bd3deaa3f2cc94f88776`；`full15_used_for_selection=false`。
- 9条候选集的corrected15=`.6980382250`、nonmotion12=`.7514784072`；相对这9条的seed1/seed4 incumbent表面delta分别为`+.0039364471`和`+.0045176335`。但该corrected15差值不能用于promotion：candidate JEPA是selected9整集合global score `.9135058`后按行重复，incumbent JEPA来自历史各shard global score后按行重复，两者统计scope不一致、不可直接比较。
- 排除不可比JEPA后，可比14项mean delta=`+.0009325909`。关键逐项delta：Instruction=`+.0222222`、Semantic=`+.0156794`、Image=`+.0064624`、Flow=`+.0039196`、Depth=`+.0033683`；Trajectory=`-.0328487`、Aesthetic=`-.0046903`、Background=`-.0031199`；Interaction与Perspectivity均为`0`。
- 将9/40替换比例折回fresh40，当前可比14项对整体full15均值的估算贡献仅约`+.000196`，且未包含完整40条统一scope的JEPA重算。因此本阶段只确认seed2带来**弱正路由headroom**，不确认可部署提分；promotion前必须在完整40条路由集合上用同一统计scope重算JEPA与最终corrected15。

### 2026-09-01 17:45 CST — Seed2 route40公平JEPA终裁：弱正GO

- 完整40条candidate路由（seed1/4 incumbent中仅替换9个generated9严格seed2赢家）与完整40条incumbent分别在相同scope重跑JEPA：candidate=`.9604148865`，incumbent=`.9608721137`，delta=`-.0004572272`。两条JEPA receipt SHA分别为`e9f5e4482696fc640527dae92d58ae5084d5bc8a54d2347f20963ab87b020064`、`fe5f01b9bbc21d006628fe05bffd04f4eb9418bcf2a98e5d79fed3163ee0806b`。
- 将各自full40 JEPA set score统一注入对应40行、其余14项沿用已corrected的逐episode值后，incumbent corrected15=`.6772225146`，route40=`.6787200538`，delta=`+.0014975392`（EWMScore_P约`+0.149754`分）；nonmotion12 delta=`+.0016984176`。
- 主要正向项：Trajectory=`+.0116408430`、Instruction=`+.0050000`、Semantic=`+.0038941`、Flow=`+.0017127`、Image=`+.0015588`；回撤：Aesthetic=`-.0013445`、Background=`-.0005003`、JEPA=`-.0004572`、Subject=`-.0000702`；Interaction与Perspectivity无变化。
- 逐条corrected15为`6胜/34负`：其中31条未替换episode仅因candidate全局JEPA略低而形成极小负差，少数seed2大幅Trajectory正收益抬高总均值。按预注册“同scope full40 corrected15严格正增益”门槛为**弱正GO**，但没有CI，不得描述为稳健晋级。
- 最终receipt=`runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/full15/route40-final-compare/route40-final-compare.complete.json`，SHA=`5b2d0f52be1fce14ba01bd3202c177ca1184c3f318eed5686869320ce1d5d5a8`；CPU MVP脚本SHA=`266244f278c30b57954690bcf2dad72eeb7654ff7edaadb681d8a28a74fbb1cd`。选择只使用generated9，full15/GT未参与路由。

### 2026-09-01 17:55 CST — Trajectory符号矛盾审计：基线不同，final receipt有效

- selected9 v2的`Trajectory=-.0328486807`使用的是诊断基线`max(seed1_mean9, seed4_mean9)`；route40 final的`Trajectory=+.0116408430`使用的是实际部署基线“冻结seed1/4 selector”。这两者不是同一个incumbent，selected9摘要也明确记录了前者的选择规则。
- 9个seed2键逐行核对后，7键incumbent一致，2键不同：`place_dual_shoes/37`从mean9-oracle seed1（Trajectory=`1.0`）切为冻结selector seed4（`.2890292260`）；`stack_blocks_three/46`从seed1（`.8993978558`）切为seed4（`.8490967837`）。Seed2对应值分别为`.5057032359`和`.7492779165`。
- 因此同一批9条的Seed2 Trajectory均值=`.7417089582`；对mean9-oracle incumbent `.7745576389`是`-.0328486807`，对冻结selector incumbent `.6899718782`则是`+.0517370800`；后者按9条替换折入40条正好为`+.0116408430`。Trajectory保持原始`[0,1]`scorer值，不属于三项GT-cap，未发生额外归一化。
- step1200的40条incumbent seed与冻结replay逐条一致；step1199的这9键均真实选择seed2。根因是早期诊断摘要与最终部署比较使用了不同基线，不是final compare实现错误；SHA=`5b2d0f52...`继续有效，无需作废。
- 逐行差异CSV SHA=`c035af83e93e3cae9f64b23669f21bc1918cb50d75d901e8ae720c1548ec38e2`；审计receipt SHA=`32eb5b12b379374a2676b1bd55d514e7b67b5dda4025ce84b8b1218e92325902`，路径=`.../full15/route40-final-compare/audit-trajectory-baselines/trajectory-baseline-audit.complete.json`。

### 2026-09-01 18:46 CST — Seed3严格增量route40完成；point-estimate GO

- 旧的seed3 `new14`路线被fail-close：它按四个seed重新取generated9最大值，改变了已冻结的seed1/4/2部署路由，不能作为对现有route40的严格增量实验。修正后先冻结seed142的40条既有选择，再只比较seed3与该冻结incumbent；最终seed3成为17条赢家，较已完成full15的14条多出`dump_bin_bigbin::episode000011`、`place_phone_stand::episode000039`、`place_shoe::episode000043`三条。最终四seed选择计数为seed1=`7`、seed4=`9`、seed2=`7`、seed3=`17`；选择仍只使用generated9，`full15_used_for_selection=false`。
- 同一full40集合、同一JEPA统计scope、同一matched-GT motion cap下，冻结seed142 route的corrected15=`.6787200538`、nonmotion12=`.7418188577`；增量加入seed3后的route40为corrected15=`.6805080600`、nonmotion12=`.7440373472`，delta分别为`+.0017880062`与`+.0022184895`。逐episode配对为新route `30`胜、旧route `10`胜、`0`平。
- 15项delta（新route减冻结seed142）：Aesthetic=`+.0002801`、Background=`+.0008781`、Depth=`-.0005081`、Dynamic(corrected)=`-.0005117`、Flow=`+.0005079`、Image=`+.0033070`、Instruction=`+.0150000`、Interaction=`+.0050000`、JEPA=`+.0002018`、Motion Smoothness(corrected)=`+.0002020`、Perspectivity=`+.0100000`、Photometric=`+.0019887`、Semantic=`-.0004150`、Subject=`-.0001416`、Trajectory=`-.0089691`。
- 预注册点估计门为“同scope full40 corrected15 delta严格为正”，因此当前判定`GO`；但receipt明确`confidence_interval_computed=false`，这只是point-estimate GO，不构成带CI的最终promotion证据。Trajectory回撤也必须在后续门中继续作为保护项审视。
- 最终receipt：`/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/full15/route40-seed142-frozen-plus3-final-compare/route40-final-compare.complete.json`，SHA256=`d0d476af5b8ad016db91ea66b0220eb4ad8513b53f2fc7f427e4b229fc8aeec3`。
