# Track 1 四 seed 正式提交包（2026-09-03）

状态：正式打包完成，本次全量复验及独立只读回验均 **PASS**。仅证明包格式与内容完整性；评分准入、上传和发送邮件另行处理。

## 本次边界

- Model Name：`HZ-World`；Version：`P0-FourSeed-Routing-v1`。
- 输入是冻结的 seed 1/4/2/3 路由，数量分别为 191/178/230/401；不是旧 `P0` 的 `full-1000-draft`。
- 924 条原字节复制；76 条只取首 `min(121, GT cap)` 帧，保持每条源 FPS，无时间重采样。
- 裁尾协议：`libx264 -preset medium -crf 12 -pix_fmt yuv420p -movflags +faststart -fps_mode passthrough -map_metadata -1`；每个 ffmpeg 2 线程。
- 4 个低优先级 CPU worker，未使用 GPU；不触碰既有训练/评分任务，不删除任何文件。
- 本任务不上传、不发邮件。Native holdout 的 full15 准入判断由主任务另行确认。

## 输入证据

远端根目录 `A=/data/di/worldarena2_track1_20260815`。

冻结目录：`A/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/test1000-routing-mvp/routing/seed1423-frozen-v1`。

| 输入 | SHA-256 |
|---|---|
| `test1000-seed1423-predictions.jsonl` | `4b80f62d979cfd6e3517a9f037602a11168a25451c9fe62a9f7805343956b936` |
| `seed1423-frozen.complete.json` | `4100b3758097165977aab7f36af9743875585d6b632acf5644e24ee18c02913f` |
| `A/submission/final-email/p0-dualseed-selector-v1/gt-frame-caps.json` | `742e17a8de0a1046b0118b7d3966ed5d1004294b1cdbb8357e7a8175fa7c718e` |
| 复用 README/计划/ZIP helper | `e7ad465b3c56f29c5955451e639da4af6df92532f51fb84f04740ebd4260073f` |
| 本地 76 条裁尾独立审计清单 | `65d3f4fa1e01c7453fb6335849cc7dbe4ba729d44f788aaaec796c1e3f7e79fe` |
| 本次独立打包脚本 | `efc70591b185e5da802bb13de43adc86f29d2dc501d7f302f1816a62c6964036` |

README 身份精确来源：
`A/submission/final-email/p0-dualseed-selector-v1/pretest-100-draft/staging/README.md`。

- Organization：`Huazhi AI`
- Responsible Person：`Disaster`
- Contact Email：`clusternlh@gmail.com`

沿用预测试 README 的 `Disaster`，未自行改名。已发预测试邮件的签名为 `Leihai Nie`，二者存在显示名称差异；本次未把邮件签名替换进 README。

## 独立输出

输出目录：`A/submission/final-email/four-seed-routing-v1`。

- ZIP：`HZ-World_P0-FourSeed-Routing-v1_Track1_Submission.zip`
- 收据：`submission.complete.json`
- 逐条证据：`progress.jsonl`
- 归档代码与测试：`tools/package.py`、`tools/test_package.py`
- 打包进程：PID `1570831`，nice `10`；任务耗时 `188.85` 秒，已产出完成收据。
- 执行日志：`/tmp/wa2-final1000-mvp-20260903/package.log`

启动前源视频合计 `287914354` bytes；保守新增预算 `921558912` bytes（小于 5 GiB），启动时剩余 `150666604544` bytes；低于 100 GiB 即停止。

本次最小适配的 3 项测试已先失败后通过，并在远端 Python 3.10 实际通过：多路径输入与 cap 规划、native-FPS/线程/不覆盖命令约束、ZIP 成员去重及内容 SHA 校验。

## 完成验收

| 产物 / 校验 | 本次实际结果 |
|---|---|
| ZIP SHA-256 | `bfc582494d2926fe2f71eb852fe1e7ca663fa0f02a8d5ec8514460fd20ebfbd2` |
| ZIP 大小 | `308026058` bytes |
| `submission.complete.json` SHA-256 | `de3b83585de3aeab5c8d8d3e961d1743964e4b44fc48d704bb17dc33840b4d9d` |
| 输入/输出新鲜校验 | 1000 条源 SHA 与完整解码通过；1000 条输出完整解码通过 |
| 帧数 / 分辨率 / 黑帧 | 每条声明帧数=实际解码帧数=`min(121, cap)`，640×480，黑帧 0 |
| 原生 FPS | 1000 条源与输出均 `24.0`；无时间重采样 |
| 裁尾 / 复制 | 76 条 prefix 裁尾；924 条源/输出 SHA 完全一致 |
| ZIP 结构 | 恰好 1001 个唯一、安全路径文件成员：根 `README.md` + `HZ-World/episode1.mp4` 至 `episode1000.mp4` |
| ZIP 内容 | CRC 通过；所有 ZIP 成员 SHA 与 staging 实际文件一致 |
| 独立回验 | 再读收据、再次计算全部源/staging/ZIP SHA、核对全部逐条帧数/FPS证据、身份和冻结输入，全部通过 |
| 结束前输出总量 | `617043962` bytes（写最终收据之前的守卫测量） |
| 结束前磁盘剩余 | `150039875584` bytes，高于 100 GiB 安全线 |

README 身份源 SHA-256：`6b1ff52686f7cbd0f759273434b1a263b9c4483444cce9f1db6c546c5ed7bd65`。

收据发布器字段已核实：`version=P0-FourSeed-Routing-v1`、`completed=true`、`video_count=1000`、`trimmed_count=76`、`zip_crc_pass=true`、`all_native_fps_preserved=true`，以及准确的 `archive_path` / `archive_size_bytes` / `archive_sha256`。

完成收据中的 `uploaded=false`、`email_sent=false`。该记录只涵盖打包任务完成时点；主任务后续若执行上传，需以独立发布收据为准。正式 full15 准入不由该打包任务证明。
