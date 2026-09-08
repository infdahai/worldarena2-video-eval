# 最终结果复现链路

这份归档能够解释并重新执行最终 `HZ-World / P0-Predicted14-Routing-v1` 的生成、选片和打包逻辑。GitHub 不保存大模型、官方测试数据或 4000 个候选视频；这些二进制输入均以公开来源、固定 revision、大小和 SHA256 绑定。

## 1. 权重与随机种子

seed1、seed2、seed3、seed4 是同一模型推理时的随机种子，**不是四套权重**。四组候选全部使用同一个 Official FlowWAM Stage-1 checkpoint：

- Hugging Face：`YixiangChen/FlowWAM`
- revision：`1e68f76cecfb2caa973abfb24fca92cbc5312a6e`
- 文件：`flowwam_worldarena_stage1.safetensors`
- 大小：`10137267208` bytes
- SHA256：`e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`

下载命令：

```bash
hf download YixiangChen/FlowWAM flowwam_worldarena_stage1.safetensors \
  --revision 1e68f76cecfb2caa973abfb24fca92cbc5312a6e \
  --local-dir inference/models/stage_1/
```

基础 Wan 模型、embodiment 资产和源码 pin 见 `final-code/source-pins/flowwam_v14_pins.json`。实际生成使用 `YixiangChen515/FlowWAM_WorldArena` commit `f06fa46042e97738c6619c868f1097be6749d48d`；每个生成收据也绑定了这个 commit 和上述 checkpoint SHA。

## 2. 官方测试输入

输入来自 `WorldArena/WorldArena2.0` revision `af1ac34d3881f84096345542c631fbb1b9540d50` 的 Track 1 包：

- archive SHA256：`7973bf1e9222086279aa12eb593b4c7adb88f781a5811b05ae5134d52c96bb41`
- 1000 个 episode，`data`、`first_frame`、`instructions` 各 1000 项
- 完整收据：`final-code/source-pins/official-track1-dataset.complete.json`
- episode 清单：`final-code/source-pins/episode-manifest.jsonl`
- 八个 125 条分片：`experiment-records/remote-receipts/submission/releases/p0-dual-seed-parallel-r1/manifests/`

该官方测试输入只用于最终推理和提交，不用于训练、选择器拟合或阈值调参。

## 3. 四 seed 候选生成

核心入口是 `final-code/generation/run_flowwam_official_stage1.py`，其调用 Official FlowWAM，并严格输出 121 帧、640x480 的 Stage-1 视频。最终参数由各 `stage1-only.receipt.json` 固定：

```text
seed = 1 / 2 / 3 / 4
height = 480, width = 640, frames = 121
flow_height = 240, flow_width = 320
num_inference_steps = 50
sigma_shift = 5
max_rollouts = 2, max_stride = 3
flow_max_magnitude = 20
stage2_refiner = identity-disabled
```

- seed1/4 使用归档内的 16 个分片清单；seed1 单卡入口保留在 `run_track1_formal_seed1_generate.sh`。
- seed2/3 的实际七卡 MVP 入口是 `launch_test1000_seed23_mvp.sh`，调度表是 `test1000_seed23_work_manifest.jsonl`。
- `worker.py` 是生成入口所依赖的基线 worker 快照。其余 Official FlowWAM 代码由上述固定 commit 提供。

## 4. 排序与最终 1000 条

流程不是“按 seed 权重混合”，而是逐 episode 选一个完整视频：

1. 对四个候选计算正式测试可获得的 generated-only 9 项特征。
2. 使用 `final-results/ranker-fit/model.json` 中冻结的标准化器、截距和岭回归系数，预测 corrected14 均值。该 JSON 就是排序器的全部权重，`alpha=10`。
3. 以 P0 为 incumbent；新候选预测增益严格大于 `0.003` 才替换。
4. 留出 16 例按原生视频、完整 15 项和 GT cap 验证为 `GO` 后，才把同一冻结规则应用到正式 1000x4 候选。

关键入口：

- 拟合：`final-code/scripts/fit_train160.py`
- 排序器实现：`final-results/ranker-fit/predicted14_ranker.py`
- 正式重排：`final-code/validation-tools/apply_ranker_test1000_mvp.py`
- 最终 route：`final-results/test1000-routing/test1000-predicted14-predictions.jsonl`
- route 收据：`final-results/test1000-routing/route.complete.json`

最终选择计数是 seed1=380、seed2=128、seed3=181、seed4=311；相对 P0 替换 368 条。predictions SHA256 为 `ea7046831a4837c880e6fdff77f7b38ca646090414cd8c8897ac2ffc883021c3`。

## 5. 打包

`final-code/validation-tools/package.py` 读取冻结 route，对 76 个超过 GT 帧数上限的视频只做尾部裁剪，其余 924 个字节复制；随后全解码核验帧数、640x480、黑帧、SHA、ZIP 成员与 CRC。

最终 ZIP：

- Hugging Face revision：`fd9e024c8ed467b7deb27453337e44f18f22f59c`
- SHA256：`6c499fa583e7de450a2a00a2e74b849bd90076bd24f57b8c909500f2119ffb2b`
- 结构：根目录 `README.md` 与 `HZ-World/episode1.mp4` 至 `episode1000.mp4`

本归档足以重建算法和执行链；重新生成视频仍需重新下载固定的公开模型、官方测试包和依赖资产，并配置本机路径。
