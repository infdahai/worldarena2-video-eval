# WorldArena 2.0 Track 1 实验归档（2026）

本目录是 HZ-World Track 1 项目的最终精简归档。它保留实验记录、决策收据、最终排序器/验证/打包代码和正式提交证据；不保留原始数据集、视频候选、模型权重、Python 环境、缓存或临时日志。

## 最终版本

- 模型名：`HZ-World`
- 版本：`P0-Predicted14-Routing-v1`
- 正式选择：1000 个 episode，在已有 seed1/2/3/4 候选中用冻结的 predicted14 排序器选择
- seed 计数：seed1=380、seed2=128、seed3=181、seed4=311
- 相对 P0 替换：368/1000
- 正式 predictions SHA256：`ea7046831a4837c880e6fdff77f7b38ca646090414cd8c8897ac2ffc883021c3`
- 正式 ZIP SHA256：`6c499fa583e7de450a2a00a2e74b849bd90076bd24f57b8c909500f2119ffb2b`
- Hugging Face revision：`fd9e024c8ed467b7deb27453337e44f18f22f59c`
- [匿名精确 revision 下载](https://huggingface.co/datasets/clusternlh/worldarena-track1-hz-wam/resolve/fd9e024c8ed467b7deb27453337e44f18f22f59c/final-1000/P0-Predicted14-Routing-v1/submission.zip?download=true)
- 正式邮件：2026-09-04 01:01（UTC+8）已发送；组委会确认与官方最终成绩仍属于外部状态

## 最终本地验证

冻结排序器在未用于拟合的原生 16 例上通过预注册完整 15 项 A/B 门槛：

- P0 `EWMScore_P=61.961235`
- predicted14 `EWMScore_P=62.312463`
- 增益 `+0.351228` 分
- non-JEPA 14 项均值增益 `+0.00379544`
- task-stratified 单侧 90% bootstrap 下界 `+0.00138519`
- JEPA 变化 `-0.00045198`
- Trajectory 变化 `+0.01951308`

这是本地公开协议下的小样本 A/B 证据，不是官方 1000 条绝对得分，也不应当被解释成官方排名。

## 目录

- `EXPERIMENT_INDEX.md`：从 Wan v1-v15、FlowWAM、ORB、refiner、四 seed 路由到最终提交的实验索引。
- `REPRODUCTION.md`：唯一 FlowWAM 权重、四个随机种子、固定参数、排序器权重和最终 ZIP 的完整复现链路。
- `experiment-records/local-reports/`：人工整理的实验报告和总账。
- `experiment-records/plans-and-specs/`：实验计划与冻结设计文档。
- `experiment-records/deliverables/`：本地交付记录；已排除 ZIP 和视频。
- `experiment-records/remote-top/`：远端实验事件、门禁、回放清单与报告；已排除 NPZ 张量。
- `experiment-records/remote-receipts/`：远端运行/提交收据；已排除视频、权重、环境和旧代码。
- `final-code/`：最终 predicted14 排序、原生 15 项验证、1000 条重排和打包链路。
- `final-results/`：冻结模型 JSON、完整 15 项比较、1000 条选择明细及提交收据。
- `MANIFEST.sha256`：归档文件内容哈希（不含 manifest 自身）。

## 复现边界

最终排序器输入是正式测试时可计算的 9 项生成侧特征，输出预测的 corrected14 均值；集合级 JEPA 和需要匹配 GT 的指标不被伪造成逐例输入。是否晋级由独立留出集的真实完整 15 项门禁决定。四个 seed 共用同一个公开可下载的 Official FlowWAM 权重；排序器权重完整保存在 `model.json`。由于数据集、基础模型和 4000 个候选视频按清理要求不在 GitHub 中，仓库通过 `REPRODUCTION.md` 固定公开来源、revision、参数、清单、哈希和执行代码，而不是直接保存这些大文件。
