# 清理边界

归档验证后第一阶段允许删除的远端范围为：

`/data/di/worldarena2_track1_20260815`

该范围包含本项目的数据集、模型权重、候选视频、评分缓存、虚拟环境和中间结果。以下内容不在清理范围：

- `/data/fjy`
- `/data/whn`
- `/home/huazhi/nlh/baseline` Git 工作区
- Hugging Face 仓库 `clusternlh/worldarena-track1-hz-wam` 及其历史 revisions
- GitHub 仓库与本地当前脏工作区

## 已完成

2026-09-08 11:10（UTC+8）已删除上述精确目录。删除前逻辑大小为 `442919399873` bytes；删除后路径不存在，`/data` 可用空间为 `1035769221120` bytes。

第一次删除后剩余 `162239070` bytes：全部归 `huazhi` 所有，原因是 53 个目录为只读 `0555`，且未发现 immutable 标志。只给这些目录恢复 owner 写权限后完成精确根目录删除。

随后确认 GitHub 归档已固定最终生成/选片所需的代码、四个 RNG seed、唯一 Official FlowWAM checkpoint 的公开 revision 与 SHA256、官方测试输入 revision 与 SHA256、排序器全部系数及最终 1000 条 route。用户据此进一步授权清理 `/data/di` 的其余内容。

2026-09-08 11:21（UTC+8）又删除：

- `/data/di/robotwin_lora_train_v1`：`188794970` bytes；仅含未完成下载的日志/状态，无最终模型权重。
- `/data/di/worldarena2_track1_baseline`：`29183702897` bytes；主要是可按归档 revision 重新下载的公开模型/数据缓存、虚拟环境和日志。

新增释放逻辑大小合计 `29372497867` bytes。完成后 `/data/di` 无任何子项，`/data` 可用空间为 `1065297862656` bytes。

清理后已再次确认 `/data/fjy`、`/data/whn`、`/home/huazhi/nlh/baseline` 仍存在。机器上的 `/data/fjy/worldarena` 外部 GPU 进程 PID `2659785` 仍在运行，未向它发送信号。

机器可解析收据见 `cleanup.complete.json`。
