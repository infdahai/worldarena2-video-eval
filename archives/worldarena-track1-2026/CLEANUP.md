# 清理边界

归档验证后允许删除的远端范围仅为：

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

清理后已确认 `/data/fjy`、`/data/whn`、`/home/huazhi/nlh/baseline` 仍存在，未删除 `/data/di` 下其他一级目录。机器上仍有 `/data/fjy/worldarena` 的外部 GPU 进程，整个清理过程未向它发送信号。

机器可解析收据见 `cleanup.complete.json`。
