# 最终代码说明

核心入口：

- `generation/`：Official FlowWAM 四 seed 生成入口、实际 seed2/3 七卡启动器、分片调度表和历史冻结脚本。
- `source-pins/`：唯一 Stage-1 权重、Official FlowWAM 源码、官方 Track 1 输入与 episode 清单的固定 revision/SHA。
- `src/worldarena_baseline/predicted14_ranker.py`：9 项 generated-only 特征到 corrected14 预测；包含拟合、预测和四 seed 排序。
- `scripts/fit_train160.py`：构造并拟合 160 episode × seed1/4 的冻结模型。
- `validation-tools/route_ranker_fresh16_mvp.py`：在留出集冻结选片，不读取 GT 倒推。
- `validation-tools/finish_ranker_fresh16.py`：按最新公开口径聚合完整 15 项并执行预注册门禁。
- `validation-tools/apply_ranker_test1000_mvp.py`：门禁 GO 后对正式 1000×4 候选重排。
- `validation-tools/package.py`、`final-results/delivery/tools/package.py`：生成并验收最终 1000 条 ZIP。
- `final-results/delivery/tools/publish_final.py`：上传固定 Hugging Face 路径并做匿名精确 revision 回读。

代码保留了当时远端绝对路径和冻结 SHA，作为执行证据。重新运行时必须显式替换数据根和环境路径；不能把预测 corrected14 当成官方 15 项绝对分。

四个 seed 是同一 Stage-1 checkpoint 的推理随机种子，不是四套权重。完整生成到提交链见上级目录 `REPRODUCTION.md`。
