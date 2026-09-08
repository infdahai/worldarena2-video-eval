# 最终代码说明

核心入口：

- `src/worldarena_baseline/predicted14_ranker.py`：9 项 generated-only 特征到 corrected14 预测；包含拟合、预测和四 seed 排序。
- `scripts/fit_train160.py`：构造并拟合 160 episode × seed1/4 的冻结模型。
- `validation-tools/route_ranker_fresh16_mvp.py`：在留出集冻结选片，不读取 GT 倒推。
- `validation-tools/finish_ranker_fresh16.py`：按最新公开口径聚合完整 15 项并执行预注册门禁。
- `validation-tools/apply_ranker_test1000_mvp.py`：门禁 GO 后对正式 1000×4 候选重排。
- `validation-tools/package.py`、`final-results/delivery/tools/package.py`：生成并验收最终 1000 条 ZIP。
- `final-results/delivery/tools/publish_final.py`：上传固定 Hugging Face 路径并做匿名精确 revision 回读。

代码保留了当时远端绝对路径和冻结 SHA，作为执行证据。重新运行时必须显式替换数据根和环境路径；不能把预测 corrected14 当成官方 15 项绝对分。
