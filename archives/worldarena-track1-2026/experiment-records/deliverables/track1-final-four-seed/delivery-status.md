# HZ-World 四seed正式提交交接

核验时间：2026-09-03 18:03 CST。**ZIP、上传、固定版本匿名下载及留出完整15项均完成；四seed未验证出优于P0，最终提交版本待确认，邮件尚未发送。**

- Model / Version：`HZ-World` / `P0-FourSeed-Routing-v1`。
- [正式1000条ZIP](https://huggingface.co/datasets/clusternlh/worldarena-track1-hz-wam/resolve/5849adbcf34140add6549e3cb40096faec6764fa/final-1000/P0-FourSeed-Routing-v1/submission.zip?download=true)
- 大小：308026058 bytes；SHA-256：`bfc582494d2926fe2f71eb852fe1e7ca663fa0f02a8d5ec8514460fd20ebfbd2`。
- HF revision：`5849adbcf34140add6549e3cb40096faec6764fa`。独立新增路径，旧tar、预测试ZIP和历史revision未修改。
- 1000条源/输出全解码及SHA通过；76条prefix裁尾、924条原字节保留；原生24FPS、640×480、逐episode帧数上限、black0通过。
- ZIP仅根README.md与HZ-World/episode1.mp4至episode1000.mp4，1001安全唯一成员，CRC及成员SHA通过。
- 无认证下载整包308026058 bytes，SHA与本地产物完全一致，ZIP结构/CRC复验通过。本机另以匿名HEAD核对revision、LFS SHA、大小一致。

## 邮件

[可直接复制的邮件文件](/Users/disaster/workspace/code/worldArena2_video_eval/deliverables/track1-final-four-seed/submission-email.txt)

收件人：`worldarenav2@outlook.com`。

主题：`HZ-World_P0-FourSeed-Routing-v1_Track1_Submission`。

README与草稿身份沿用预测试README：Huazhi AI / Disaster / clusternlh@gmail.com。不再发送预测试，不声称官方排名或未经验证的四seed总分。

## 尚未完成的边界

1. 与提交一致原生处理的16条留出完整15项已于17:55完成。corrected15×100：P0 64.185950、四seed 63.617427，四seed −0.568523分。主要回撤Subject −0.047635、Background −0.049560、Instruction −0.025000；Image +0.004289、Trajectory +0.013039、JEPA +0.006137未抵消回撤。非JEPA14项配对95% CI跨0；n16且非task-disjoint，不等于官方正式成绩。主控独立复算、全部输入输出SHA、96个GT封顶及384个其他项不变检查PASS。建议保守保留P0，等待用户确认；没有自动切换包、重选视频或追加实验。[完整实验记录](../track1-holdout16-native-results-20260903.md)。
2. 邮件尚未发送，官方接收/下载/正式评测尚未发生或未经确认。用户明确确认发送后才发正式邮件；也可由用户使用上述文件自行发送。
3. 截止为2026-09-04 24:00 UTC+8，即2026-09-05 00:00北京时间。预检100条66.2867与完整1000正式成绩不可混为一谈。

本目录保存submission.complete.json、hf-upload.complete.json、hf-anonymous.complete.json三份实际验收收据；原视频、P0和历史提交均保留，未删除任何数据。
