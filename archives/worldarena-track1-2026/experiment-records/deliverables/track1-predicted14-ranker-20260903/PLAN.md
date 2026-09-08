# 一次性预测14排序器 MVP

2026-09-03 用户批准：预测15项目标，独立验证后重排已有1000×4候选。不上新生成模型，不做参数搜索，不建生产框架。

## 固定实验

- Fit：Official FlowWAM selector200 的 train160×seed1/4；仅320行。9项 raw generated features → corrected14（排除集合级JEPA）。历史标签为81帧，只作训练，不能据此宣称原生提交效果。
- 模型：episode等权 StandardScaler + ridge alpha=10，单次多目标求解。部署输入没有task/seed/GT。替换条件是预测14均值较冻结P0 incumbent严格超过0.003，无附加veto，无搜索。
- 新验证：4任务×4未在扫描项目实验记录中出现的episode，四seed共64视频，排序器冻结后才选片与读GT评分。旧native16已曝光，完全隔离。
- 固定episode：open_microwave 48/28/44/11；press_stapler 49/35/15/43；put_object_cabinet 11/35/10/31；stack_bowls_three 27/30/21/43。
- 这些是episode-disjoint，不是task-disjoint；官方parent预训练暴露未知。新验证结果不能证明官方1000排名。
- 选片特征沿用正式generated9的81帧预处理；验收视频采用提交的native24FPS/640×480、仅超GT帧数上限时裁尾。GT使用完整序列。

## 只做一次通过/失败判断

比较冻结P0与新ranker选中集合：11项非motion逐例指标原值 + 3项逐episode min(candidate,matchedGT) + 真实集合级JEPA。

`EWM = 100 / 15 * (mean_i(sum corrected14_i) + JEPA_set)`。

晋级须同时：EWM点估计增益≥0.3分；非JEPA14配对均值增益>0且按task分层配对bootstrap单侧90%下界>0；JEPA与Trajectory各均值回撤不超过0.005。小样本不做强泛化结论。失败则停止本候选，不用新16调参重试。所有阈值在新16评分前固定。

## 时间与交付

- 新16生成与CPU fit并行，生成收据先完成的任务先评分；不等待全部卡。
- 预计新16全链2.5–3小时，按上一批实测外推，不是保证。
- 目标今晚完成验证结论；硬停止新增选模2026-09-04 12:00 CST。失败或超时保留原P0与已验收四seed包，由用户确认最终提交版本。
- 通过才用CPU重排正式1000×4已有候选，不重新生成正式视频；新版本另存，旧ZIP/HF revisions永久保留。
- 正式截止2026-09-04 24:00 CST；邮件未经明确授权不发送。

## 当前边界

轻量模块208行、43 tests主控复跑通过。真实fit与fresh16生成正在由两个子代理处理，尚不能声明模型已训练或有提分。新增≤20GiB、余量≥100GiB；仅合法空闲GPU0–6，GPU7外部任务不触碰。

## 20:07–20:10 实际执行（覆盖上段待启动状态）

- CPU fit已完成：160个训练episode×2seed=320行，alpha10；仅0.071秒数值拟合，无超参搜索。fit留在远端，未导出完整私人manifest。最初上传审批阻断，经主控只读核验用户指定host/目录所有权后批准，未绕过权限。
- 模型SHA `1409145a4efadfa7d274b2742a85318da6a56322472bc687e3f13844a7eb0fd5`；fit receipt SHA `36937e59c7b8db59d40b8ba85af4936ca2a3952988d20864d5b0b770316b99d3`；train rows SHA `727e5294615f60a40a54f7ed670d6371c617d34d640857658fb0e5c7e10ef2b1`。
- 远端H：`/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/ranker-fresh16-native-20260903`；模型在H/ranker-fit/model.json。
- 新manifest SHA `d661604840181036b5ddd8742654f8082c799012996344a38c790d8a394979d7`；input-contract SHA `73167b152951501041073400edbf423589694004b65b249de4dd57532949f3cc`；六旧清单交集均0。
- 20:04:47 GPU0–6生成worker实际启动，PID依次1727109/1727110/1727112/1727113/1727114/1727117/1727119；主控20:10回验7真实compute子进程、UUID、/proc、命令、cwd全部属本次H。GPU7外部2659785不变。
- 20:07:02七个评分接续worker启动：1730464/1730470/1730476/1730482/1730488/1730494/1730500。当前仅等待各自GPU生成锁，不声称已经开始评分；完成生成的卡自动接评分，无全局等待。
- 主控回读model/fit/train SHA全部一致。当前generation终态0、failed0；没有新验证分数、没有修改正式1000选择。
- 自动化已ACTIVE每10分钟，需按本记录接续。榜单与官方scorer源码/权重/预处理一致性由独立子代理只读核查，不扩展实验方案。

## 官方评测器一致性与接续入口

- 子代理核查官方main `b17a5b86cd38e54eabe699a848c701a829a216b5` 与现用 `7b3feee108427bee3380064bb5154970ed7468b5` 的 `video_quality_ood/` 全部tracked文件Git blob SHA零差异，包含VLM prompt、采样、JEDi、config、requirements。不升级代码、不重装环境、不重下载现成权重。
- 本地模型种类/路径匹配官方config；未证明与官方后台权重/依赖快照逐字节一致，最终1000 GT及后台上游调整未公开，不能保证本地绝对分等同官方分。同集合/同GT/同prompt/同环境A/B仍有效。
- 24FPS/121帧是我们生成协议，不是官方强制值；最终视频仅按逐例GT长度上限裁尾。Base读全部帧，VLM沿官方16帧采样；JEPA是整个集合MMD变换，禁止平均分片JEPA。
- 官方目录：https://github.com/WorldArena2/WorldArena-2.0/tree/b17a5b86cd38e54eabe699a848c701a829a216b5/video_quality_ood 。最终公告：https://v2.world-arena.ai/WorldArena-2.0/final-update.html 。Motion公告：https://v2.world-arena.ai/WorldArena-2.0/motion-quality-evaluation-update.html 。
- 已部署H/tools/route_ranker_fresh16_mvp.py，SHA `be5bec91bdd4ea9b26be88177bec2ffe8523771d29427c7f5717f3fb574419d4`；主控7接线tests PASS。输入16个completed generated9 receipts、冻结P0 model/policy、新ranker model；输出H/route/p0-selected16.jsonl、ranker-selected16.jsonl及全候选预测决策。当前未执行真实路由（评分未齐）。PYTHONPATH需远端baseline/src、H/tools、H/ranker-fit。
- 新native两组只用尚未占用step1232/1233，禁止覆盖旧1230/1231。H/tools/run_ranker_native_phase_mvp.py SHA `91521ec6bfd93162190ce0cff906e04107098748aee52cb93ed1b13c0349b551`，仅换目录/编号并将H全体新增数据纳入20GiB预算；100GiB余量及20GiB总预算测试PASS。
- H/tools/stage_native_holdout_eval_mvp.py SHA `07bc2739ebbb58b21a4a784661fee6d853a640cd920b096b7aa29b7d9a9ec670`，沿用已测试原生包工具，另须black0全解码验收。完整15项尚未启动，不误报结果。

## 20:15 本轮巡检

- GPU0–6真实生成compute仍全部运行，16.5GiB显存、利用率100%；GPU7外部不变。生成终态0、评分终态0、failed0。评分worker正常等各自生成锁，不重复启动。
- /data余149940527104 bytes，无需清理。
- 无新增GPU合法工作（7卡已在用）；派原CPU收尾子代理制作一次性 `finish_ranker_fresh16.py`，不是后台watcher。两组模型名固定 `RankerFreshP0Native16`（1232）、`RankerFreshPred14Native16`（1233）。仅在canonical raw15、真实两组JEPA及共享GTmotion收据完成后汇总，预注册门槛不变。

## 21:00 后本轮核验与接续

- 会话已恢复执行模式。新验证生成16/16批、64/64视频全部完成；主控逐份检查parent SHA、job manifest SHA、4条/收据、declared=decoded=121、640×480、black0记录，并重算全部64视频SHA通过。未重复生成。
- 本轮评分已由原有worker自动接续：首次检查5/16分片，随后6/16分片（24/64条）终态；已完成CSV/source receipt/manifest SHA回验通过，failed标记0。GPU0/1生成完成后转评分，不以模型加载间隙误判可重复启动；GPU7外部PID2659785未触碰。
- 已部署H/tools/finish_ranker_fresh16.py，SHA `1086090a5999c08fe7a3deebef3576ab86b90c5dd79bafc0c7ff356ada10d336`；主控本地6 tests再次PASS，上传后远端SHA与现有flowwam-v16环境的`--help`导入检查PASS。此脚本仅CPU一次性收尾，不是watcher，不会启动评分或改正式选择。
- route尚未执行，1232/1233尚未stage，完整15项尚无新结果；不提前宣称新排序器晋级。
- 后续仍为16份generated9收据齐全→冻结route→两组native package并全解码black0→官方base/VLM/整组JEPA+共享真实GTmotion→canonical aggregate→CPU finalizer。finalizer命令：`PYTHONPATH=/home/huazhi/nlh/baseline/src A/envs/flowwam-v16/bin/python -B H/tools/finish_ranker_fresh16.py --output-root H/analysis/full15-final`（A/H使用本文件固定绝对根）。输出comparison.complete.json/comparison.md与两份GTcap CSV；不直接修改正式1000条。
- 共享GTmotion复用baseline/scripts/run_latest_wa2_gt_motion.py，必须使用新1232预处理产生的完整GT帧目录，并先验证两组GT一致；输出仅H/full15-gt-motion，receipt固定H/full15-gt-motion/gt-motion.complete.json。不能复用旧native16的GT分数。

## 截止时间控制（用户再次强调9月4日24点前）

- 21:09实查7张项目GPU均已在实际VLM评分，7个worker及compute的UUID/user/PPID/PGID/cmdline/cwd核验通过；6/16评分终态，无failed。没有可以重复启动的空卡工作；保持当前任务运行。
- 目标今晚取得新16完整15项结论；剩余评分→路由/原生打包→完整15项按依赖接续，不能为追时限改回9项验收或删减指标。
- 2026-09-04 12:00 CST停止新增选模与调参；若新方案未PASS或出现时限风险，立即通知用户在保留的P0/已验收四seed包之间确认最终版本。不因超时自动宣称新方案通过，也不擅自停止外部或未核验进程。
- 内部交付目标：9月4日18:00前完成选定版本ZIP及匿名下载验收；20:00前备妥最终邮件和链接并取得发送确认，给24:00官方硬截止留余量。正式发送仍需要用户明确授权，前100预测试不重发。

## 21:12 接续预检

- generated9完成10/16分片（40/64条），CSV SHA逐份核验PASS，失败0；尚未凑齐路由输入，未提前用部分结果选片。
- GPU2本轮没有compute，但原评分worker1730476仍存活并等待其他已claim工作完成；不重复启动。其余6张项目卡继续剩余评分，GPU7外部不变。
- 下一阶段预检：官方source固定7b3feee；base/VLM/JEPA/aggregate所需环境及权重路径均无缺失。route、native stage、phase wrapper、CPU finalizer四工具冻结SHA全部PASS。H占用512404538 bytes，/data余149405409280 bytes，预算有余量。没有新增依赖或代码阻断，待16/16评分收据齐全即可接续。

## 21:22 本轮推进

- generated9完成13/16分片（52/64条），已完成CSV SHA通过，无failed。剩余seed3的press_stapler/put_object_cabinet/stack_bowls_three分别在GPU6/5/3继续VLM；GPU0/1/2/4的原worker仍持锁等待，不能把没有compute误判为可重复接单。
- 六个native phase均已通过真实远端`--dry-run`（不是启动）：1232 P0的base/VLM/JEPA候选GPU0/2/3；1233 ranker的base/VLM/JEPA候选GPU4/5/6；GTmotion预留GPU1。实际启动时必须重新核验UUID、外部进程、原worker退出和锁；映射不授予抢卡权限。
- 两组`--expected-count 16 --split dev-clean-50 --estimated-additional-bytes 1073741824`通过，reserve=107374182400。包验证状态仍为`deferred_until_videos_are_staged`，不能宣称原生包或15项已经通过。

## 21:37–21:41 实际接续（取代上述待评分/待路由状态）

- generated9 已16/16收据、64/64视频评分完成，全部CSV SHA回验通过；原7评分worker均退出，不再生成或重评。
- 唯一冻结route已完成；P0选seed1/4各8条，新ranker选seed1=7、seed4=7、seed2=2。预测不当实测。P0 predictions SHA `31690f1665e98bf9c96705035d0159a6845ad2d12ddd58a42b1d6cc48ad88c2d`；ranker SHA `85e8fd067858567ee2f16b32cf691c91624df1a4c15492472ae6d073620dc14f`。
- step1232/1233两组原生包已完成，各16条。主控再次全解码32视频验证尺寸/帧数/SHA/black0，并核验两组GT HDF5与编码GT SHA/帧数完全一致。未用GT倒推选片。
- 六phase真实dry-run包验证passed后独立上卡：1232 base GPU0 PID1850255、VLM GPU2 PID1850265、JEPA GPU3 PID1850275；1233 base GPU4 PID1850285、VLM GPU5 PID1850295、JEPA GPU6 PID1850305。每GPU独立H锁+公共锁+phase锁，/proc及UUID检查无冲突；GPU7外部2659785不触碰。
- 21:40回验两JEPA已完成并退出：P0=0.9360034465789795，ranker=0.9355514645576477，差-0.0004519820213318（仅JEPA门目前通过，不代表总分GO）。base两路真实GPU detection/tracking，VLM两路真实评分，非空转launcher。
- 新1232全部16 GT预处理帧已逐episode与原生包计数核验一致；共享GTmotion接GPU1独立计算。剩余base/VLM/GTmotion齐→两组canonical aggregate→H/tools/finish_ranker_fresh16.py固定门限比较。不得复用旧GT/旧16结果；不得改阈值。
- 21:41:54共享GTmotion已实际启动，GPU1 launcher/PGID1858143，16例5656 GT帧；接续记录H/gt-motion-launch.json、日志H/logs/native-gt-motion.log。自动化已同步真实启动/JEPA完成状态，ACTIVE每10分钟，无变化静默。
- 时间线不变：今晚争取完整15项结果；9月4日12:00锁版，18:00 ZIP和匿名验收，20:00邮件材料/固定链接与最终发送确认，24:00官方截止。无确认不发送正式邮件。

## 21:50后 用户批准执行：正式重排输入预检

- 保持当前两组base/VLM与共享GTmotion运行；/proc和UUID核验五路真实compute正常，无Traceback。没有重新启动生成/评分，也没有修改冻结排序器、阈值或正式选择。
- 对正式1000×4输入做CPU只读预检PASS：四seed各1000条，1000唯一episode，4000完整有限9项特征，4000原生视频实际SHA全部吻合。未调用排序器预测、未读GT、未输出新正式选择。
- seed1/4特征与原生视频映射：A/submission/releases/p0-dual-seed-parallel-r1/final-p0-r3/test1000-scored-candidates.jsonl，SHA `64aaeb9aede4f41db16c8f9661eadd59cd8a8907996a385c00594eb317459499`。P0 incumbent来自同目录test1000-predictions.jsonl，SHA `ae70bc87505ee10aad0138651283efe2cb07dcab35b2c65c83e29acc90671101`，逐例seed/视频SHA与候选一致。
- seed2/3：E/test1000-routing-mvp/scoring/generated9/seed{2,3}/shard{0..7}/package；各125条completed生成侧CSV、source Stage1 receipt SHA、package的source_video/source_sha256已验证。按Video_ID数字后缀与package episode_id联接，不能误取81帧score_video当原生视频。
- 通过后沿用冻结rank_candidates，阈值.003、无veto，task/episode只作身份（正式episode须有非纯数字全局键），不做训练；先严格验证H/analysis/full15-final/comparison.complete.json为GO及输入SHA，再输出新独立正式路由目录。未GO时不得执行此分支。

## 21:52后 GTmotion收尾恢复（未重跑评分）

- GPU三项评分已经各16/16完成并退出；原wrapper期待`reference_dataset_results.json`，实际官方写出`reference_results.json`，导致仅收据收尾FileNotFoundError。不是评分缺失，其他base/VLM继续正常。
- 主控按固定manifest验证三项各16个task+episode身份、有限归一化分数，确认旧GT父子进程退出且shared-gt-motion锁可独占。保留原`reference_results.json`，创建字节一致的`gt_reference_results.json`副本，再用原命令加`--finalize-existing`仅CPU恢复收据。
- H/full15-gt-motion/gt-motion.complete.json SHA `ef2b204e2ea2bd3b585bd0bee7998eb264de769a766c15466739ee25ba9caa5f`；结果SHA `e5297d409402e6e98a98dddcfc2acd2b88026bbbd9b410cd0bd687d5df7c97c5`，三项各16例、completed=true。没有改分数、没有重跑GPU、没有重算旧样本。
- 余下两组base/VLM齐后各自CPU aggregate，再执行冻结finalizer；不再启动GTmotion或JEPA。

## 22:11–22:14 用户最终计划执行回验

- 当前按用户批准的唯一最终计划执行，不新增生成器训练、seed、正式视频生成或参数搜索。无需重新启动已经完成的fit/64候选/generated9/route/native staging。
- 22:11远端实查：1232/1233的VLM均13/16，base仍在正常detection/tracking，四路日志均无Traceback；GPU0/2/4/5实际compute及父进程、UUID、user/PPID/PGID/cmdline/cwd核验通过。GPU7外部任务未触碰。没有可重复拆分当前输出的安全入口，不为填空卡重跑已有评分。
- 共享GTmotion完成收据SHA及两组JEPA值已复核。冻结ranker model/module、native phase wrapper、CPU finalizer的远端SHA全部一致；本地主控对排序器43项、路由7项、收尾6项共56项测试复跑通过。首次测试因漏加独立ranker模块的PYTHONPATH未能收集，补正确模块路径后通过，未修改模型或代码。
- 目前仍没有base/VLM完成收据或comparison.complete.json，因此没有GO结论，没有重排正式1000条，没有生成/上传新ZIP，也没有发送邮件。已验收四seed保留包与打包工具SHA复核一致。
- 每个组base/VLM齐全即CPU aggregate，不等另一组；两组aggregate齐全即运行冻结finalizer一次。只有全部门槛通过才调用现成rank_candidates对正式4000特征重排，不重新训练；否则停止本候选，请用户确认保留版本。
- 自动化继续每10分钟接续，清除旧待生成/待fit状态，以本轮实际收据和固定门槛为准。9月4日12:00锁版、18:00 ZIP与匿名验收、20:00邮件材料及发送确认、24:00官方截止不变。

## 22:17 VLM阶段完成

- 1233新ranker组VLM已16/16完成，launcher1850295及compute1850470退出、GPU5释放。严格核验completed/phase/step/model/source_commit/package16条通过；完成收据SHA `7aca7dd9420459621a666ed72e0b1821a681467ba46e346f9c33cec8d5f16ca2`，canonical VLM结果16条、SHA `b58b684d0a0e14d1bc250e26423e4e78eff370b731bc3b17d410378774e8ad98`。
- 1232 P0组VLM15/16继续运行；两组base仍在GPU0/4正常推进，尚未满足任一组aggregate条件。没有重启已完成评分或提前生成总分。
- 本轮8卡UUID及compute/launcher的user、PPID、PGID、cmdline、cwd已核验；GPU7外部不变。余量148421599232bytes，无需清理。

## 22:28 两组VLM全部完成

- 1232 P0组VLM已16/16完成，launcher1850265/compute1850469退出，GPU2释放；completed/phase/step/model/source_commit/package16条均通过。完成收据SHA `9b85ecf09ba2804593910f5ec9194a36cbf99c4e22f0044f43d4f8bb2451ba38`；canonical VLM结果16条，SHA `397262f48b76ef39f2a1b04fd6e395780dacb82c502e1d709ea3c65eae85d2d6`。
- 两组VLM与JEPA收据回验全部通过，不再重跑。仅两组base仍在GPU0/4的detection/tracking推进，日志无Traceback，父子进程与UUID核验正常。尚无base/aggregate收据，不能提前聚合或宣称总分通过。
- GPU7外部任务不变，/data余148395319296bytes。接续仍为各组base完成即CPU aggregate，两aggregate齐后冻结finalizer，不改变门槛。

## 23:31–23:40 最终验证、1000条重排与ZIP闭环

- 两组base均完成；1232 base receipt SHA `8ba3af7364287a01ed290d0ad8e694803c5521b89a79c5aac2e93868399df62d`，1233 SHA `3585ac172e5caedb0316a5f663e0a7087ea27d785a674e5d0e2640df44c76f1d`。canonical aggregate receipt SHA分别为 `b366c80d73538dfc7e7b43cee9f10783aeaee4200d73c655b05d493f69d41aa3` 与 `85cfb49ffe97ba60d13aa8198fc581599d66e9207ae4b72b1db3cb898b769e92`。
- 冻结finalizer完成，comparison receipt SHA `585655d738f0fae403282da53c8f7c09d400fa4189d282e74b3f59917a5d72cf`，决定 `GO`。P0 EWMScore_P=61.961235，ranker=62.312463，提升 `+0.351228` 分；non-JEPA14 mean delta `+0.00379544`，task内配对bootstrap单侧90%下界 `+0.00138519`，95%区间 `[+0.00006103,+0.00826803]`；wins/ties/losses=7/7/2。JEPA delta `-0.00045198`；Trajectory delta `+0.01951308`。五项预注册门槛全部通过。此16例是本地公开协议A/B证据，不冒充官方1000条绝对分。
- 正式1000条只做一次CPU重排。最初MVP被真实旧package合同 `rows=4/videos=125` 拦截，未创建输出；加入精确legacy-contract回归后47 tests PASS，随后新独立目录成功完成。routing receipt SHA `20d02d7c959a02b8d7ea328a96731fce4e6517fd8cea1a081403e23673469706`；predictions SHA `ea7046831a4837c880e6fdff77f7b38ca646090414cd8c8897ac2ffc883021c3`；details SHA `4f14ad3f45dcb9fee212de46f843e549cc65d316aba9967082fe41feebc33f33`。1000唯一episode、4000候选输入，阈值严格大于0.003；相对P0替换368条，seed1/2/3/4计数=380/128/181/311。未使用隐藏GT或集合JEPA路由。主控再次重算1000个所选源视频SHA全部一致。
- 新提交版本 `P0-Predicted14-Routing-v1` 已生成；打包器仅替换冻结输入、版本与GO收据，4项合同测试PASS，SHA `71d3f5ae4e2e8c5b02cf209eb6412c8d485bc7322e3fc9b43215cd8cc925a6fd`。ZIP路径 `/data/di/worldarena2_track1_20260815/submission/final-email/predicted14-routing-v1/HZ-World_P0-Predicted14-Routing-v1_Track1_Submission.zip`，307615897 bytes，SHA `6c499fa583e7de450a2a00a2e74b849bd90076bd24f57b8c909500f2119ffb2b`。1000视频、76按GT帧上限裁尾、924原字节复制、640x480、原生FPS、全解码black0、ZIP CRC/安全成员/逐成员SHA均通过；独立回读1001成员与README身份通过。package receipt SHA `a4ae41da43ced375f2d71b3c285f93b469ac51c852b438f3a1c6db24ded8d56d`。
- HF发布脚本仅改新ROOT/DEST/version并加入native full15 GO断言，py_compile通过，SHA `552db6f55a3f20420b7c199ed2b1ef48e1d491955b82b798f431d60b34cbf823`。预定目的地为公开dataset `clusternlh/worldarena-track1-hz-wam` 的新路径 `final-1000/P0-Predicted14-Routing-v1/submission.zip`，保留所有历史revision。实际外部上传被安全审批拦截，要求用户明确确认该具体repo；没有上传、没有匿名回读、没有发送邮件。当前唯一阻断就是这项明确授权。

## 9月4日00:49 正式提交授权与权限阻断

- 用户明确批准正式提交本版本。已将上述相同SHA发布脚本传至新包目录的tools/，未改视频或ZIP。
- 上传命令在执行前再次被权限审查拒绝：要求用户明确授权具体公开仓库 `clusternlh/worldarena-track1-hz-wam`；泛化的正式提交授权未被审查接受。不绕过拒绝、不换途径发布。
- 新版ZIP尚未上传，尚无新版匿名回读或正式邮件发送证据。已确认原Gmail账户可访问，但未点击发送。待用户明确确认公开仓库后继续上传、匿名验收和正式邮件发送，勿重复评分/打包。

## 9月4日01:01–01:03 正式提交已发送

- 用户已逐字授权公开上传至 `clusternlh/worldarena-track1-hz-wam` 并发送链接至 `worldarenav2@outlook.com`。权限审查通过后运行同一冻结发布器，无重新打包、重排、评分或生成。
- 新HF revision `fd9e024c8ed467b7deb27453337e44f18f22f59c`，路径 `final-1000/P0-Predicted14-Routing-v1/submission.zip`。匿名精确revision整包下载307615897 bytes，SHA `6c499fa583e7de450a2a00a2e74b849bd90076bd24f57b8c909500f2119ffb2b`，1001安全唯一成员、1000视频、CRC全部通过。发布器exit0；保留旧P0、四seed与历史revision。
- Gmail精确已发送搜索先确认本标题无重复；随后仅发送一次。Gmail显示“Message sent”，已发送页面回读正文、完整收件人、日期、标题和固定下载链接通过。收件人 `worldarenav2@outlook.com`；主题 `HZ-World_P0-Predicted14-Routing-v1_Track1_Submission`；Gmail显示发送时间2026-09-04 01:01 UTC+8。
- 已发送链接：https://mail.google.com/mail/u/0/#sent/KtbxLvHgMkZmbcTsGSgBfsxMVXdbRdVFGB 。邮件只指定本1000条ZIP为唯一最终正式版本，不重复预测试，不宣称本地62.312463是官方1000分数。
- 本地交付证据：`deliverables/track1-final-predicted14/` 中的 `delivery.complete.json`、`delivery-status.md`、`submission-email.txt`、`hf-upload.complete.json`、`hf-anonymous.complete.json`。发送已完成不等于组委会已确认接收；官方确认回信和最终成绩仍待返回。提交自动化收尾后暂停，禁止重复发送。
