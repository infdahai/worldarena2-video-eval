# Generated9 merge recovery — 2026-09-02 20:10 CST

Outcome: seed2 shards 0–4 now have five verified terminal receipts, 625 unique episodes, and nine generated-only metrics per episode. No videos or GPU scores were recomputed. This is not an official full15 score or a model improvement result.

Root cause: the reused four-example adapter's `merge_package` hard-coded `expected_count=4`; the test shards contain 125. GPU base/VLM scoring succeeded but final CSV merging failed. Legacy package/phase `rows=4` metadata is preserved; actual package video membership, video SHA, CSV identities and row count are verified independently.

Minimal fix: only `tools/score_test1000_generated9_shard.py`; the shared scorer remains unchanged. Manifest-sized merge and `--merge-only` were added. Wrapper SHA256: `44ab66ff0a828b764415e5e7f43e28fecdd04372f5eee5980223ecfee76eb744`.

Verification: three regression tests failed before the fix (including the actual wrong-row-count failure) and passed after it in the existing remote FlowWAM environment. Tests cover 125-row success without rescoring, same-count wrong episode rejection, and incomplete-phase rejection without starting GPU work. Python compile passed locally.

Recovery: checked each target had no active process and no terminal receipt; acquired its existing per-shard flock; ran the wrapper with `--merge-only`. Read back five receipts and all CSVs, verifying SHA, 125 unique rows per shard, nine metrics, and 625 unique rows across shards.

| seed2 shard | CSV SHA256 |
|---|---|
| 0 | 96bafb319d422219cef7ff3b2b3af61deba4bffcbbd3ee0d2f74fb729df72212 |
| 1 | 42586348254e50494c1f384c59c7654a4e3c78baad4234c5e78f9b3042864e79 |
| 2 | c9759629a2aeea83a2923b88589defd13469821660fb1e6ffeb6f96083130df0 |
| 3 | 8b4b536cea5fa1f41a67d5dd8eb668cca6917f0e85437e39599bf91e59e55fa0 |
| 4 | 378abb10ea7b20b35969c57303b000c20d49d3f4996e27f6f7145e5e6dcc3c19 |

Remote root: `/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/test1000-routing-mvp`.
Terminal pattern: `scoring/generated9/seed2/shardN/package/receipts/test1000-generated9.complete.json`.

20:10 snapshot: seven project GPUs remain busy. GPU0: seed2 shard6 VLM (59 raw JSONs); GPU1: seed2 shard5 VLM (84 raw JSONs); GPU5: seed2 shard7 base. GPU2/3/4/6: seed3 shards0/1/4/3 base. GPU7 remains external `/data/fjy`; untouched. Raw JSON counts are progress indicators, not completed validated rows.

Next: running processes loaded the old wrapper before deployment and may still hit the old merge failure on exit. After their processes exit and claims unlock, use the fixed wrapper's CPU-only merge recovery. New starts use the fixed wrapper. Seed3 shards2/5/6/7 remain pending until a project GPU becomes safely available. Do not restart active GPU scorers or repeat completed work.

## 22:00 CST update

Seed2 shard5 base and VLM finished (125/125), and its old wrapper exited with the known four-row merge error. Verified the exact wrapper/children were absent and acquired the original work lock before CPU-only `--merge-only` recovery. Exit 0; read back all six terminal receipts and CSV hashes: **750 unique episodes, nine metrics per episode**. Shard5 CSV SHA256: `fe940c245ed635fdd0586e49323a6aa6b7f7bc597a28a26855c0b98862e66bfc`. No GPU scores or videos were recomputed.

GPU1 UUID `GPU-a3ba221d-3832-3906-cef7-f8aae054c149` was free of compute processes and live project wrappers. Started the previously unstarted seed3 shard2 on that GPU with the verified wrapper and original exclusive work lock. Wrapper PID/PGID `461648`, user `huazhi`; lock confirmed held. Package staging completed; at 22:00 its child PID `475134` was running CPU preprocessing. GPU1 had not yet begun GPU computation and must not be treated as available for duplicate work.

Remaining: seed2 shard6 VLM (104/125 raw JSONs at 21:58), shard7 base evaluation; seed3 shards0/1/3/4 base tracking and shard2 preprocessing. Only seed3 shards5/6/7 remain unstarted. GPU7 external `/data/fjy` process `2659785` remains untouched. Disk: 147 GiB available. Final selection/1000-video package are not yet complete.

## 22:59 CST update

Seed2 shard6 completed base and VLM (125/125), then its old wrapper exited with the already known four-row merge error. Verified no live target wrapper/children, acquired the original work lock, and ran only `--merge-only`: exit 0. Read back all seven receipts and CSV SHA values, confirming **875 unique episodes with nine generated-only metrics**. Shard6 CSV SHA256: `631c5c7ee96933db1cdd1a4552c41e2104b91f55a7f1ea7436465d3fe11daebe`. No GPU rescoring.

GPU0 UUID `GPU-66b5e202-9179-0f61-c0dd-34d5599ffde2` was free of compute processes and live project wrappers. Started seed3 shard5 with the verified wrapper, PID/PGID `535066`; original work lock confirmed held. At 22:59:42 it was staging score videos via ffmpeg child `547259`, not yet GPU computation. The original candidate videos remain unchanged. Do not mistake CPU staging for an available GPU assignment.

At 22:58 seed2 shard7 remained on GPU5 in VLM (11/125 raw JSONs). Seed3 shards0/1/3/4 had advanced to VLM on GPU2/3/6/4 (4/4/4/1 raw JSONs); shard2 was tracking on GPU1. **Only seed3 shards6/7 remain unstarted.** GPU7 external job remains untouched. Disk: 146 GiB available. Seed2 global 1000-row merge/routing remains blocked on shard7, so no route was frozen from incomplete input.

## 2026-09-03 03:32 CST — seed2 complete and three-seed route frozen

At 03:25, seed2 shard7 and seed3 shards0/4 had completed both GPU phases and exited with the known old four-row merge error. Verified exact wrappers/children absent and original locks available. Ran CPU-only merge recovery for all three; no GPU rescoring. Seed2 now has eight verified terminal receipts and **1000 unique episodes × nine metrics**; seed3 has two terminal receipts / 250 unique episodes.

Recovered CSV SHA256:
- seed2 shard7: `e8612fd85f611885b779598ac0b7fa17bbef69a5da64d58510be0328ea57f4b4`
- seed3 shard0: `3afa6f954246623de020d4a52659d619a3635cac49d8ab9ce6264034ad884af9`
- seed3 shard4: `6c659227817215f82032d27c544d376b9078f170d606c9d83553c4a0258dd8a0`

Immediately started the last two unstarted score shards after source receipt, all 250 source video SHA/structure records, GPU UUID, process absence and exclusive work-lock checks:
- seed3 shard6 → GPU2 UUID `GPU-af789443-b4a8-6d90-1dcd-4404cb382fea`, wrapper PID/PGID `792083`.
- seed3 shard7 → GPU4 UUID `GPU-651bd630-9a1d-c4c2-1089-8501d6ef1051`, wrapper PID/PGID `792223`.

Both original locks were confirmed held. At 03:32 they had real compute PIDs `827492` and `827558` respectively (4724 MiB each). GPU0/1/3/6 continue existing seed3 VLM work. GPU5 is free: all remaining unique score work is already claimed; no duplicate work launched. External GPU7 `/data/fjy` remains untouched.

CPU MVP `tools/worldarena_freeze_seed142_mvp.py` applies the previously validated conservative rule: keep the original P0 seed1/4 choice unless seed2 mean9 is strictly above **both** seed1 and seed4. Four narrow rule tests ran RED before implementation, then GREEN. It verifies frozen P0 input SHAs, all eight terminal receipts and CSV identities, source videos, and exact episode1–1000 coverage. It reads generated-only inputs; no GT or seed3 inputs. Output is write-once and does not package or submit videos.

Frozen route: **seed1=320, seed4=331, seed2=349**. Independently re-read all 1000 choices against original P0 and generated9, rechecked selected video SHA, CSV and receipt SHA. This is a test selection artifact, **not an official full15 score or confirmed performance improvement**.

Remote output root: `test1000-routing-mvp/routing/seed142-frozen-v1`.
- `seed142-frozen.complete.json` SHA: `1368b724cfa831da20268ff1ccd059e50eeab6a6135b01a13cfe2c792b335f74`
- `test1000-seed142-predictions.jsonl` SHA: `0e3bed24e98b2c696b37e22bbeb43bb69a72ea0be7e06f0d4017dbe046c08a5f`
- `seed2-generated9.csv` SHA: `2d284c9b94e26b1c0696d24c484eb47716a2b201a0a6d6a643d3a780cfa7da39`
- CPU script SHA: `2e35d5c0e3bd9cbb119b8634845d048d7d764d3fe17f626b877e91eadae49c16`

Next: seed3 shards1/2/3/5/6/7 finish asynchronously. Old in-memory wrappers on shards1/3 may still need CPU-only merge recovery after exit. All seed3 work has started; do not restart it. Once eight seed3 receipts validate, append seed3 only when its mean9 strictly exceeds the selected frozen seed142 mean9, preserving prior choices on ties. Formal 1000 ZIP/upload/email remains subject to user approval.

## 2026-09-03 03:36 CST — seed3 shard1 complete

Shard1 base and VLM completed 125/125; the old wrapper exited with the already diagnosed four-row merge error. Fresh `/proc` checks confirmed no target wrapper or child, and the original per-work lock was acquired. CPU-only `--merge-only` exited 0. Verified all three seed3 terminal CSVs for shards0/1/4: **375 unique episodes**; no GPU scores recomputed.

- Shard1 CSV SHA: `ea7cc9d208b8bf2f2504258d99a3222ebc30730c4069ae885c3b06cf254d5875`
- Shard1 receipt SHA: `104d1767fcbe43430a8141cc41051d37872d7bc02e1860d5e5c470069670d1ea`

At 03:35 the five remaining unique shards were active: shard2 GPU1 VLM 52/125, shard3 GPU6 VLM 119/125, shard5 GPU0 VLM 25/125; shard6 GPU2 and shard7 GPU4 detection/tracking, compute PIDs827492/827558 at 90%/94% utilization. Every compute PID and project wrapper was checked through `/proc` for user, parent, process group, command and cwd. GPU3/5 are free, but there is no unclaimed unique GPU work. External GPU7 unchanged and untouched. Frozen seed142 receipt and prediction SHA reverified; disk143GiB available. No official score, final package or new email was produced.

## 2026-09-03 03:57 CST — seed3 reaches 500 completed rows

Shard3 base/VLM completed 125/125; its old wrapper exited with the known merge-count error. Verified exact target process absence and exclusive original claim, then CPU-only merge exited0. All four completed seed3 CSVs (shards0/1/3/4) were re-read: **500 unique episodes × nine metrics**, matching terminal hashes. No GPU work rerun.

- Shard3 CSV SHA: `e2963f96375e22e1a62abce1262a8c5040fc1a949f7fc064d9bd75552cef5f6c`
- Shard3 receipt SHA: `98b0d3e9ee17f446e68ccef114815d67c63ad48916d2c181f69825cba56154a4`

At 03:56, remaining shard2 GPU1 VLM61/125, shard5 GPU0 VLM34/125, shard6 GPU2 tracking99% and shard7 GPU4 tracking98%; process lineage verified through `/proc`. GPU3/5/6 have no remaining unclaimed work. All four active wrappers already use the fixed merge implementation. External GPU7 unchanged; disk143GiB. Frozen seed142 receipt/predictions unchanged. Next completion should self-write terminal receipts; recover only on a freshly verified failure, never merely because a GPU looks idle.

## 2026-09-03 06:29 CST — seed3 reaches 625 completed rows

Shard2 self-completed with the fixed wrapper (`merge_only=false`), without recovery or GPU rescoring. Re-read five terminal CSVs (shards0/1/2/3/4): **625 unique episodes × nine generated-only metrics**. For newly completed shard2, independently verified all 125 source video SHA values, generation receipt structure (declared/decoded121, 640x480, black0), parent/seed/manifest binding, exact manifest-to-CSV identities and nine finite scores.

- Shard2 CSV SHA: `f0e1d6dae577b51a8954844a8e2c30dc8f1452e8884491707b42bf218d368547`
- Shard2 terminal SHA: `346522b4e7606cf604c08a6162be5393278f4cb8dccac3ea44fd0ea867efe805`
- Shard2 source receipt SHA: `06a413fae4d3331a2400e1abc5856ea1b6064c5f84378d7383da26c35dd4c72b`

GPU1 released. Remaining shard5 on GPU0: VLM101/125 raw progress, wrapper535066/compute749196. Shards6/7 finished detection/tracking and entered six-dimension base evaluation on GPU2/GPU4: wrappers792083/792223, compute982281/983415, respectively1478MiB. All compute and wrapper process lineage rechecked via `/proc`. GPU1/3/5/6 are free with no unclaimed unique work; external GPU7 remains untouched. Disk142GiB available. Frozen seed142 receipt and predictions remain hash-identical. No final seed3 route, official score, package, upload or email produced this tick.

## 2026-09-03 07:10 CST — all base phases complete; remaining three shards in VLM

Shards6/7 finished their base phases and their existing wrappers automatically started VLM. No restart, duplicate launch or GPU rescoring. Each base receipt has SHA `c2724751797db8c6a7eed1e625a2c3eb9838eeaa91c054540ae4cc806a0cef37`; these minimal legacy receipts still contain `rows=4`, not proof of final125-row coverage. The fixed terminal merger will independently enforce manifest-sized coverage when VLM finishes.

- Shard5 GPU0: wrapper535066/compute749196; VLM119/125 raw JSONs.
- Shard6 GPU2: wrapper792083/new VLM compute1013598; raw3/125,19402MiB.
- Shard7 GPU4: wrapper792223/new VLM compute1012573; raw4/125,19402MiB.

Every compute and wrapper user/PPID/PGID/cmdline/cwd was freshly verified through `/proc`. Seed2 remains1000 unique completed rows; seed3 remains625, all completed CSV hashes and nine finite metrics revalidated. All source shards still contain125 videos. Frozen seed142 receipt/predictions and wrapper hashes unchanged. External GPU7 untouched; disk142GiB. No new terminal route, package, upload or email.

## 2026-09-03 07:30 CST — seed3 reaches 750 completed rows

Shard5 self-completed (`merge_only=false`) and wrapper535066/compute749196 exited. No recovery or rescoring was needed. Re-read all six completed seed3 CSVs (shards0–5), validating750 distinct episode IDs, nine finite metrics and terminal-bound CSV hashes. Independently verified shard5's125 source video SHA values, declared/decoded121 frames,640x480,black0, frozen parent/official commit,seed3, manifest SHA and exact manifest/video/CSV identity coverage.

- Shard5 CSV SHA: `4f2c33143eb685c0518d24b14ce8898efab32fc2640a00f08563881b662f8322`.
- Shard5 terminal SHA: `3e9215e7c54c6f63a4c8b2800c08797b1fdf15b6bea555693477c582f8bcebe1`.
- Shard5 source receipt SHA: `e00b0925bc3895973bca83f67cec7526bb8a9304ce7d9ebd2dad34b5017d6bbe`.

Only shards6/7 remain: GPU2 wrapper792083/compute1013598 and GPU4 wrapper792223/compute1012573, respectively12/125 and12/125 VLM raw JSONs. `/proc` lineage and UUIDs verified. GPU0 released; no unclaimed unique GPU work exists. Seed2 remains1000 complete; frozen seed142 receipt/predictions and scoring wrapper hashes unchanged. External GPU7 untouched; disk142GiB. Final seed3 routing awaits both full terminal receipts; no package/upload/email produced.

## 2026-09-03 09:28 CST — CPU seed3 append prepared and deployed

Added only the separate CPU MVP `tools/worldarena_freeze_seed1423_mvp.py` and its narrow rule tests; the frozen seed142 script, route and active scoring workers are unchanged. New script SHA256: `57398a5c2f01fd1895a573511d5bbe49b78634e1dca52b81c4a9a9749ce9406d`; test SHA256: `3c39a7ce9204318b729873beb4678920e1f69c9c17f5b3cc36c7964f6b1aadb1`.

Eight new tests first failed because the implementation was absent, then passed. They exercise comparison against only the frozen selected seed (not the best of all old seeds), strict improvement, ties, weaker candidates, invalid input, missing final receipts and incomplete receipts. Local new+old route tests:12 PASS; remote existing FlowWAM Python environment:8 new tests PASS; local Python compile PASS. Deployment SHA values were independently read back. No new GPU dependencies or framework were introduced.

Remote `--check-inputs` exited2 with exactly the two missing shard6/7 terminal receipts; `routing/seed1423-frozen-v1` remains absent. This proves the missing-input boundary only, **not successful full1000 execution**. Once all eight terminal receipts exist, run the check first, then the same script without flags once. It validates frozen prior hashes,1000 unique IDs, all source receipt/manifest/video bindings and nine finite metrics before writing the merged seed3 CSV, final predictions and receipt. Rule: seed3 replaces only if its mean9 strictly exceeds the previously selected frozen seed142 mean9; ties retain that exact prior video. Output is write-once; it neither packages nor submits.

Live09:28: seed2 remains1000 terminal rows, seed3 remains750 terminal rows; shard6 VLM checkpoint64/125, shard7 checkpoint62/125. At09:24 all14 complete CSVs and hashes were rechecked; both source seeds still have125 videos per shard; frozen142 hashes unchanged. GPU2/4 workers remain healthy and unchanged; GPU7 external Track2 remains untouched. Free disk141.7GiB.

The preceding read-only utilization audit found a real coarse-shard long tail: GPU0/1/3/5/6 are idle while two125-video VLM loops finish. An already claimed shard does not imply no possible parallelism. However, active workers build their missing-video list only once and rewrite aggregate/receipt after each video; helpers must not share their live output directories. Seven-card remaining-only redistribution would require first preparing isolated subsets and a single-writer recovery merge, then a fresh exact project-process handoff. This has **not** been implemented or launched; no worker was stopped this turn. Do not confuse CPU route readiness with GPU redistribution readiness.

## 2026-09-03 11:50 CST — seed3 reaches 875 completed rows

Shard6 self-completed with the fixed wrapper (`merge_only=false`); wrapper792083 and its GPU workload exited, releasing GPU2. No restart, recovery or rescoring. Re-read seven seed3 terminal CSVs, verifying875 unique episodes and bound CSV hashes. Independently verified shard6's125 source video SHA values, source receipt structure (declared/decoded121,640x480,black0), frozen parent/official commit/seed/manifest and exact125 CSV identities with nine finite scores.

- Shard6 CSV SHA: `ca0188729a77b5668cf1af9ceeec237f56af7360db88925afe17af3f5af1da77`.
- Shard6 terminal SHA: `0cdaa50ab2198b038ab28b1435a74364d5a9be177ae0420a9b4dcca2d97f0192`.
- Shard6 source receipt SHA: `a3eaddb46ea911cc0b0783f34800ab48416b6feec7392517a2cfa761d1ebe8e1`.

Only shard7 remains, at122/125 VLM checkpoint/raw files at11:50; wrapper792223/compute1012573 on GPU4 retain verified project lineage. All other project GPUs are idle, with no unclaimed unique score work. External GPU7 is unchanged and untouched. Seed2 remains1000 terminal; frozen142 and CPU append script hashes are unchanged. Final1423 directory is still absent. Await the last terminal, then validate inputs and run the prepared CPU append once; do not infer full1000 readiness from raw progress.

User-facing target communicated at10:45:12:15 scoring merge/selection freeze,13:00 structural acceptance and result ledger; formalZIP/upload/anonymous readback/email target16:00 only if user confirms final version by13:00. These are estimates, not a waiver of validation or submission authorization. No formal package, upload or email created.

## 2026-09-03 12:04 CST — all scoring complete; final route frozen and source decode accepted

Shard7 self-completed with the fixed wrapper and exited. Seed2 and seed3 now each have1000 terminal unique rows. Shard7 terminalSHA=`279871c5163a9516f086d7f634715449f45d7526972182f869761b2216b44d81`, CSV SHA=`5ef401f49a3c5fd9261dbb63d4365d73605f44542970e16d5b21f95432ae77da`, source receiptSHA=`6899c86146ab3d1951dc3f31f51b63633d929d60739eb365b4645a944c04a23d`. All project GPUs released; externalGPU7 untouched.

Verified freeze script/helper hashes, ran `--check-inputs` successfully with no output, then ran the prepared CPU freeze once into its previously absent directory. Independently recomputed1000 choices and checked each selected videoSHA. Finalcounts seed1=191,seed4=178,seed2=230,seed3=401. FinalreceiptSHA=`4100b3758097165977aab7f36af9743875585d6b632acf5644e24ee18c02913f`; predictionsSHA=`4b80f62d979cfd6e3517a9f037602a11168a25451c9fe62a9f7805343956b936`; seed3CSV SHA=`e0eb2aa0535db14f10150ef6089a4c6b478811a584ade8af5a04938aab4bc4f2`.

Existing submission probe fully decoded all1000 frozen sources using8 CPU threads in46.6seconds: declared=decoded121,640x480,black0,allSHA match,failures0. **76 sources require episode-specific trimming before formal delivery;924 already meet the frame cap.** Sources were not modified and no formal package/upload/email was created. Final ZIP acceptance remains pending user confirmation, trims and package verification; do not describe source acceptance as an already compliant formal ZIP.

Result ledger: `deliverables/track1-seed1423-final-selection-20260903.md`. Detailed source audit and76-item trim plan: `deliverables/track1-seed1423-source-audit-20260903.json`. Official testfull15 remains unknown; dev+0.32855 is only a point estimate.
