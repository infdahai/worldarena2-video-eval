# Track1 four-seed routing — frozen selection, 2026-09-03

## Outcome at 12:04 CST

Both new seeds have 1000 completed generated-only nine-metric scores. The final whole-video route is frozen and independently verified across all 1000 unique episodes. No new model training, generation, rescoring, frame blending, formal package, upload or email was performed in this completion step.

| Selected seed | Videos |
| --- | ---: |
| 1 | 191 |
| 4 | 178 |
| 2 | 230 |
| 3 | 401 |
| Total | 1000 |

The exact rule preserves the frozen seed1/4/2 incumbent unless seed3 mean9 is strictly higher than that incumbent's mean9. Ties retain the prior video. The original seed1/4/2 decisions were not reoptimized. Independent readback checked every choice, prior path preservation, source SHA and output SHA; no GT or full15 was used for selection.

## Source acceptance, not final ZIP acceptance

All 1000 selected source videos were fully decoded using the existing submission probe, with 8 CPU threads and no GPU use. All have declared=decoded=121 frames,640x480,black_frames=0 and matching frozen SHA. Runtime46.6seconds; failures0.

The existing episode frame-cap list has 1000 unique positive entries. **76 source videos exceed their episode cap and must be trimmed before formal packaging;924 already fit.** No videos have been trimmed in this step. Thus the source integrity audit passes, but the final submission's frame-cap/ZIP contract is not yet complete. The detailed76-episode trim plan is in `track1-seed1423-source-audit-20260903.json` alongside this document. After user confirmation, write trimmed copies only, preserve frozen sources, then decode/check all final outputs and the ZIP.

## Frozen artifacts

Remote root: `/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/test1000-routing-mvp/routing/seed1423-frozen-v1`.

- `seed1423-frozen.complete.json` SHA256: `4100b3758097165977aab7f36af9743875585d6b632acf5644e24ee18c02913f`.
- `test1000-seed1423-predictions.jsonl` SHA256: `4b80f62d979cfd6e3517a9f037602a11168a25451c9fe62a9f7805343956b936`.
- `seed3-generated9.csv` SHA256: `e0eb2aa0535db14f10150ef6089a4c6b478811a584ade8af5a04938aab4bc4f2`.
- CPU freeze script SHA256: `57398a5c2f01fd1895a573511d5bbe49b78634e1dca52b81c4a9a9749ce9406d`.
- Episode cap list SHA256: `742e17a8de0a1046b0118b7d3966ed5d1004294b1cdbb8357e7a8175fa7c718e`.
- Existing decode probe script SHA256: `e7ad465b3c56f29c5955451e639da4af6df92532f51fb84f04740ebd4260073f`.

## Score conclusion and remaining delivery

The same-scope fresh40 development comparison is67.7223→67.8720→68.0508 EWM-equivalent points, about+0.32855 versus original seed1/4. This is a development-set point estimate without a confidence interval. **Official test1000 full15 and rank remain unknown.** Generated9 is a selection proxy, not an official score.

All project GPU scoring is finished. Protected GPU7 external Track2 remains untouched. StableP0,seed142 and historical submissions remain unchanged. The sole100-example pretest has already been sent and must not be repeated.

Next boundary requires final-version confirmation:76 trims→final1000 decode/size/SHA checks→README plus HZ-World ZIP structure/CRC→fixed-revision upload→anonymous exact-revision download/SHA/ZIP check→formal email. Target today16:00 if confirmed by13:00; deadline September4 24:00 UTC+8. Sending and official receipt/evaluation are separate statuses.
