# Frozen routing: submission-format holdout verification

## Scope approved September 3

Run the frozen P0 and four-seed routing on previously unused, real-GT samples using the final submission video processing. This is verification, not another hyperparameter search. No new training, seed values, selector, formal ZIP, upload or email is authorized by this approval.

## Why the previous number is insufficient

The fresh40 result 68.0508 is reproducible but belongs to a repeatedly used development set and an 81-frame resampling protocol. Final source videos have 121 frames, with episode-specific tail trimming still required. The old number is neither a final-file test nor an official test1000 score. Its 30/40 paired-win statement included a shared set-level JEPA change: excluding JEPA, the change versus seed142 is 7 wins, 10 losses and 23 ties.

## Locked supplemental sample

Four tasks, four samples each; candidate indices10–49 sorted by SHA256 of `wa2-frozen-route-holdout16-20260903:{task}:{index}`. No quality or difficulty filtering.

| Task | Source episodes | Native GT frames |
| --- | --- | --- |
| open_microwave | 23,21,33,38 | 467,433,457,425 |
| press_stapler | 26,21,33,12 | 124,116,119,116 |
| put_object_cabinet | 41,20,32,21 | 266,272,261,266 |
| stack_bowls_three | 16,28,17,36 | 439,490,446,480 |

Lineage audit found no use of these samples in this project's scanned training, tuning, model-selection or scoring records (13,485 experiment metadata files,10,190 supplementary files, plus six large manifests). These are **episode-disjoint, not task-disjoint**: the tasks were used in development. Exposure in the Official parent's original pretraining is unknown. This is an auditable project-history claim, not proof that no unrecorded use ever occurred.

## Protocol

1. Freeze source identities, source hashes and first non-empty seen instruction. Generate the existing seeds1/4/2/3:64 raw candidates, no added seed values.
2. Apply the exact frozen generated9 routing, without reading GT metrics. Also retain the original seed1/4 selector output as paired baseline.
3. Preserve native121 frames/FPS; only trim tails exceeding matching GT length using final submission encoding. Never call the old121→81 staging. Encode full realGT, not81 resampled frames.
4. Score all15 metrics. Apply normalized per-video matched-GT caps to Dynamic/Flow/Motion Smoothness only.
5. Recompute JEPA on each whole selected16-video set. Report corrected14 paired changes/CI separately from set-level JEPA and corrected15 point estimates; do not treat shared JEPA as16 independent improvements.
6. Report verification outcome without tuning again on this holdout. Public evaluation compatibility is not an organizer result or a rank prediction.

The official main commit is b17a5b86cd38e54eabe699a848c701a829a216b5. A read-only audit compared26 core scorer/processing/JEDi/aggregation Python and shell blobs against the deployed7b3feee108427bee3380064bb5154970ed7468b5: all26 match. Third-party dependency/weight identity and the organizer's private matching/aggregation runtime remain unverified. The public per-video cap note is in official commit6607c2f8214723445a437ab7cbead6efc868ff74.

## Official pretest feedback

Gmail account clusternlh@gmail.com was searched with `in:anywhere from:worldarenav2@outlook.com` during this verification. Two received threads were found: August31 standard-ZIP request and September1 announcement/reminder. No reply scoring or accepting the September2 first100 pretest was found. No new mail was sent. The one pretest remains SENT, official evaluation UNKNOWN.

## Execution

Remote experiment root: `/data/di/worldarena2_track1_20260815/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/holdout16-nativecap-20260903`.

Input preparation and native-package/generated9 adapters are in progress. No new holdout score is available yet. GPU7 belongs to the external Track2 service and must remain untouched. Existing P0, seed1423 test1000 selection and historical submissions are preserved.
