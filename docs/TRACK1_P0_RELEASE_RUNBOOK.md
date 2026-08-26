# WorldArena 2.0 Track 1 P0 release runbook

## Status and boundary

The public model name is frozen as `HZ-World`. The full P0 release version is
`P0-DualSeed-Selector-v1`; the resource-constrained single-seed fallback uses
`Stage1-Seed4-v1`. These versions distinguish materially different inference
routes under the same public model name. The full release smoke is complete. It covers the immutable
P0 gates, a deterministic 1,000-episode paired plan, exact selector replay, one
official-input dual-seed generation, strict packaging, public Hugging Face
upload, and an anonymous proxy download with an identical archive SHA.

The smoke did **not** execute all 1,000 episodes and did **not** trigger the
official submission/email. Official test inputs remain final-inference-only and
must never be used to fit or revise the model, selector, features, thresholds,
or candidate set.

Aggregate receipt:

```text
/data/di/worldarena2_track1_20260815/submission/releases/p0-r1/track1-release-verification.complete.json
SHA256 bfccefb91925af8dcaa57a56405273f3a585f4cde177d8144153cd7b0cc07c73
```

## Frozen artifacts

```text
release config
  baseline/source_inputs/flowwam_p0_track1_release.json

remote release root
  /data/di/worldarena2_track1_20260815/submission/releases/p0-r1

final decision
  /data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/final-decision.complete.json
  SHA256 5eb9e43450bef05e24e8f8f600b5668f66971370b9c31e85422f63e272fad9cf

selector model
  /data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/selector/postgen/postgen-selector.model.json
  SHA256 4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa

policy
  /data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/selector/policy-selection.complete.json
  SHA256 aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7

official test receipt
  /data/di/worldarena2_track1_20260815/datasets/WorldArena2.0-official-track1/official-track1-dataset.complete.json
  SHA256 d6f66953860abca36b2b495ac96879924fa965f08761f38521386bc817b73eb1
```

The model uses exactly nine generated-only metrics for each seed and their
difference: Instruction Following, Interaction Quality, Perspectivity, Image
Quality, Aesthetic Quality, Photometric Consistency, Dynamic Degree, Flow
Score, and Motion Smoothness. Hidden GT features are rejected.

## Official video requirements versus the frozen release contract

The current official Track 1 submission guideline recommends MP4 videos at
`640x480` or higher and 24 fps. It does not mandate a universal fixed frame
count; generated length should align with the corresponding GT trajectory:

```text
https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/TRACK1_HF_SUBMISSION_GUIDELINE_EN.md
```

This release nevertheless freezes `121` frames, `640x480`, and 24 fps because
those are the released FlowWAM Stage-1 inference settings used for every
candidate and validated by the production smoke. The `121`-frame staging gate
is therefore a model/release lineage invariant, not a claim that the benchmark
requires every Track 1 model to output exactly 121 frames.

## Release CLI

The entrypoint is `baseline/scripts/run_track1_release.py`. Remote execution
uses the isolated runtime and code copy:

```bash
export RELEASE_ROOT=/data/di/worldarena2_track1_20260815/submission/releases/p0-r1
export RELEASE_PY=/data/di/worldarena2_track1_20260815/submission/venv/bin/python
export PYTHONPATH=/data/di/worldarena2_track1_20260815/submission/code
```

Freeze P0 and the official dataset lineage:

```bash
$RELEASE_PY -m worldarena_baseline.track1_release_cli freeze \
  --config "$RELEASE_ROOT/release-config.json" \
  --output "$RELEASE_ROOT/p0-release-frozen.complete.json"
```

Build the paired seed1/seed4 plan. The result must be 1,000 rows and 2,000
candidate videos:

```bash
$RELEASE_PY -m worldarena_baseline.track1_release_cli plan \
  --manifest /data/di/worldarena2_track1_20260815/datasets/WorldArena2.0-official-track1/episode-manifest.jsonl \
  --output "$RELEASE_ROOT/test1000-paired-plan.jsonl" \
  --receipt "$RELEASE_ROOT/test1000-paired-plan.complete.json" \
  --expected-count 1000 --shard-count 4
```

Each scored-candidate JSONL row passed to `select` must have this schema:

```json
{
  "episode_id": 1,
  "seed1": {
    "video": "/absolute/path/seed1/episode1.mp4",
    "video_sha256": "...",
    "features": {"Instruction Following": 0.0}
  },
  "seed4": {
    "video": "/absolute/path/seed4/episode1.mp4",
    "video_sha256": "...",
    "features": {"Instruction Following": 0.0}
  }
}
```

`features` must contain all nine frozen metric names, and only those names.
The command rejects missing videos, SHA drift, non-finite/out-of-range values,
schema drift, model drift, and policy drift:

```bash
$RELEASE_PY -m worldarena_baseline.track1_release_cli select \
  --model /data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/selector/postgen/postgen-selector.model.json \
  --policy /data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/selector/policy-selection.complete.json \
  --candidates "$RELEASE_ROOT/test1000-scored-candidates.jsonl" \
  --predictions "$RELEASE_ROOT/test1000-predictions.jsonl" \
  --receipt "$RELEASE_ROOT/test1000-selection.complete.json" \
  --expected-count 1000
```

Stage and package only after all 1,000 selected videos pass 121-frame,
640x480, black-frame, source SHA, and copied SHA checks:

```bash
$RELEASE_PY -m worldarena_baseline.track1_release_cli stage \
  --predictions "$RELEASE_ROOT/test1000-predictions.jsonl" \
  --staging "$RELEASE_ROOT/final-staging" \
  --metadata "$RELEASE_ROOT/release-metadata.json" \
  --expected-count 1000

$RELEASE_PY -m worldarena_baseline.track1_release_cli package \
  --staging "$RELEASE_ROOT/final-staging" \
  --archive "$RELEASE_ROOT/submission.tar.gz" \
  --receipt "$RELEASE_ROOT/submission.package.complete.json" \
  --expected-count 1000
```

The archive is deterministic and must contain exactly one `model_readme.md`,
one `videos/` directory, and `episode_000001.mp4` through
`episode_001000.mp4`, with no links, special files, or path traversal.

## Isolated Hugging Face configuration

Only the project-scoped credential is allowed:

```text
HF home:  /data/di/worldarena2_track1_20260815/submission/hf-home
env file: /data/di/worldarena2_track1_20260815/submission/project-hf.env
owner:    huazhi
mode:     0600
account:  clusternlh
proxy:    http://127.0.0.1:7890
```

Never print the token and never read or replace another project's global HF
credential. Load it only in the submission shell:

```bash
set -a
. /data/di/worldarena2_track1_20260815/submission/project-hf.env
set +a
```

The verified public smoke repository is
`clusternlh/worldarena-track1-p0-submission-smoke`, commit
`8165ebbaa2a178e38d9e7e5fae8897236466c60c`. Its archive SHA is
`83873237aad18f353a542b722092c744e63146a65ffa717791b48d094b54d609`.
Anonymous download was executed after unsetting `HF_TOKEN`, `HF_HOME`, and
`HF_TOKEN_PATH`, and returned the same SHA.

For the real submission, create a separate public dataset repository with a
non-smoke name only after all 1,000 rows and the final package receipt pass.
Do not reuse the smoke repository, do not upload partial outputs, and do not
send the official submission email without explicit user authorization.

## Verified smoke evidence

```text
P0 freeze receipt SHA          30875c2735c41803234a96b9478ab4b7db142c5459c680bf1dc86090dfdfb20f
test-1000 plan receipt SHA     19fdbf53eefcc5ff3fb98a902c09846c8a67b23faf0ce39808f0eeb279c60355
selector replay receipt SHA    bfd1e76679a6b0e81d836b0e18873922502d4edb5f8be28972f3a86f0b297ff0
seed1 generation receipt SHA   181cbe45d3be43fba2625c52cb55f09b8287f7a1d1577196301e6cace3a7e9fe
seed4 generation receipt SHA   2e683e520c5a920881a7ce10c726054b41c8be9506ffdfe19548b542962faddb
package receipt SHA            0ab58239e96a301aa48dfa101d5ee269c45cdb354130bcd12fedacbd3689aff4
HF upload receipt SHA          0988308a0362edacdbcc6cbb750133856b124fe2bcf5f957eec062edec79ce84
anonymous receipt SHA          9dd9b48fad2d2a5a413a65ae9bba10148c085ad1108b102d0a1c04595094f9ff
aggregate verification SHA     bfccefb91925af8dcaa57a56405273f3a585f4cde177d8144153cd7b0cc07c73
```

The exact selector replay reproduced predictions SHA
`102184cde847bcdf4d9fb7a4f47a43b5d6f1762b32af4b5490951da209655ea0`,
selection counts 27/23, raw mean `0.662720295147882`, and corrected mean
`0.6592060841461832`.

The official-input smoke generated episode 1 at both seeds with 121 frames and
640x480 resolution. GPU0/GPU4 exited normally. The aggregate validator also
re-opened and decoded both videos and rechecked their generator SHA values.

## Final go/no-go gate

Proceed to all 1,000 episodes only if:

1. The aggregate smoke receipt and frozen hashes above are unchanged.
2. GPU ownership and locks are checked immediately before every shard.
3. The feature scorer writes only the nine generated-only selector metrics.
4. All 2,000 candidate receipts exist before selection closes.
5. The selected 1,000 videos pass staging and deterministic package validation.
6. The final public HF repository is anonymous, ungated, and has exactly one
   root `.tar.gz`/`.tgz` archive.
7. The downloaded archive SHA equals the local package SHA.

Any drift is a stop condition, not permission to refit P0 on official test.

## Resource-constrained formal fallback

When only physical GPU0 is available, the fail-closed fallback runs the
immutable Official FlowWAM Stage-1 checkpoint with flow20 and seed4 for all
1,000 episodes. It reuses only videos that independently pass the frozen
121-frame, 640x480, black0 gate and records their SHA256 values. It does not
pretend to execute the two-candidate P0 selector.

```bash
bash scripts/run_track1_formal_seed4_submit.sh 0
```

The supervisor owns a physical-GPU UUID lock, refuses an occupied GPU, resumes
valid existing rows, stages exactly 1,000 videos, creates a deterministic
archive, uploads only the complete archive with the project-isolated HF token,
and anonymously downloads it again. Completion requires
`final-submission.complete.json`; PID presence is not completion.

Create this release-scoped sentinel when seed4 should finish generating but
must not upload before the final seed4-versus-P0 decision:

```bash
touch /data/di/worldarena2_track1_20260815/submission/releases/seed4-resource-constrained-r1/HOLD_BEFORE_UPLOAD
```

With the sentinel present, the supervisor releases GPU0 after generation and
exits before staging, packaging, repository creation, or upload. If no further
GPU capacity becomes available, remove the sentinel and resume the same script
to package the validated seed4 fallback. If seed1 capacity becomes available,
leave the sentinel in place, preserve the seed4 videos, complete seed1 and the
frozen P0 selector, and publish only the selected 1,000-video package.
