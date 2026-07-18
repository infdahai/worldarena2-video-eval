# WorldArena 2.0 Track 1 OSCAR baseline

Minimal zero-training inference baseline for the official 1000-episode Track 1 test split.
Each joint14 trajectory is temporally resampled to one 81-frame dual-arm
Aloha skeleton video and passed to the public OSCAR-2B checkpoint together
with the first frame and instruction.

The remote setup entry is `scripts/bootstrap.sh`; it installs the pinned
CUDA 12.6 environment, downloads the official inputs and OSCAR checkpoint,
and prepares the 81-frame controls. It then starts `scripts/campaign.sh` in a
PID-guarded user-space background process. By default the campaign waits for
eight GPUs with at least 22 GB free, runs a smoke gate and a length-stratified
gate, generates all 1000 MP4 files with resumable workers, validates the
archive, and publishes a public Hugging Face dataset repository. If the server
is not logged into Hugging Face, the packaged campaign pauses at
`WAITING_FOR_HF_AUTH` without losing generated results.

For a one-GPU, local-only run, use:

```bash
ROOT=/data/di/worldarena2_track1_baseline \
GPU_INDICES=0 SMOKE_COUNT=1 GATE_COUNT=20 NO_PUBLISH=1 \
bash "$ROOT/baseline/scripts/campaign.sh"
```

`GPU_INDICES` is a comma-separated GPU list and defaults to `0,1,2,3,4,5,6,7`.
Only the selected GPUs are checked for readiness; `MIN_FREE_MIB` (default
`22000`) and `MAX_UTIL_PERCENT` (default `10`) can be adjusted when needed.
