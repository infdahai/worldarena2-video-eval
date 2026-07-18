# WorldArena 2.0 Track 1 OSCAR baseline

Minimal hybrid baseline for the official 1000-episode Track 1 test split.
Each joint14 trajectory is temporally resampled to one 81-frame dual-arm
Aloha skeleton video and passed to the public OSCAR-2B checkpoint together
with the first frame and instruction.

The remote setup entry is `scripts/bootstrap.sh`; it installs the pinned
CUDA 12.6 environment, downloads the official inputs and OSCAR checkpoint,
and prepares the 81-frame controls. It then starts `scripts/campaign.sh`,
which waits until all eight
GPUs have at least 22 GB free, runs a three-episode smoke gate, generates all
1000 MP4 files with resumable workers, validates the archive, and publishes a
public Hugging Face dataset repository. If the server is not logged into
Hugging Face, the packaged campaign pauses at `WAITING_FOR_HF_AUTH` without
losing generated results.
