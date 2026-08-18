# V7 Single-GPU Opportunistic Probe Design

## Goal

Run the bounded Wan v7 SE(3) mechanism probe whenever GPU6 is free, without
waiting for the seven-card formal topology or contaminating its lineage.

## Scope

This is a separate experimental lineage, not a fallback representation of the
seven-rank result. It keeps the frozen parent, clean-1000 input, v7 SE(3)
condition, source closure, and step 10/25/50 mechanism gates. It changes only
the execution topology to one rank on physical GPU6.

## Contract

- `world_size=1`, `rank_mapping=[6]`, and an explicit
  `wan-action-v7-se3-single-gpu/1` contract.
- Replay, checkpoints, receipts, output roots, and source closure digest are
  all single-GPU-specific. A seven-rank checkpoint/replay is rejected, and a
  single-GPU checkpoint/replay is rejected by the seven-rank runner.
- GPU6 is the only permitted device. GPU0-5 and GPU7 are never selected,
  inspected for ownership only, and never stopped.
- Use the same frozen parent and gate-only parameter whitelist. There is no
  LoRA, adapter change, VAE/T5 training hot path, or Stage B behavior.
- A single-GPU production smoke runs three complete steps and records peak
  allocated/reserved memory and step time. Any OOM, non-finite loss/gradient,
  zero gate gradient, original-parameter gradient, or peak >=22 GiB fails.
- If smoke fails for memory, stop cleanly and retain the result; do not reduce
  resolution, frame count, or substitute a different model.
- All persistent artifacts stay below
  `/data/di/worldarena2_track1_20260815/runs/v7-se3-single-gpu/`.

## Execution

1. Reuse the already validated cached-only clean-1000 sidecars once cache is
   complete. The cache contract remains topology independent.
2. Build a deterministic one-rank replay from the trusted v6 replay using the
   same first 50 global records, selecting rank-6 records in each step.
3. Run preflight, smoke, train10, train25, audit25, then train50 only if the
   shared mechanism gate passes.
4. Record this lineage separately. Its gate result is evidence for the v7
   mechanism only; it cannot promote or resume the seven-rank lineage.

## Verification

- Unit tests reject cross-topology replay/checkpoint use and GPU lists other
  than exactly `CUDA_VISIBLE_DEVICES=6`.
- Remote: closure validation, focused PyTorch tests, one-card smoke, then the
  bounded 10/25/50 sequence.
