# v16 Step50 Seed4 LoRA-Scale Gate Implementation Plan

> **For Codex:** Execute this plan in order. Preserve all existing seed1, seed4, v16-step50, and v16-step100 artifacts; every new candidate must have its own lineage and receipt.

**Goal:** Decide whether v16 step50 combined with seed4 and a reduced inference-only LoRA scale can beat the current Official Stage-1 seed4 champion without reopening training or expanding the experiment tree.

**Architecture:** Keep the trained adapter contract frozen at rank/alpha 8/8 and add a separate, explicitly receipted inference scale. Generate three Breadth20 candidates sequentially on physical GPU6 under the central lock, evaluate them against the existing matched Official+seed4 baseline, and permit exactly one full dev-clean50 evaluation only when the frozen fast gate passes. In parallel, compute the seed1/seed4 oracle upper bound from existing episode CSVs and audit whether the deployed service performs a second lossy encode.

**Tech Stack:** Python 3.10, pytest, Bash/flock, FlowWAM/DiffSynth LoRA hotload, pinned WorldArena evaluator, ffmpeg/ffprobe, JSON receipts.

---

## Frozen scope and stop rules

- No training, step200, SeedVR2 repair, seed5+, flow/Text-CFG/prompt grids, Motion FT, new loss, or new action architecture.
- GPU6 only for new generation/evaluation; never GPU7 and never another user's process.
- Candidates: seed4 with inference scale `0.50`, `0.75`, and `1.00` from the exact v16 step50 adapter.
- Independent lineages: `breadth20-step50-seed4-scale050`, `...-scale075`, `...-scale100`.
- Completion is receipt-, artifact-, hash-, count-, decode-, and metric-validation gated, never PID-gated.
- Fast gate: black=0; valid count not lower; trajectory and JEPA not materially worse; arithmetic mean of the requested fast metrics is at least `0.003` above matched Official+seed4.
- Only the best passing scale may run full dev-clean50. If none passes, permanently freeze Official+seed4. If the full 15-metric mean does not exceed `0.65865975675`, permanently freeze Official+seed4.

## Task 1: Compute the zero-generation oracle upper bound

**Inputs:**
- `/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0400/dev-clean-50/csv_results/aggregated_results.csv`
- `/data/di/worldarena2_track1_20260815/official_track1_eval/checkpoints/step-0600/dev-clean-50/csv_results/aggregated_results.csv`

1. Validate identical 50 `Video_ID` values and 15 finite official metrics, excluding blank `Action Following`.
2. Compute each episode's 15-metric mean for seed1 and seed4, select the maximum, and average the 50 maxima.
3. Record base means, oracle mean, uplift over seed4, and seed win counts under the automation directory.
4. Apply the selector rule: `<0.003` no selector; `>0.01` selector research may be valuable; otherwise no investment.

## Task 2: Add an independent inference-only LoRA scale with TDD

**Files:**
- Modify: `tests/test_flowwam_official_pipeline.py`
- Modify: `scripts/run_flowwam_official_stage1.py`

1. Add a dry-run test proving rank/alpha remain 8/8 while `--lora-scale 0.5` is separately recorded.
2. Run the focused test and observe the expected failure because `--lora-scale` does not exist.
3. Add `--lora-scale` restricted to `0.5`, `0.75`, or `1.0`.
4. Pass the explicit scale to `pipe.load_lora`; do not derive it by changing training alpha.
5. Record scale in the dry-run payload and final receipt.
6. Run focused and full FlowWAM official/v16 tests.

## Task 3: Create receipt-gated scale queue

**Files:**
- Add: `scripts/run_flowwam_v16_step50_seed4_scale_generation.sh`
- Add: `scripts/run_flowwam_v16_step50_seed4_scale_queue.sh`
- Test: `tests/test_flowwam_v16_scripts.py`

1. Test that the launcher pins seed4, flow20, step50 adapter, GPU6, central lock, explicit scale, and independent output path.
2. Generate scales sequentially, skipping only a candidate with a validated receipt and exactly 20 videos.
3. Validate every video fully decodes to 121 frames at 640x480 with black=0 and record SHA256.
4. Stop the queue on the first unrepairable launcher/runtime failure; never launch duplicates.

## Task 4: Evaluate the expanded fast metric gate

**Requested metrics:** Trajectory, coverage, black, Instruction, Interaction, Perspectivity, Image, Aesthetic, Photometric, JEPA.

1. Reuse the existing Official+seed4 Breadth20 videos, but produce a matched validated baseline metric receipt if one does not already exist.
2. Stage each candidate in an independent evaluator lineage; never overwrite step-0100/0200 or dev-clean50 receipts.
3. Run bounded GPU6 evaluator phases and validate every per-video row is finite and error-free.
4. Compute candidate-minus-baseline deltas and the same metric-set arithmetic mean.
5. Select at most one best passing candidate. Record the decision even when all candidates fail.

## Task 5: Run at most one full dev-clean50 evaluation

1. Only a fast-gate winner may generate seed4 dev-clean50 at its selected scale.
2. Require exactly 50 fully decoded 121-frame 640x480 MP4s with hashes and generation receipt.
3. Run the pinned complete 15-metric evaluator in a new lineage.
4. Replace the champion only if the finite 15-metric mean exceeds `0.65865975675`; otherwise freeze Official+seed4.

## Task 6: Audit final-service encoding

1. Trace the service response path from generated MP4 to returned artifact.
2. Use `ffprobe` and source inspection to distinguish evaluator-only 121-to-81 staging from deployment encoding.
3. If the service returns the generator MP4 unchanged, record no second lossy encode and stop.
4. Only if a second lossy encode is confirmed, compare 5-10 existing clips with a single high-quality encode and stop unless Image/Aesthetic/Photometric/JEPA show a material benefit.

## Task 7: Consolidate evidence

Update and re-read:
- `reports/2026-08-24-wan-action-v1-v15-total-experiment-report.md`
- `reports/2026-08-20-v15-flowwam-balanced-campaign.md`
- `reports/2026-08-23-worldarena-track1-final-score-report.md`

Include exact receipts, artifact counts/hashes, oracle result, encoding audit, scale comparisons, fast/full gates, final winner, stopped branches, and remaining deployment action.

## Self-review checklist

- The plan changes inference scale, not training alpha or adapter weights.
- The final comparison is against seed4, not the obsolete seed1-only incumbent.
- Baseline and candidates use identical Breadth20 inputs and evaluator contracts.
- No candidate can overwrite a prior lineage.
- There is no path to step200, SeedVR2, extra seeds, or a second full clean50 candidate.
- Service encoding is audited separately from evaluator staging encodes.
