# FlowWAM Object-Response Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Every task requires an implementation agent, a fresh spec reviewer, and a fresh code-quality reviewer before integration.

**Goal:** Add a causal Object-Response Bridge (ORB) to the frozen Official FlowWAM parent, prove that robot-only flow produces localized non-robot object response in decoded RGB, and promote candidates only with the official corrected15 protocol.

**Architecture:** Training-only full-scene flow and contact pseudo-labels supervise a small response predictor. An immutable pure-flow token source writes a zero-initialized residual into RGB tokens at pre-registered DiT blocks; the new branch never writes into flow tokens. The original Official FlowWAM main path remains frozen and bidirectional. Parent distillation protects static and semantic quality, while an optional hard-static compositor is evaluated as an independent inference candidate. All GPU work is UUID-claimed, asynchronous, and subordinate to external-task ownership checks.

**Tech Stack:** Python 3.11, PyTorch, HDF5, OpenCV/official flow dependencies, pytest, official WorldArena 15-metric scorers, `flock`, `nvidia-smi`.

**Approved spec:** `docs/superpowers/specs/2026-08-30-flowwam-object-response-bridge-design.md`

## Frozen boundaries

- Official source commit: `f06fa46042e97738c6619c868f1097be6749d48d`.
- Official parent checkpoint SHA256: `e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`.
- Released data revision: `YixiangChen/FlowWAM_RoboTwin@506c4e014f7dbd291e7d7683c79fc685dd3e2714`.
- Dataset consumers must use `BoundSplit`, `load_bound_split`, and `open_bound_row_file`; direct opening of row paths is forbidden.
- `targeted15_proposal.py`, `flowwam_targeted_forward.py`, and `flowwam_targeted_losses.py` stay immutable historical regression evidence. Proposal/3 V1/V2/I1/I2/C1/C2/R1, old LoRA launchers, Q0, and old R1 may not be launched.
- The new ORB branch is strictly one-way Flow-to-RGB. The frozen Official main path remains bidirectional and must not be described as globally asymmetric.
- Generated-only nine metrics may select seed/candidate artifacts. Only corrected15 with per-episode GT caps may promote a model.
- Never stop, signal, modify, or attach to `/data/fjy`, `/data/whn`, or another task. A GPU is launchable only after two UUID-specific idle audits at least ten minutes apart and claim/flock checks.

---

### Task 1: Freeze the ORB campaign contract

**Files:**
- Create: `source_inputs/flowwam_orb_campaign.template.json`
- Create: `src/worldarena_baseline/flowwam_orb_contract.py`
- Create: `tests/test_flowwam_orb_contract.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class OrbVariant:
    candidate_id: str
    block_indices: tuple[int, ...]
    response_rank: int
    bridge_rank: int
    loss_coefficients: Mapping[str, Decimal]

@dataclass(frozen=True)
class OrbContract:
    official_source_commit: str
    parent_sha256: str
    dataset_binding_sha256: str
    split_sha256: Mapping[str, str]
    variants: tuple[OrbVariant, ...]
    budgets: Mapping[str, int]

def canonical_orb_contract_bytes(payload: Mapping[str, object]) -> bytes: ...
def load_orb_contract(path: Path) -> OrbContract: ...
def validate_orb_contract(contract: OrbContract, *, bound_split: BoundSplit) -> None: ...
def finalize_orb_contract(
    template_path: Path, *, bound_split: BoundSplit, output_path: Path,
) -> OrbContract: ...
```

- [ ] Write failing tests that reject floats, unknown fields, proposal/3 family names, an unpinned full Official commit, parent/dataset/split hash drift, invalid block indices, and budget expansion.
- [ ] Implement restricted canonical JSON, Decimal coefficient parsing, exact schema validation, self-pin SHA validation, and write-once finalization. The checked-in template is explicitly `blocked_data_binding` with null binding/empty split hashes; it cannot validate or execute until `finalize_orb_contract` fills them from a reviewed `BoundSplit` and writes the run-root contract.
- [ ] Run `baseline/.venv/bin/python -m pytest -q tests/test_flowwam_orb_contract.py tests/test_flowwam_randomized500_v2.py`.
- [ ] Run `python3 -m py_compile src/worldarena_baseline/flowwam_orb_contract.py` and `git diff --check`.
- [ ] Obtain fresh spec and quality reviews; integrate only a PASS commit.

### Task 2: Audit native supervision resolution independently

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_resolution.py`
- Create: `scripts/audit_flowwam_orb_resolution.py`
- Create: `tests/test_flowwam_orb_resolution.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class OrbResizeContract:
    source_height: int
    source_width: int
    output_height: int
    output_width: int
    interpolation: str

def audit_native_supervision(
    *, bound_split: BoundSplit,
    rows: Sequence[Mapping[str, object]],
    official_resize: OrbResizeContract,
) -> OrbResolutionReceipt: ...
```

- [ ] Test `native`, `bicubic-upsampled`, `unknown`, forged dimensions, HDF5/path hash drift, and ancestor-symlink rejection.
- [ ] Read all row payloads only through `open_bound_row_file` and emit per-task provenance plus input/output SHA256.
- [ ] Make the receipt disable Image/Aesthetic detail loss for low-resolution supervision without disabling contact/response learning.
- [ ] Run the focused pytest, py_compile, diff-check, and fresh two-stage review.

### Task 3: Build pure-tensor object-response pseudo-targets

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_targets.py`
- Create: `tests/test_flowwam_orb_targets.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class OrbPseudoTargets:
    robot_support: Tensor
    contact_support: Tensor
    response_flow: Tensor
    response_support: Tensor
    occlusion: Tensor
    confidence: Tensor
    persistent_state: Tensor
    hard_static: Tensor
    ignore: Tensor

def build_orb_pseudo_targets(
    *, rgb: Tensor, scene_forward: Tensor, scene_backward: Tensor,
    robot_flow: Tensor, gripper: Tensor, eef_xy: Tensor,
    config: OrbTargetConfig,
    multiview: OrbMultiViewEvidence | None = None,
) -> OrbPseudoTargets: ...

def audit_target_partition(targets: OrbPseudoTargets) -> OrbTargetAudit: ...
```

- [ ] Test coordinate alignment, forward/backward occlusion, non-robot residual extraction, contact timing, persistent response, low-confidence ignore, and disjoint hard-static support.
- [ ] Test multi-view input as confidence multiplication only; it must not synthesize response direction.
- [ ] Test that robot/contact/response/occlusion pixels can never be relabeled hard-static.
- [ ] Run focused pytest, numerical gradient-independent tests, py_compile, diff-check, and fresh review.

### Task 4: Materialize immutable pseudo-label shards

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_pseudolabels.py`
- Create: `scripts/build_flowwam_orb_pseudolabels.py`
- Create: `tests/test_flowwam_orb_pseudolabels.py`

**Interfaces:**

```python
def build_pseudolabel_shard(
    *, bound_split: BoundSplit, row_indices: Sequence[int],
    config: OrbTargetConfig, output_root: Path,
) -> OrbShardReceipt: ...

def load_orb_pseudolabel(
    row: Mapping[str, object], receipt: OrbShardReceipt,
) -> OrbPseudoTargets: ...
```

- [ ] Test write-once shards, duplicate work keys, interrupted temp files, receipt/output tampering, final-holdout access, and cross-split replay.
- [ ] Bind each shard to repo/revision/archive/episode manifest/camera/flow-code/config/output hashes.
- [ ] Allow `model-train` rows only; reject selector-fit/dev/final rows during training materialization.
- [ ] Run focused tests and fresh review before any pseudo-label GPU job.

### Task 5: Implement the pinned Official FlowWAM adapter

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_official.py`
- Create: `tests/test_flowwam_orb_official.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class OfficialOrbRuntime:
    pipe: object
    flow_stream: nn.Module
    shared_blocks: tuple[nn.Module, ...]

def build_pinned_orb_runtime(
    *, official_source: Path, checkpoint: Path,
    local_model_path: Path, device: torch.device,
) -> OfficialOrbRuntime: ...

def install_orb_bridge(
    runtime: OfficialOrbRuntime, bridge: OrbBridge,
    *, block_indices: tuple[int, ...],
) -> OrbBridgeHandles: ...

def audit_frozen_parent(runtime: OfficialOrbRuntime, handles: OrbBridgeHandles) -> FrozenParentAudit: ...
```

- [ ] Use CPU fake blocks to test exact pinned source shape/API checks, frozen parent parameters, and no unexpected trainable module.
- [ ] Capture `pure_flow_source = flow_tokens.detach()` immediately after flow patchification. Every ORB block reads this immutable source, not later flow tokens polluted by joint attention.
- [ ] Pool the immutable source per time step to at most 6x8 spatial K/V tokens; reject a full 37,200-token full-width attention path.
- [ ] Apply the bridge after the Official block FFN and before return; return the exact original flow tensor object.
- [ ] Test that a disabled/zero gate skips bridge execution and reproduces all parent tensors bit-for-bit.
- [ ] Keep the Official source checkout unmodified; install through the adapter/hook layer and fail closed on code-shape drift.
- [ ] Run focused tests and a remote read-only source audit before fresh review.

### Task 6: Implement the predictor and one-way bridge

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_model.py`
- Create: `tests/test_flowwam_orb_model.py`

**Interfaces:**

```python
class ObjectResponsePredictor(nn.Module):
    def forward(
        self, *, first_rgb_features: Tensor, instruction_tokens: Tensor,
        robot_flow_tokens: Tensor, gripper: Tensor, eef_xy: Tensor,
    ) -> OrbPrediction: ...

class AsymmetricFlowToRgbBridge(nn.Module):
    def forward(
        self, *, rgb_tokens: Tensor, pure_flow_tokens: Tensor,
        response_tokens: Tensor,
    ) -> Tensor: ...

class OrbBridge(nn.Module):
    def forward_block(
        self, block_index: int, rgb_tokens: Tensor,
        flow_tokens: Tensor, conditioning: OrbCondition,
    ) -> tuple[Tensor, Tensor]: ...
```

- [ ] Test shapes, zero initialization, exact flow identity, detached immutable flow K/V, RGB-only residual, task/gripper/eef conditioning, and no generic PEFT q/v target modules.
- [ ] Test response/locality masks at token resolution and deterministic serialization.
- [ ] Freeze four pre-registered bridges at width 256/eight heads; fail contract validation if a variant expands block count, width, heads, or pooled K/V shape.
- [ ] Run focused tests, autograd assertions that flow/parent receive no new gradients, py_compile, diff-check, and fresh review.

### Task 7: Implement ORB losses and the frozen-parent trainer

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_losses.py`
- Create: `src/worldarena_baseline/flowwam_orb_training.py`
- Create: `scripts/train_flowwam_orb.py`
- Create: `tests/test_flowwam_orb_losses.py`
- Create: `tests/test_flowwam_orb_training.py`

**Interfaces:**

```python
def orb_response_loss(
    *, prediction: OrbPrediction, targets: OrbPseudoTargets,
) -> OrbLossBreakdown: ...

def orb_prior_loss(
    *, candidate_x0: Tensor, parent_x0: Tensor,
    candidate_rgb: Tensor, parent_rgb: Tensor,
    targets: OrbPseudoTargets, frame0: Tensor,
    correct_task_fm: Tensor, mismatched_task_fm: Tensor,
) -> OrbLossBreakdown: ...

class OrbTrainer:
    def training_step(self, batch: OrbTrainingBatch) -> OrbTrainStep: ...
```

- [ ] Test Decimal coefficients from the ORB contract as the only coefficient authority.
- [ ] Test response-flow/contact/state losses only on confident response support and teacher preservation outside robot/contact/response support.
- [ ] Test frame0 exact invariance, mismatched-task margin, loss telemetry, finite gradients, and no evaluator/fresh40/test1000 fields in a training batch.
- [ ] Enforce a trainable-name whitelist for predictor, bridge projections, and explicit gates; freeze parent, VAE, text, and flow stream.
- [ ] Add 5-step mechanism-smoke CLI with hash-complete receipt, peak-memory receipt, and no promotion flag.
- [ ] Keep parent and ORB checkpoint files separate. Strictly load only ORB predictor/bridge/gate keys after verifying the original parent SHA; reject missing, extra, or parent-prefixed ORB keys.
- [ ] Require batch-1, 121-frame, 640x480 full forward/backward/optimizer peak allocation below 21.5 GiB; a reduced-resolution run is diagnostic only and cannot pass.
- [ ] Run focused tests, CPU fake smoke, fresh reviews, then a single legal-GPU memory smoke.

### Task 8: Prove causality in decoded RGB

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_gates.py`
- Create: `scripts/audit_flowwam_orb_mechanism.py`
- Create: `tests/test_flowwam_orb_gates.py`

**Interfaces:**

```python
def make_flow_counterfactuals(
    robot_flow: Tensor, *, swap_index: Tensor, shift: int,
) -> Mapping[str, Tensor]: ...

def measure_decoded_response(
    *, parent_rgb: Tensor, candidate_rgb: Tensor,
    targets: OrbPseudoTargets,
) -> DecodedResponseAudit: ...

def evaluate_orb_mechanism(...) -> OrbMechanismReceipt: ...
```

- [ ] Generate correct/reversed/swapped/time-shifted/zero-flow counterfactuals with unique work keys.
- [ ] Measure decoded object displacement direction, contact timing, response locality, static leakage, robot-versus-object localization, and frame0 identity.
- [ ] Require 121 declared/decoded frames, 640x480, black0, safe paths, and video SHA; internal loss or probe improvement cannot pass alone.
- [ ] Run focused tests and independent review before n8 generation.

### Task 9: Implement the independent hard-static compositor

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_compositor.py`
- Create: `scripts/run_flowwam_orb_compositor.py`
- Create: `tests/test_flowwam_orb_compositor.py`

**Interfaces:**

```python
def build_deployable_masks(...) -> OrbDeployMasks: ...

def compose_static_anchor(
    *, parent_rgb: Tensor, candidate_rgb: Tensor,
    first_frame: Tensor, masks: OrbDeployMasks,
    feather_radius: int,
) -> Tensor: ...
```

- [ ] Test exact zero delta on frame0 and robot/contact/response support, boundary-only feathering, occlusion exclusion, and idempotence.
- [ ] Test fail-closed behavior when masks are unavailable or uncertain.
- [ ] Require decoded trajectory/interaction/instruction/structure gates before full15 scoring.
- [ ] Keep this candidate independent from bridge-training claims and fresh-review it separately.

### Task 10: Wire ORB inference without changing the pinned source

**Files:**
- Modify: `scripts/run_flowwam_official_stage1.py`
- Create: `tests/test_flowwam_orb_inference.py`

- [ ] Add mutually exclusive `--orb-checkpoint` and `--orb-contract`; reject simultaneous LoRA and ORB flags.
- [ ] Expand `--physical-gpu` choices to all eight indices, while keeping UUID/process/claim/flock ownership validation mandatory.
- [ ] Load only contract/checkpoint SHA-matched predictor/bridge/gate state through the adapter.
- [ ] Emit bridge block points, frozen-parent audit, contract/checkpoint/input/output hashes, and zero-gate audit in Stage1 receipts.
- [ ] Run inference tests plus `tests/test_scripts.py`, py_compile, diff-check, and fresh review.

### Task 11: Freeze corrected15 promotion and budget accounting

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_campaign.py`
- Create: `scripts/evaluate_flowwam_orb_gate.py`
- Create: `tests/test_flowwam_orb_campaign.py`

**Interfaces:**

```python
def evaluate_orb_stage(
    *, stage: Literal["n8", "n24", "n64", "n128"],
    candidate: PairedCorrected15,
    baseline: PairedCorrected15,
) -> OrbPromotionDecision: ...

def claim_orb_budget(
    ledger: OrbBudgetLedger, request: OrbBudgetRequest,
) -> OrbBudgetClaim: ...
```

- [ ] Lock n24 `+0.002`, n64 `+0.003`, n128 `+0.004`, nonmotion12, primary, LCB90/95, protected deltas, and 76/128 wins exactly as the approved spec.
- [ ] Enforce three rounds, six candidates, 1600 raw videos, 800 full15 evaluations, 256 final reserve, and 384 GPU-hours atomically.
- [ ] Test that generated9 may select a deployable candidate but cannot promote it or read matched-GT/full15 during selection.
- [ ] Test duplicate work, stale claim, budget race, stage skipping, and official-test1000 access rejection.
- [ ] Run focused tests and fresh review.

### Task 12: Build the UUID-asynchronous queue and dry-run launcher

**Files:**
- Create: `src/worldarena_baseline/flowwam_orb_queue.py`
- Create: `scripts/run_flowwam_orb_worker.py`
- Create: `scripts/launch_flowwam_orb_8gpu.py`
- Create: `tests/test_flowwam_orb_queue.py`

- [ ] Define work keys over campaign/proposal/parent/code/data/split/pseudolabel/config/seed SHA values.
- [ ] Implement UUID-specific claims, exclusive flock, terminal-receipt double check, stale adjudication, and missing-only recovery.
- [ ] Require two read-only GPU audits at least ten minutes apart with physical index, UUID, compute PID, user, PPID, PGID, command, cwd, memory, and utilization.
- [ ] Treat every `/data/fjy`, `/data/whn`, non-project PID, live project parent/child, or conflicting claim as a hard per-UUID block; never stop an external process.
- [ ] Schedule ready work independently: pseudo-labels, bridge variants, prior ablation, compositor, generated9, structure, mechanism gate, and full15 scoring without a campaign barrier.
- [ ] Dry-run against synthetic `nvidia-smi`/`/proc` fixtures and prove GPU0 occupancy does not block legal GPU1-7 work.
- [ ] Run focused tests, py_compile, diff-check, and fresh review before remote deployment.

### Task 13: Execute the bounded asynchronous experiment

**Remote roots:**
- Campaign: `/data/di/worldarena2_track1_20260815/runs/flowwam-orb-20260904`
- Frozen fallback: `/data/di/worldarena2_track1_20260815/submission/releases/p0-dual-seed-parallel-r1/final-p0-r3`

- [ ] Verify the ORB contract, all code/data/pseudolabel hashes, disk reserve, and current external GPU ownership.
- [ ] Start each legal UUID independently after its cooldown; leave occupied UUIDs untouched.
- [ ] Produce response pseudo-label shards first where ready, launch reviewed 5-step smokes as soon as their inputs exist, and route completed videos immediately to structure and causal gates.
- [ ] Advance passing candidates n8 → n24 → n64 → n128 without waiting for sibling variants; close two-failure mechanisms with receipts.
- [ ] Score selected videos using generated9-first selection followed by corrected15/GT-cap aggregation; never use full15 oracle selection.
- [ ] Preserve P0 unless a final task/sample-disjoint holdout passes the frozen promotion contract.

### Task 14: Final integration and evidence lock

**Files:**
- Update: `docs/superpowers/specs/2026-08-30-flowwam-object-response-bridge-design.md`
- Update: `docs/superpowers/plans/2026-08-30-flowwam-object-response-bridge.md`
- Create: `docs/superpowers/reviews/2026-08-30-flowwam-orb-final-review.md`

- [ ] Run all ORB tests and historical Task2/Task3 regression tests.
- [ ] Verify every promotion receipt, input/output SHA, budget ledger, GPU claim, frame contract, and corrected15 calculation.
- [ ] Record rejected candidates and why they closed; do not hide negative experiments.
- [ ] Record the locked model or explicit P0 fallback. Do not package, upload, or send email without a separate user confirmation.

## Required final verification commands

```bash
baseline/.venv/bin/python -m pytest -q \
  tests/test_flowwam_randomized500_v2.py \
  tests/test_flowwam_orb_contract.py \
  tests/test_flowwam_orb_resolution.py \
  tests/test_flowwam_orb_targets.py \
  tests/test_flowwam_orb_pseudolabels.py \
  tests/test_flowwam_orb_official.py \
  tests/test_flowwam_orb_model.py \
  tests/test_flowwam_orb_losses.py \
  tests/test_flowwam_orb_training.py \
  tests/test_flowwam_orb_gates.py \
  tests/test_flowwam_orb_compositor.py \
  tests/test_flowwam_orb_inference.py \
  tests/test_flowwam_orb_campaign.py \
  tests/test_flowwam_orb_queue.py

python3 -m py_compile \
  src/worldarena_baseline/flowwam_orb_*.py \
  scripts/*flowwam_orb*.py

git diff --check
```

Execution is complete only when the final review is PASS and either a holdout-qualified ORB candidate is locked or the immutable P0 fallback is explicitly retained.
