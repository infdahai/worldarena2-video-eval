# v11 WorldArena-Balanced Bimanual Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the single-GPU Stage-A v11 persistent left/right closed-loop arm controller without adding object, coordination, depth, semantic, or native-Wan trainables.

**Architecture:** A frozen `clean-gated-step10` Wan parent is executed through an explicit block loop. After blocks 6, 16, and 24, physically separate left/right slot streams deterministically gather local visual tokens, read visual state, update persistent per-time slots, and write support-masked residuals back to Wan tokens. Per-arm counterfactual binding and visual-read-dependent EEF supervision train only the sparse controllers.

**Tech Stack:** Python 3.11, PyTorch/BF16, Wan2.2 TI2V-5B, activation checkpointing, NumPy cache inputs, pytest, Bash, single RTX 4090 GPU6.

**Spec:** `docs/superpowers/specs/2026-08-19-v11-worldarena-balanced-bimanual-controller-design.md`

## Global Constraints

- Parent is immutable `clean-gated-step10`; v10 relation modules are absent.
- Initial topology is exactly `CUDA_VISIBLE_DEVICES=6`, world size 1, microbatch 1.
- Stage A uses optimizer2060/audit20/dev20 with zero overlap; official test is inaccessible.
- Controller points are exactly post-block 6, 16, and 24.
- Slots are `(B,21,4,384)` per arm; latent0 is exact zero and no cross-time attention exists.
- Left/right modules and parameters are physically disjoint.
- Wan, raster parent, T5, VAE, and all non-controller modules are frozen.
- Stage-A trainable parameter count must be within the reviewed 40M-80M range.
- All persistent output is under `/data/di/worldarena2_track1_20260815/runs/v11-worldarena-balanced-bimanual-world-controller`.
- Production allocated and reserved CUDA peaks must each remain below 22 GiB.
- No Object Stream, Coordination Stream, V-JEPA, depth, instruction router, position/velocity objective, native delta V/O, or clean-5k work is included.

---

## File Structure

New focused modules:

- `src/worldarena_baseline/wan_v11_state.py`: slot packing, arm content, counterfactual construction, immutable state types.
- `src/worldarena_baseline/wan_v11_sparse.py`: deterministic tube top-K, gather, and support-masked scatter.
- `src/worldarena_baseline/wan_v11_controller.py`: per-arm read/update/write stage and three-stage controller.
- `src/worldarena_baseline/wan_v11_model.py`: frozen-parent model, explicit Wan block loop, functional slot checkpointing, trainable whitelist.
- `src/worldarena_baseline/wan_v11_objective.py`: per-arm FM energy, EEF loss, sequential pairwise-gradient helper, calibration.
- `src/worldarena_baseline/wan_v11_training.py`: optimizer groups, samples-seen schedule, checkpoint and gate contracts.
- `src/worldarena_baseline/wan_v11_audit.py`: step100/500/2060 audit metrics and fail-closed decisions.
- `src/worldarena_baseline/wan_v11_sync_closure.py`: exact recursive runtime source receipt.
- `scripts/train_wan_v11_bimanual.py`: preflight, smoke, train, audit entry point.
- `scripts/validate_wan_v11_sync_closure.py`: closure CLI.
- `scripts/run_wan_v11_gpu6.sh`: guarded phase launcher.
- Corresponding `tests/test_wan_v11_*.py` files contain CPU/Torch and static launcher regressions.

Existing v10 data/cache readers are reused read-only. No v10 production module is modified unless a failing compatibility test proves a minimal shared fix is necessary.

---

### Task 1: Immutable slot and counterfactual contracts

**Files:**
- Create: `src/worldarena_baseline/wan_v11_state.py`
- Create: `tests/test_wan_v11_state.py`

**Interfaces:**
- Consumes: v10 relation-cache tensors `anchored_se3`, `velocity`, `uv`, `gripper`, `arm_present`, `support`, and fixed destination slots.
- Produces: `ArmActionContent`, `BimanualCondition`, `BimanualSlots`, `pack_initial_slots()`, and `build_counterfactual()`.

- [ ] **Step 1: Write failing packing and isolation tests**

```python
def test_pack_initial_slots_has_exact_time_contract():
    condition = fixture_condition(batch=2)
    slots = pack_initial_slots(condition, width=384)
    assert slots.left.shape == (2, 21, 4, 384)
    assert torch.count_nonzero(slots.left[:, 0]).item() == 0
    assert torch.count_nonzero(slots.right[:, 0]).item() == 0
    assert slots.destination_time.tolist() == [list(range(21))] * 2

def test_left_and_right_tokenizers_share_no_parameters():
    tokenizer = BimanualSlotTokenizer(width=384)
    left = {id(p) for p in tokenizer.left.parameters()}
    right = {id(p) for p in tokenizer.right.parameters()}
    assert left and right and left.isdisjoint(right)
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_state.py -q`

Expected: collection fails because `wan_v11_state` does not exist.

- [ ] **Step 3: Implement immutable state types and tokenization**

```python
@dataclass(frozen=True)
class BimanualSlots:
    left: Tensor
    right: Tensor
    destination_time: Tensor

@dataclass(frozen=True)
class BimanualCondition:
    left: ArmActionContent
    right: ArmActionContent
    support: Tensor          # (B,21,2,H,W)
    arm_present: Tensor      # bool (B,21,2)
    destination_time: Tensor # int64 (B,21)

def pack_initial_slots(
    tokenizer: BimanualSlotTokenizer,
    condition: BimanualCondition,
) -> BimanualSlots:
    left = tokenizer.left(condition.left)
    right = tokenizer.right(condition.right)
    left[:, 0].zero_()
    right[:, 0].zero_()
    return BimanualSlots(left, right, condition.destination_time)
```

The implementation validates finite tensors, exact shapes, bool presence,
fixed destination slots 0 through 20, and zeroes every absent-arm slot.

- [ ] **Step 4: Add counterfactual leakage tests**

Test wrong-left/right preserve destination time, presence, anchor, support, and
mask byte-for-byte; only motion content changes. Test active-arm-null retains
presence/anchor/opening and zeroes relative motion/gripper delta. Test SE(3)
re-anchoring against an analytic homogeneous-transform oracle.

- [ ] **Step 5: Implement `build_counterfactual()` and run GREEN**

```python
def build_counterfactual(
    condition: BimanualCondition,
    kind: Literal["wrong-left", "wrong-right", "active-arm-null"],
) -> BimanualCondition:
    if kind == "wrong-left":
        replacement = _reanchor_motion(condition.right, condition.left)
        return replace(condition, left=replacement)
    if kind == "wrong-right":
        replacement = _reanchor_motion(condition.left, condition.right)
        return replace(condition, right=replacement)
    if kind == "active-arm-null":
        left_active = condition.left.motion_active.any(dim=1)
        right_active = condition.right.motion_active.any(dim=1)
        if torch.equal(left_active, right_active):
            raise ValueError("active-arm-null requires exactly one active arm")
        return replace(
            condition,
            left=_active_null(condition.left) if left_active.all() else condition.left,
            right=_active_null(condition.right) if right_active.all() else condition.right,
        )
    raise ValueError(f"unsupported v11 counterfactual: {kind}")
```

`_reanchor_motion(source, destination)` copies only source relative motion and
gripper delta into a dataclass replacement of the destination arm while all
identity, anchor, presence, support, and phase fields remain unchanged.
`_active_null(arm)`
retains anchor, absolute opening, presence, and masks while zeroing relative
translation/rotation/image motion/gripper delta.

Run: `PYTHONPATH=src pytest tests/test_wan_v11_state.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/worldarena_baseline/wan_v11_state.py tests/test_wan_v11_state.py
git commit -m "feat: add v11 arm slot contracts"
```

---

### Task 2: Deterministic sparse tube gather and masked scatter

**Files:**
- Create: `src/worldarena_baseline/wan_v11_sparse.py`
- Create: `tests/test_wan_v11_sparse.py`

**Interfaces:**
- Consumes: visual `(B,L,3072)`, grid sizes `(B,3)`, and support `(B,21,2,H,W)`.
- Produces: `SparseTubeSelection(indices, valid, weights)`, gathered `(B,21,2,48,3072)`, and support-masked visual residuals.

- [ ] **Step 1: Write RED tests for deterministic selection**

```python
def test_select_tube_tokens_is_stable_and_pads():
    support = torch.zeros(1, 21, 2, 3, 4)
    support[0, 3, 0, 1, 2] = 1
    selected = select_tube_tokens(support, max_tokens=48)
    assert selected.indices.shape == (1, 21, 2, 48)
    assert selected.valid[0, 3, 0].sum().item() == 1
    assert selected.indices[0, 3, 0, 0].item() == 6

def test_counterfactual_content_does_not_change_selection():
    assert torch.equal(select_tube_tokens(correct.support).indices,
                       select_tube_tokens(wrong.support).indices)
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_sparse.py -q`

- [ ] **Step 3: Implement stable top-K and gather**

Use support score descending and flattened index ascending as the tie break.
Reject grids other than exact `(21,H,W)` and sequence lengths other than
`21*H*W`. Padding indices are zero but `valid=false`.

- [ ] **Step 4: Test and implement scatter invariants**

Cover support-outside exact zero, absent-arm zero, overlap addition, padded
token exclusion, and no cross-time writes.

```python
def scatter_arm_residual(
    local: Tensor,
    selection: SparseTubeSelection,
    support: Tensor,
    *,
    sequence_length: int,
) -> Tensor:
    batch, time, count, channels = local.shape
    spatial = support.shape[-2] * support.shape[-1]
    if sequence_length != time * spatial:
        raise ValueError("sequence length differs from support grid")
    output = local.new_zeros((batch, sequence_length, channels))
    for latent_time in range(time):
        index = selection.indices[:, latent_time] + latent_time * spatial
        valid = selection.valid[:, latent_time, :, None].to(local.dtype)
        source = local[:, latent_time] * valid
        output.scatter_add_(1, index[:, :, None].expand_as(source), source)
    flat_support = support.reshape(batch, sequence_length, 1).to(local.dtype)
    return output * flat_support
```

- [ ] **Step 5: Run GREEN and commit**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_sparse.py -q`

```bash
git add src/worldarena_baseline/wan_v11_sparse.py tests/test_wan_v11_sparse.py
git commit -m "feat: add v11 sparse tube operations"
```

---

### Task 3: Physically separate closed-loop controller stages

**Files:**
- Create: `src/worldarena_baseline/wan_v11_controller.py`
- Create: `tests/test_wan_v11_controller.py`

**Interfaces:**
- Consumes: visual tokens, `BimanualSlots`, fixed `SparseTubeSelection`, and arm presence.
- Produces: `ControllerOutput(visual, slots, visual_reads, direct_writes, telemetry)`.

- [ ] **Step 1: Write RED tests for read/update/write behavior**

Test exact left/right parameter disjointness, visual-values-zero makes
`DeltaC_visual` exact zero, absent-arm zero, slot0 zero after every stage,
overlap residual addition, and same-time-only behavior.

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_controller.py -q`

- [ ] **Step 3: Implement one arm stage**

```python
class ArmControllerStage(nn.Module):
    def forward(
        self,
        visual: Tensor,
        slots: Tensor,
        selection: SparseTubeSelection,
        present: Tensor,
        *,
        force_zero_visual_values: bool = False,
        force_zero_controller: bool = False,
    ) -> ArmStageOutput:
        local = gather_visual_tokens(visual, selection)
        read = self.read(slots, local, selection.valid,
                         zero_values=force_zero_visual_values)
        updated = self.update(slots, read)
        local_delta = self.write(local, updated, selection.valid)
        direct = scatter_arm_residual(
            local_delta,
            selection,
            selection.support,
            sequence_length=visual.shape[1],
        )
        direct = direct * self.gate.float().to(direct.dtype)
        return ArmStageOutput(updated, read, direct)
```

Read/write attention is only over four slots and at most 48 local visual
tokens within the same latent time. Left/right instances are separate modules.

- [ ] **Step 4: Implement the three-stage container**

```python
class BimanualControllerStage(nn.Module):
    def forward(
        self,
        visual,
        slots,
        selection,
        present,
        *,
        force_zero_visual_values=False,
        force_zero_controller=False,
    ):
        left = self.left(
            visual,
            slots.left,
            selection.for_arm(0),
            present[:, :, 0],
            force_zero_visual_values=force_zero_visual_values,
            force_zero_controller=force_zero_controller,
        )
        right = self.right(
            visual,
            slots.right,
            selection.for_arm(1),
            present[:, :, 1],
            force_zero_visual_values=force_zero_visual_values,
            force_zero_controller=force_zero_controller,
        )
        return ControllerOutput(
            visual=visual + left.direct_write + right.direct_write,
            slots=BimanualSlots(left.slots, right.slots, slots.destination_time),
            visual_reads=(left.visual_read, right.visual_read),
            direct_writes=(left.direct_write, right.direct_write),
            telemetry={"left": left.telemetry, "right": right.telemetry},
        )
```

Each of stages 6, 16, and 24 owns independent left/right modules.

- [ ] **Step 5: Add gradient and residual telemetry tests**

Assert tokenizer/updater/read/write/gate/EEF families receive gradients;
cross-arm action gradients are exactly zero; FP32 gates remain FP32 after
`.to(torch.bfloat16)`; direct-write RMS is recorded per stage/arm.

- [ ] **Step 6: Run GREEN and commit**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_controller.py -q`

```bash
git add src/worldarena_baseline/wan_v11_controller.py tests/test_wan_v11_controller.py
git commit -m "feat: add v11 closed-loop arm stages"
```

---

### Task 4: Explicit Wan block loop and frozen-parent integration

**Files:**
- Create: `src/worldarena_baseline/wan_v11_model.py`
- Create: `tests/test_wan_v11_model.py`
- Create: `tests/test_wan_v11_model_static.py`

**Interfaces:**
- Consumes: Wan backbone, frozen parent adapter, three controller stages, action raster/support, and `BimanualCondition`.
- Produces: `ParentPlusBimanualControllerWan.forward`, EEF predictions, ablation telemetry, and `v11_trainable_parameter_names()`.

- [ ] **Step 1: Pin the upstream Wan forward structure in a static test**

The test verifies the project Wan source still performs patch embedding,
time/context embedding, sequential blocks, head, and unpatchify in the expected
order. A source drift fails before GPU execution.

- [ ] **Step 2: Write RED numerical-parent and functional-state tests**

Use a small 27-block fixture. At zero gates, v11 output must equal the frozen
parent. Assert stages run after blocks 6/16/24 and slots are passed as explicit
arguments/results, not stored on modules.

- [ ] **Step 3: Implement the explicit forward loop**

Copy the minimum project-Wan forward orchestration into the wrapper while
calling the original frozen embedding, block, head, and unpatchify modules.

```python
for index, block in enumerate(self.backbone.blocks):
    if index in self.parent_injection_points:
        visual = visual + parent_residuals[index].to(visual.dtype)
    visual = block(visual, **block_kwargs)
    if index in self.controller_points:
        visual, left, right = self._checkpointed_stage(
            index, visual, left, right, selection, present
        )
```

`_checkpointed_stage` calls `torch.utils.checkpoint.checkpoint` with tensor
inputs/outputs `(visual,left,right)` and `use_reentrant=False`.

- [ ] **Step 4: Implement exact freeze and whitelist**

```python
def v11_trainable_parameter_names(model: ParentPlusBimanualControllerWan) -> set[str]:
    expected_ids = {id(p) for p in model.controller.parameters()}
    expected_ids |= {id(p) for p in model.eef_heads.parameters()}
    actual = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if {id(p) for p in actual.values()} != expected_ids:
        raise ValueError("v11 trainable whitelist drift")
    return set(actual)
```

Reject v10 relation modules, native trainables, parameter aliases, shared
left/right IDs, non-FP32 gates, and trainable count outside the final declared
40M-80M range.

- [ ] **Step 5: Test checkpoint recompute and sequential forwards**

Compare checkpointed versus non-checkpointed outputs/gradients and ensure a
rejected/failed forward leaves no state that affects the next forward.

- [ ] **Step 6: Run GREEN and commit**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_model.py tests/test_wan_v11_model_static.py -q`

```bash
git add src/worldarena_baseline/wan_v11_model.py tests/test_wan_v11_model.py tests/test_wan_v11_model_static.py
git commit -m "feat: integrate v11 controller with frozen Wan"
```

---

### Task 5: Per-arm objectives and exact sequential pairwise gradients

**Files:**
- Create: `src/worldarena_baseline/wan_v11_objective.py`
- Create: `tests/test_wan_v11_objective.py`

**Interfaces:**
- Consumes: unreduced FM target/prediction, fixed correct arm masks, EEF visual reads/targets/validity, and correct/wrong energies.
- Produces: per-arm energies, binding loss, EEF loss, gradient calibration, and sequential-gradient execution helpers.

- [ ] **Step 1: Write RED masking tests**

Test left/right masks cannot cancel, invalid arm-events are excluded, empty
eligibility fails closed, and fixed correct masks are reused for negatives.

- [ ] **Step 2: Implement per-arm energy and EEF loss**

```python
def per_arm_fm_energy(
    prediction: Tensor,
    target: Tensor,
    arm_masks: Tensor,
    valid: Tensor,
) -> Tensor:  # (B,2), NaN prohibited; invalid marked separately
    token_error = (prediction.float() - target.float()).square().mean(dim=1)
    weighted = token_error[:, :, None] * arm_masks.float()
    denominator = arm_masks.float().sum(dim=(-1, -2, -3))
    if torch.any(valid & (denominator <= 0)):
        raise ValueError("eligible arm energy has an empty mask")
    return weighted.sum(dim=(-1, -2, -3)) / denominator.clamp_min(1)

def visual_read_eef_loss(
    visual_reads: Mapping[int, Tensor],
    targets: Tensor,
    valid: Tensor,
) -> Tensor:
    losses = []
    expanded_valid = valid[:, :, :, None, None]
    if not expanded_valid.any():
        raise ValueError("visual EEF loss has no eligible labels")
    for logits in visual_reads.values():
        error = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), reduction="none"
        )
        losses.append((error * expanded_valid).sum() / expanded_valid.sum())
    return torch.stack(losses).mean()
```

- [ ] **Step 3: Write the simultaneous-versus-sequential gradient oracle**

On a tiny dropout-free fixture, compare gradients from the simultaneous
pairwise softplus loss with:

```python
0.5 * softplus((E_correct.detach() - E_wrong) / tau)
0.5 * softplus((E_correct - E_wrong.detach()) / tau)
```

The sum of parameter gradients must match the simultaneous oracle within
FP32 tolerance.

- [ ] **Step 4: Implement calibration**

Calibrate binding and EEF gradient norms on actual write projections to ratios
`0.5` and `0.2` of FM. Reject zero/non-finite norms and unreviewed lambdas.
Emit immutable `wan-v11-loss-calibration/1` JSON content.

- [ ] **Step 5: Run GREEN and commit**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_objective.py -q`

```bash
git add src/worldarena_baseline/wan_v11_objective.py tests/test_wan_v11_objective.py
git commit -m "feat: add v11 per-arm objectives"
```

---

### Task 6: Optimizer, checkpoint, and audit gates

**Files:**
- Create: `src/worldarena_baseline/wan_v11_training.py`
- Create: `src/worldarena_baseline/wan_v11_audit.py`
- Create: `tests/test_wan_v11_training.py`
- Create: `tests/test_wan_v11_audit.py`

**Interfaces:**
- Consumes: model whitelist, calibration, lineage, optimizer state, per-arm audit episodes, GPU smoke telemetry.
- Produces: optimizer/scheduler, checkpoint payloads, resume validation, and step100/500/2060 gate receipts.

- [ ] **Step 1: Write RED optimizer/lineage tests**

Require non-empty tokenizer/updater/read/write/gate/EEF optimizer groups,
complete AdamW state, exact LR/weight decay, topology GPU6/world-size1, and all
lineage hashes. Reject foreign parent/replay/data/source/calibration and partial
optimizer states.

- [ ] **Step 2: Implement training contract**

Use checkpoint steps `(100,500,2060)`, samples-seen equal completed optimizer
updates, 50-step warmup, cosine decay to 0.2 of base LR, global clip norm 1.0,
and fresh optimizer from the clean parent. Base LRs are explicit by family and
frozen in the checkpoint contract.

- [ ] **Step 3: Write and implement audit metric tests**

Metrics report eligible/invalid/wins/win-rate/mean/median independently for
left/right, overall swap, bimanual/crossing, locality, forced-zero controller,
forced-zero visual values, phase, routing, FM, background, and broken outputs.

- [ ] **Step 4: Implement exact gate boundaries**

Encode the spec's health, direction, early-stop conjunction, and step2060 hard
gate. Test every threshold on both sides and prove one passing metric cannot
hide another failing metric.

- [ ] **Step 5: Run GREEN and commit**

Run: `PYTHONPATH=src pytest tests/test_wan_v11_training.py tests/test_wan_v11_audit.py -q`

```bash
git add src/worldarena_baseline/wan_v11_training.py src/worldarena_baseline/wan_v11_audit.py tests/test_wan_v11_training.py tests/test_wan_v11_audit.py
git commit -m "feat: add v11 training gates"
```

---

### Task 7: Trainer, source closure, and GPU6 launcher

**Files:**
- Create: `src/worldarena_baseline/wan_v11_sync_closure.py`
- Create: `scripts/validate_wan_v11_sync_closure.py`
- Create: `scripts/train_wan_v11_bimanual.py`
- Create: `scripts/run_wan_v11_gpu6.sh`
- Create: `tests/test_wan_v11_scripts.py`
- Create: `tests/test_wan_v11_runtime.py`

**Interfaces:**
- Consumes: all Task 1-6 APIs and existing validated v10 data/cache readers.
- Produces: `dry-run`, `preflight`, `smoke`, `train100`, `audit100`, `train500`, `audit500`, `train2060`, and `audit2060` phases.

- [ ] **Step 1: Write RED static launcher tests**

Assert exact formal root, GPU6 visibility, source validation before Python/CUDA,
no T5/VAE imports in the trainer, no Object/Coordination/V-JEPA/Depth modules,
and continuation only from passing gated checkpoints.

- [ ] **Step 2: Implement exact source closure**

Build a recursive local-import closure rooted at the trainer and cache readers.
Reject missing, untracked, staged, modified, symlinked, dynamic-local-import,
and manifest-drift files. Emit canonical file SHA256 values and one closure
SHA256 before every GPU phase; compare it with the preflight receipt before
CUDA initialization.

- [ ] **Step 3: Implement trainer modes**

The trainer validates all paths under `/data/di`, parent SHA, zero leakage,
2060/20 manifests, replay coverage through step2060, caches, source closure,
topology, and resume checkpoint before `torch.cuda.set_device(0)`.

Training performs the exact sequential flow:

```text
correct no-grad reference
wrong grad half-binding backward; release
correct grad FM + EEF + half-binding backward
clip; optimizer.step; zero_grad
```

Append one JSONL row per completed step atomically enough to preserve completed
records across interruption. Checkpoints use partial + fsync + atomic replace.

- [ ] **Step 4: Implement production smoke**

Warm up, reset CUDA peaks, then run three complete optimizer iterations. Record
allocated/reserved/time per iteration and gradient maxima for every trainable
family plus zero frozen gradients. Fail at 22 GiB or any growth/error.

- [ ] **Step 5: Implement launcher**

`run_wan_v11_gpu6.sh` checks GPU6 has no compute PID, memory `<1024 MiB`, and
utilization `<10%` before every GPU phase. It never inspects or alters other
processes beyond read-only ownership checks. Dry-run starts no process.

- [ ] **Step 6: Run local GREEN and commit**

Run:

```bash
PYTHONPATH=src pytest tests/test_wan_v11_state.py tests/test_wan_v11_sparse.py tests/test_wan_v11_controller.py tests/test_wan_v11_model.py tests/test_wan_v11_model_static.py tests/test_wan_v11_objective.py tests/test_wan_v11_training.py tests/test_wan_v11_audit.py tests/test_wan_v11_scripts.py tests/test_wan_v11_runtime.py -q
bash -n scripts/run_wan_v11_gpu6.sh
git diff --check
```

```bash
git add src/worldarena_baseline/wan_v11_sync_closure.py scripts/validate_wan_v11_sync_closure.py scripts/train_wan_v11_bimanual.py scripts/run_wan_v11_gpu6.sh tests/test_wan_v11_scripts.py tests/test_wan_v11_runtime.py
git commit -m "feat: add guarded v11 gpu6 run"
```

---

### Task 8: Formal remote verification and bounded run

**Files:**
- Create: `reports/2026-08-19-v11-stagea-run.md`
- Remote artifacts: `/data/di/worldarena2_track1_20260815/runs/v11-worldarena-balanced-bimanual-world-controller/`

**Interfaces:**
- Consumes: clean committed source closure and the existing formal runtime/data.
- Produces: verified source receipt, preflight, smoke, checkpoints, audits, and final report.

- [ ] **Step 1: Verify local closure and scoped diff**

Run the closure validator and confirm no unrelated dirty file enters the v11
closure. Record commit SHA and closure SHA.

- [ ] **Step 2: Sync only the reviewed closure**

Use the explicit closure file list to copy source into
`/home/huazhi/nlh/baseline`. Rebuild the closure remotely and require exact
digest equality before any GPU command.

- [ ] **Step 3: Run remote zero-skip focused tests**

Run with `/data/di/worldarena2_track1_20260815/venv_reuse/bin/python` and
`PYTHONPATH=/home/huazhi/nlh/baseline/src:/home/huazhi/nlh/Wan2.2`. Any failed
or unexpected skipped v11 test blocks smoke.

- [ ] **Step 4: Run dry-run and preflight**

Verify the launcher phases, immutable inputs, source closure, calibration,
trainable inventory, deterministic initialization, and GPU6 availability.

- [ ] **Step 5: Run production smoke**

Require three iterations, allocated/reserved `<22 GiB`, all trainable-family
gradients nonzero, frozen gradients zero, and no GPU ownership conflict.

- [ ] **Step 6: Run bounded gates**

Run train/audit at step100, then step500. Continue to step2060 only if the
health/direction contracts permit it. Never bypass a failed required gate.

- [ ] **Step 7: Write and verify the report**

Include data/lineage hashes, architecture/parameter count, smoke telemetry,
training curves, per-arm denominators, every gate reason, checkpoint/artifact
SHA256 values, incidents/restarts, GPU release state, and the explicit decision
on whether Stage B is authorized.

```bash
git add reports/2026-08-19-v11-stagea-run.md
git commit -m "docs: report v11 stagea run"
```
