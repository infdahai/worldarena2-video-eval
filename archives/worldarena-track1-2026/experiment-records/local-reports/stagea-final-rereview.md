# V7 Stage-A final re-review — `9941a47`

**Verdict: BLOCK — one Important source-closure bypass remains.**

`9941a47` closes the originally reported launch-order issue:

- Every non-dry-run launcher phase calls `validate_source_closure` before `run`/`torchrun` (`scripts/run_wan_se3_probe_v7.sh:81-109`).
- For smoke, train, and audit, the trainer calculates the current closure digest and verifies the preflight receipt before `torch.cuda.set_device` and `dist.init_process_group` (`scripts/train_wan_se3_probe_v7_fsdp.py:621-645`).
- The launcher independently compares a reused receipt with the freshly written closure receipt (`scripts/run_wan_se3_probe_v7.sh:59-79`).

## Important — declared source closure is not a transitive runtime closure

`validate_sync_closure` only discovers direct imports of the two Python entrypoints (`src/worldarena_baseline/wan_v7_sync_closure.py:23-27, 100-106`). The tracked manifest consequently omits runtime modules that are imported by listed files:

- `wan_v7_training.py` imports `wan_v6_training` (`src/worldarena_baseline/wan_v7_training.py:23`), but `wan_v6_training.py` is absent from `source_inputs/wan-v7-stagea-sync-closure.json`.
- `wan_cached_dataset.py` imports `robotwin_action_cache` (`src/worldarena_baseline/wan_cached_dataset.py:12`), which in turn imports `action_audit`, `action_raster`, `robotwin_manifest`, and `skeleton` (`src/worldarena_baseline/robotwin_action_cache.py:14-18`); none is covered by the closure manifest.
- `wan_probe.py` imports `environment_manifest` (`src/worldarena_baseline/wan_probe.py:9-13`), which is also omitted.

Therefore a committed or working-tree change to any omitted dependency does not change `closure_sha256`; a preflight receipt can be reused while executing changed runtime source. If the selective sync is intended to copy only the claimed closure, the same omissions can instead make the remote import fail. The test only asserts direct-entrypoint coverage (`tests/test_wan_v7_sync_closure.py:35-46`), so it cannot catch this path.

Required closure: recursively resolve local `worldarena_baseline` imports from both entrypoints (including relative imports), reject every missing transitive dependency, and include the complete resolved set in the digest. Add a regression test that changes an indirect dependency and proves validation/digest binding fails.

## Local verification

`env UV_CACHE_DIR=/private/tmp/worldarena-v7-uv-cache uv run pytest -q tests/test_wan_v7_sync_closure.py tests/test_wan_v7_scripts.py tests/test_wan_v7_training.py tests/test_wan_v7_lineage.py` passed: `12 passed`.

`bash -n scripts/run_wan_se3_probe_v7.sh` and `uv run python -m py_compile` for the changed Python entrypoints passed.

No remote, GPU, or training action was performed.
