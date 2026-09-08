#!/usr/bin/env python3
"""Use the existing scorer for two isolated native16 packages, approved budget only."""
from pathlib import Path
import runpy
import shutil
import sys

GIB = 1024 ** 3
A = Path('/data/di/worldarena2_track1_20260815')
H = A / 'runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/ranker-fresh16-native-20260903'
ROOTS = tuple(A / f'official_track1_eval/checkpoints/step-{step}/dev-clean-50' for step in (1232, 1233)) + (H,)


def check_budget(*, used, estimate, free):
    if min(used, estimate, free) < 0:
        raise ValueError('negative disk accounting')
    if used + estimate > 20 * GIB:
        raise RuntimeError('native holdout shared 20GiB budget exceeded')
    if free - estimate < 100 * GIB:
        raise RuntimeError('native holdout 100GiB reserve would be breached')
    return 100 * GIB


if __name__ == '__main__':
    from worldarena_baseline import official_track1_eval as official
    for flag, allowed in (('--checkpoint', {'1232', '1233'}), ('--split', {'dev-clean-50'}), ('--expected-count', {'16'})):
        if flag not in sys.argv or sys.argv[sys.argv.index(flag) + 1] not in allowed:
            raise ValueError(f'unsupported native holdout argument: {flag}')
    official.SUPPORTED_DEVELOPMENT_CHECKPOINTS = (1232, 1233)
    original = official.require_eval_disk_budget

    def approved_budget(**kwargs):
        used = sum(p.stat().st_size for root in ROOTS for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        reserve = check_budget(used=used, estimate=kwargs['estimated_additional_bytes'], free=shutil.disk_usage(A).free)
        return original(**kwargs, reserve_bytes=reserve)

    official.require_eval_disk_budget = approved_budget
    runpy.run_path('/home/huazhi/nlh/baseline/scripts/run_official_track1_eval.py', run_name='__main__')
