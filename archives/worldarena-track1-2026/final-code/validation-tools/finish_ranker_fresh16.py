#!/usr/bin/env python3
"""One-shot CPU GTcap + preregistered comparison; never aggregates or scores videos."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from worldarena_baseline import latest_wa2_scorer as scorer
from worldarena_baseline.official_track1_eval import FORMAL_CORE_METRIC_COLUMNS

A = Path('/data/di/worldarena2_track1_20260815')
H = A / 'runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/ranker-fresh16-native-20260903'
MODELS = {1232: 'RankerFreshP0Native16', 1233: 'RankerFreshPred14Native16'}
METRICS = list(FORMAL_CORE_METRIC_COLUMNS)
MOTION = scorer.LATEST_MOTION_COLUMNS
FIXED_SHA = {
    'manifest.jsonl': 'd661604840181036b5ddd8742654f8082c799012996344a38c790d8a394979d7',
    'input-contract.json': '73167b152951501041073400edbf423589694004b65b249de4dd57532949f3cc',
    'ranker-fit/model.json': '1409145a4efadfa7d274b2742a85318da6a56322472bc687e3f13844a7eb0fd5',
    'ranker-fit/fit.complete.json': '36937e59c7b8db59d40b8ba85af4936ca2a3952988d20864d5b0b770316b99d3',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gates(ewm_delta, mean14_delta, lower90, jepa_delta, trajectory_delta):
    return {'ewm_gain_ge_0_3_points': ewm_delta >= .3,
            'nonjepa14_mean_gain_gt_0': mean14_delta > 0,
            'nonjepa14_one_sided90_lower_gt_0': lower90 > 0,
            'jepa_delta_ge_minus_0_005': jepa_delta >= -.005,
            'trajectory_delta_ge_minus_0_005': trajectory_delta >= -.005}


def paired(delta, indices):
    boot = delta[indices].mean(axis=1)
    return {'mean_delta': float(delta.mean()), 'one_sided90_lower': float(np.quantile(boot, .1)),
            'two_sided95': np.quantile(boot, [.025, .975]).tolist(),
            'wins': int((delta > 0).sum()), 'ties': int((delta == 0).sum()),
            'losses': int((delta < 0).sum())}


def finish(output_root):
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f'refuse existing output: {output}')
    if not output.resolve().is_relative_to((H / 'analysis').resolve()):
        raise ValueError('output must be under the new experiment analysis root')
    snapshots = {}

    def track(path, expected=None, root=None):
        path = Path(path)
        if root is not None and not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError(f'input outside new experiment root: {path}')
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError(f'SHA mismatch: {path}')
        if str(path) in snapshots and snapshots[str(path)] != value:
            raise ValueError(f'input changed: {path}')
        snapshots[str(path)] = value
        return path

    def read(path, **kwargs):
        return json.loads(track(path, **kwargs).read_text())

    for relative, expected in FIXED_SHA.items():
        track(H / relative, expected, H)
    manifest = [json.loads(line) for line in (H / 'manifest.jsonl').read_text().splitlines() if line]
    ids = {f"{r['task']}_episode{r['episode_index']:06d}": r['task'] for r in manifest}
    samples = {r['episode_id'] for r in manifest}
    if len(manifest) != 16 or len(ids) != 16 or len(samples) != 16 or sorted(Counter(ids.values()).values()) != [4] * 4:
        raise ValueError('manifest must be exact 4 tasks x 4 unique episodes')
    fit = read(H / 'ranker-fit/fit.complete.json')
    if fit.get('status') != 'cpu_fit_complete':
        raise ValueError('ranker fit incomplete')
    track(H / 'ranker-fit/model.json', fit['output_sha256']['model.json'], H)
    route = read(H / 'route/route.complete.json')
    if (route.get('completed') is not True or route.get('rows') != 16
            or route.get('contract') != 'worldarena-ranker-fresh16-native-route-mvp/1'
            or route.get('hidden_gt_used_for_selection') is not False
            or route.get('manifest_sha256') != snapshots[str(H / 'manifest.jsonl')]):
        raise ValueError('route receipt incomplete or manifest mismatch')
    track(route['ranker_model'], route['ranker_model_sha256'], H / 'ranker-fit')
    if route['ranker_model_sha256'] != FIXED_SHA['ranker-fit/model.json']:
        raise ValueError('route ranker model SHA mismatch')
    track(route['decisions'], route['decisions_sha256'], H / 'route')
    predictions = {}
    for step, prefix in [(1232, 'p0'), (1233, 'ranker')]:
        path = track(route[prefix + '_predictions'], route[prefix + '_predictions_sha256'], H / 'route')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != 16 or {r['episode_id'] for r in rows} != samples:
            raise ValueError('route prediction identities differ from exact manifest')
        predictions[step] = snapshots[str(path)]
    gt_receipt = read(H / 'full15-gt-motion/gt-motion.complete.json')
    if (gt_receipt.get('completed') is not True or gt_receipt.get('episode_count') != 16
            or gt_receipt.get('metric_episode_counts') != {m: 16 for m in MOTION}):
        raise ValueError('shared GT motion is incomplete')
    gt = track(gt_receipt['results_json'], gt_receipt['results_json_sha256'], H / 'full15-gt-motion')
    caps = scorer._load_gt_caps(gt)
    if any(set(values) != set(ids) for values in caps.values()):
        raise ValueError('GT identities differ from exact manifest')
    ordered = sorted(ids)
    raw, csvs, jepa = {}, {}, {}

    def matrix(path, model):
        _, rows = scorer._read_raw_rows(path, expected_count=16)
        indexed = {r['Video_ID']: r for r in rows}
        if set(indexed) != set(ids) or any(r['Model_Name'] != model for r in rows):
            raise ValueError('CSV model or task+episode identity mismatch')
        return np.array([[float(indexed[k][m]) for m in METRICS] for k in ordered])

    ji = METRICS.index('JEPA Similarity')
    for step, model in MODELS.items():
        root = A / f'official_track1_eval/checkpoints/step-{step}/dev-clean-50'
        package = read(root / 'receipts/package.complete.json', root=root)
        if (package.get('completed') is not True or package.get('rows') != 16
                or package.get('model_name') != model
                or package.get('manifest_sha256') != snapshots[str(H / 'manifest.jsonl')]
                or package.get('predictions_sha256') != predictions[step]):
            raise ValueError('native package does not bind frozen manifest and route')
        for phase in ('base', 'vlm', 'jepa', 'aggregate'):
            receipt = read(root / f'receipts/{phase}.complete.json', root=root)
            if (receipt.get('completed') is not True or receipt.get('phase') != phase
                    or receipt.get('step') != step or receipt.get('model_name') != model
                    or receipt.get('source_commit') != '7b3feee108427bee3380064bb5154970ed7468b5'
                    or receipt.get('package_validation', {}).get('video_count') != 16):
                raise ValueError(f'incomplete or wrong phase receipt: {step}/{phase}')
        track(root / 'output/generated_results.json', root=root)
        track(root / f'output_VLM/{model}/{model}_summary_val_all_intern.json', root=root)
        jepa[step] = float(read(root / 'output_JEDi/results.json', root=root)['score'])
        csvs[step] = track(root / 'csv_results/aggregated_results.csv', root=root)
        raw[step] = matrix(csvs[step], model)
        if not np.isfinite(jepa[step]) or not np.all(raw[step][:, ji] == jepa[step]):
            raise ValueError('canonical CSV must contain the real collection JEPA value')

    # All dependency, SHA, membership and numeric checks precede output creation.
    output.mkdir(parents=True, exist_ok=False)
    corrected, cap_receipts = {}, {}
    for step, model in MODELS.items():
        target = output / f'step-{step}.gtcap.csv'
        cap_receipts[step] = scorer.apply_gt_reference_motion_caps(
            raw_csv=csvs[step], gt_results_json=gt, corrected_csv=target,
            receipt_json=output / f'step-{step}.gtcap.complete.json', expected_count=16)
        corrected[step] = matrix(target, model)
    nonjepa = [i for i in range(15) if i != ji]
    rng = np.random.default_rng(20260903)
    indices = np.concatenate([rng.choice([i for i, key in enumerate(ordered) if ids[key] == task],
                                         size=(10000, 4), replace=True)
                              for task in sorted(set(ids.values()))], axis=1)
    delta14 = corrected[1233][:, nonjepa].mean(axis=1) - corrected[1232][:, nonjepa].mean(axis=1)
    stats = paired(delta14, indices)
    ewm = {s: float(100 / 15 * (corrected[s][:, nonjepa].sum(axis=1).mean() + jepa[s])) for s in MODELS}
    metrics = {}
    for i, name in enumerate(METRICS):
        row = {f'{label}_{kind}_mean': float(values[s][:, i].mean())
               for label, s in [('p0', 1232), ('ranker', 1233)] for kind, values in [('raw', raw), ('corrected', corrected)]}
        row['delta'] = float((corrected[1233][:, i] - corrected[1232][:, i]).mean())
        if i != ji:
            row['paired'] = paired(corrected[1233][:, i] - corrected[1232][:, i], indices)
        metrics[name] = row
    checks = gates(ewm[1233] - ewm[1232], stats['mean_delta'], stats['one_sided90_lower'],
                   jepa[1233] - jepa[1232], metrics['Trajectory Accuracy']['delta'])
    result = {'completed': True, 'finished_at': datetime.now(timezone.utc).isoformat(),
              'label': 'local-public-protocol-not-official', 'decision': 'GO' if all(checks.values()) else 'NO-GO',
              'gates': checks, 'ewm15': {'p0_points': ewm[1232], 'ranker_points': ewm[1233], 'delta_points': ewm[1233] - ewm[1232]},
              'nonjepa14': stats, 'jepa': {'p0': jepa[1232], 'ranker': jepa[1233], 'delta': jepa[1233] - jepa[1232]},
              'metrics15': metrics, 'matched_gt_counts': {m: len(caps[m]) for m in MOTION},
              'gtcap_receipts': cap_receipts, 'episodes': ordered, 'task_counts': dict(Counter(ids.values())),
              'bootstrap': {'seed': 20260903, 'resamples': 10000, 'unit': 'paired episodes within 4 fixed tasks; 4 per task',
                            'jepa_excluded': True, 'one_sided90_quantile': .1, 'two_sided95_quantiles': [.025, .975]},
              'limitations': 'n=16, fixed tasks are not independent task-level replicates; episode-disjoint, not task-disjoint; parent pretraining exposure unknown; do not retune on these results; no official score/rank claim',
              'input_sha256': snapshots, 'script_sha256': sha(__file__), 'gtcap_scorer_sha256': sha(scorer.__file__)}
    for path, digest in snapshots.items():
        if sha(path) != digest:
            raise ValueError(f'input SHA changed during comparison: {path}')
    text = [f"# Fresh16: {result['decision']}", '', 'local-public-protocol-not-official', '',
            f"EWM15: {ewm[1232]:.6f} → {ewm[1233]:.6f}；Δ {ewm[1233]-ewm[1232]:+.6f} points。",
            f"非JEPA14 Δ {stats['mean_delta']:+.6f}；单侧90%下界 {stats['one_sided90_lower']:+.6f}；双侧95% CI {stats['two_sided95']}。",
            f"JEPA Δ {jepa[1233]-jepa[1232]:+.6f}；Trajectory Δ {metrics['Trajectory Accuracy']['delta']:+.6f}。", '',
            '| 预注册门槛 | 通过 |', '|---|---|', *[f'| {key} | {value} |' for key, value in checks.items()], '',
            'JEPA只用真实集合值，不构造逐例JEPA置信区间。其他14项按4 tasks × 4 episodes分层配对bootstrap，10000次，seed=20260903。', '',
            result['limitations'], '', '逐项均值、差值、GTcap与SHA证据见 comparison.complete.json；不自动改正式提交选择。']
    (output / 'comparison.md').write_text('\n'.join(text) + '\n')
    (output / 'comparison.complete.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    result = finish(parser.parse_args().output_root)
    print(json.dumps({k: result[k] for k in ('decision', 'ewm15', 'nonjepa14', 'gates')}, sort_keys=True))
