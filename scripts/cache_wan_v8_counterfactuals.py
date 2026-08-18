#!/usr/bin/env python3
"""CPU-only FK rendering of complete Wan v8 action counterfactual sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from worldarena_baseline.action_condition import EpisodeTimeline
from worldarena_baseline.action_raster import rasterize_v3_action
from worldarena_baseline.robotwin_action_cache import _v3_support_and_loss_weight
from worldarena_baseline.skeleton import AlohaSkeletonRenderer, CameraCalibration
from worldarena_baseline.wan_se3_condition import build_se3_condition
from worldarena_baseline.wan_v8_counterfactual import build_joint14_counterfactual


ROOT = Path("/data/di/worldarena2_track1_20260815")
FAMILIES = (("reverse", 0, "reverse"), ("shift_plus", 1, "shift"), ("shift_minus", -1, "shift"), ("swap", 0, "swap"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_root(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"v8 output/input escapes artifact root: {path}")
    return resolved


def _load_sample(dataset_root: Path, row: dict[str, object]) -> tuple[np.ndarray, CameraCalibration, str]:
    sample = row.get("sample"); relative = row.get("hdf5")
    if not isinstance(sample, str) or not isinstance(relative, str):
        raise ValueError("manifest row lacks sample/hdf5")
    hdf5_path = (dataset_root / relative).resolve(strict=True)
    if dataset_root.resolve() not in hdf5_path.parents:
        raise ValueError("HDF5 escapes dataset root")
    with h5py.File(hdf5_path, "r") as handle:
        actions = np.asarray(handle["joint_action/vector"], dtype=np.float64)
        intrinsic = np.asarray(handle["observation/head_camera/intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(handle["observation/head_camera/extrinsic_cv"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 2:
        raise ValueError(f"invalid joint14 trajectory: {sample}")
    if intrinsic.shape != (len(actions), 3, 3) or extrinsic.shape != (len(actions), 3, 4):
        raise ValueError(f"invalid camera trajectory: {sample}")
    if not np.allclose(intrinsic, intrinsic[:1]) or not np.allclose(extrinsic, extrinsic[:1]):
        raise ValueError(f"time-varying camera unsupported: {sample}")
    return actions, CameraCalibration(intrinsic[0], extrinsic[0], source_hash=_sha(hdf5_path)), _sha(hdf5_path)


def _render(renderer: AlohaSkeletonRenderer, actions: np.ndarray, camera: CameraCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory = renderer.project_actions(actions, num_frames=81, camera_calibration=camera)
    raster = rasterize_v3_action(trajectory)
    support, _ = _v3_support_and_loss_weight(raster)
    left, right = renderer.fk_endposes(actions)
    se3 = build_se3_condition(left, right, EpisodeTimeline.build(source_length=len(actions), num_frames=81))
    return raster.astype(np.float32), support.astype(np.float32), se3.arm_transform, se3.arm_present


def _valid(path: Path, hdf_sha: str, urdf_sha: str) -> bool:
    if not path.is_file() or path.is_symlink(): return False
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            expected = {f"{name}_{field}" for name, _, _ in FAMILIES for field in ("raster", "support", "se3", "arm_present")} | {"source_hdf5_sha256", "urdf_sha256", "schema"}
            if keys != expected or str(z["schema"].item()) != "wan-v8-complete-action/1": return False
            if str(z["source_hdf5_sha256"].item()) != hdf_sha or str(z["urdf_sha256"].item()) != urdf_sha: return False
            return all(z[f"{name}_raster"].shape == (81,10,60,80) and z[f"{name}_support"].shape == (2,21,15,20) and z[f"{name}_se3"].shape == (2,21,4,4) and z[f"{name}_arm_present"].shape == (2,21) for name,_,_ in FAMILIES)
    except (OSError, ValueError, KeyError): return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.dataset_root,args.manifest,args.output_root): _require_root(path)
    if not args.urdf.is_file() or args.urdf.is_symlink(): raise ValueError("URDF must be a regular source file")
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if len(rows)!=1785: raise ValueError(f"expected clean-1785 manifest, got {len(rows)}")
    renderer=AlohaSkeletonRenderer(args.urdf,width=80,height=60); urdf_sha=_sha(args.urdf); out=args.output_root/"wan_v8_counterfactuals"; out.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(rows,1):
        sample=str(row["sample"]); actions,camera,hdf_sha=_load_sample(args.dataset_root,row); target=out/f"{sample}.npz"
        if _valid(target,hdf_sha,urdf_sha): continue
        payload={"schema":np.asarray("wan-v8-complete-action/1"),"source_hdf5_sha256":np.asarray(hdf_sha),"urdf_sha256":np.asarray(urdf_sha)}
        for label,direction,family in FAMILIES:
            raster,support,se3,present=_render(renderer,build_joint14_counterfactual(actions,family,shift_direction=direction),camera)
            payload.update({f"{label}_raster":raster,f"{label}_support":support,f"{label}_se3":se3,f"{label}_arm_present":present})
        partial=target.with_name(f".{target.name}.{os.getpid()}.partial")
        with partial.open("wb") as h: np.savez_compressed(h,**payload); h.flush(); os.fsync(h.fileno())
        if not _valid(partial,hdf_sha,urdf_sha): raise RuntimeError(f"invalid rendered sidecar: {sample}")
        os.replace(partial,target)
        if index % 25 == 0: print(f"cached={index}/1785",flush=True)
    print(f"complete=1785 output={out}",flush=True)


if __name__ == "__main__": main()
