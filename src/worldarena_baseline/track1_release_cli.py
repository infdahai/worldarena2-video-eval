from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .track1_release import (
    build_deterministic_submission_archive,
    build_test1000_plan,
    finalize_release_verification,
    freeze_p0_release,
    materialize_test1000_shards,
    publish_hf_smoke,
    select_frozen_p0_candidates,
    stage_selected_videos,
    validate_submission_archive_strict,
)


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen WorldArena Track 1 P0 release pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--receipt", type=Path, required=True)
    plan.add_argument("--expected-count", type=int, default=1000)
    plan.add_argument("--shard-count", type=int, default=4)

    materialize = commands.add_parser("materialize-shards")
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--expected-count", type=int, default=1000)
    materialize.add_argument("--shard-count", type=int, default=4)

    stage = commands.add_parser("stage")
    stage.add_argument("--predictions", type=Path, required=True)
    stage.add_argument("--staging", type=Path, required=True)
    stage.add_argument("--metadata", type=Path, required=True)
    stage.add_argument("--expected-count", type=int, required=True)
    stage.add_argument("--expected-frames", type=int, default=121)
    stage.add_argument("--expected-width", type=int, default=640)
    stage.add_argument("--expected-height", type=int, default=480)

    select = commands.add_parser("select")
    select.add_argument("--model", type=Path, required=True)
    select.add_argument("--policy", type=Path, required=True)
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--predictions", type=Path, required=True)
    select.add_argument("--receipt", type=Path, required=True)
    select.add_argument("--expected-count", type=int, default=1000)

    package = commands.add_parser("package")
    package.add_argument("--staging", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--receipt", type=Path, required=True)
    package.add_argument("--expected-count", type=int, required=True)

    hf_smoke = commands.add_parser("hf-smoke")
    hf_smoke.add_argument("--archive", type=Path, required=True)
    hf_smoke.add_argument("--repo-id", required=True)
    hf_smoke.add_argument("--receipt", type=Path, required=True)

    anonymous = commands.add_parser("anonymous-verify")
    anonymous.add_argument("--repo-id", required=True)
    anonymous.add_argument("--revision", required=True)
    anonymous.add_argument("--filename", required=True)
    anonymous.add_argument("--expected-sha256", required=True)
    anonymous.add_argument("--download", type=Path, required=True)
    anonymous.add_argument("--receipt", type=Path, required=True)
    anonymous.add_argument("--expected-count", type=int, required=True)
    anonymous.add_argument("--proxy")

    finalize = commands.add_parser("finalize-verification")
    finalize.add_argument("--freeze-receipt", type=Path, required=True)
    finalize.add_argument("--plan-receipt", type=Path, required=True)
    finalize.add_argument("--frozen-selector-receipt", type=Path, required=True)
    finalize.add_argument("--selector-replay-receipt", type=Path, required=True)
    finalize.add_argument("--seed1-generation-receipt", type=Path, required=True)
    finalize.add_argument("--seed4-generation-receipt", type=Path, required=True)
    finalize.add_argument("--package-receipt", type=Path, required=True)
    finalize.add_argument("--hf-upload-receipt", type=Path, required=True)
    finalize.add_argument("--anonymous-receipt", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def _anonymous_verify(args: argparse.Namespace) -> dict[str, Any]:
    if "smoke" not in args.repo_id.lower() or "smoke" not in args.filename.lower():
        raise ValueError(
            "anonymous smoke verification requires smoke in repo and filename"
        )
    quoted_repo = "/".join(
        urllib.parse.quote(part, safe="") for part in args.repo_id.split("/")
    )
    quoted_revision = urllib.parse.quote(args.revision, safe="")
    quoted_filename = urllib.parse.quote(args.filename, safe="")
    url = (
        f"https://huggingface.co/datasets/{quoted_repo}/resolve/"
        f"{quoted_revision}/{quoted_filename}?download=true"
    )
    handlers: list[Any] = []
    if args.proxy:
        handlers.append(
            urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
        )
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(
        url, headers={"User-Agent": "worldarena-track1-public-smoke/1"}
    )
    args.download.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{args.download.name}.", dir=args.download.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with (
            opener.open(request, timeout=300) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        os.replace(temporary_path, args.download)
    finally:
        temporary_path.unlink(missing_ok=True)
    actual = _sha256(args.download)
    if actual != args.expected_sha256:
        raise ValueError(
            f"anonymous download SHA256 mismatch: expected {args.expected_sha256}, got {actual}"
        )
    archive_report = validate_submission_archive_strict(
        args.download, expected_count=args.expected_count
    )
    return {
        "completed": True,
        "terminal_state": "hf_public_smoke_anonymously_verified",
        "contract": "worldarena-track1-hf-anonymous-smoke/1",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "filename": args.filename,
        "url": url,
        "authorization_header_used": False,
        "download": str(args.download),
        "download_sha256": actual,
        "archive_validation": archive_report,
        "official_submission_not_triggered": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        payload = freeze_p0_release(args.config, args.output)
    elif args.command == "plan":
        payload = build_test1000_plan(
            args.manifest,
            args.output,
            expected_count=args.expected_count,
            shard_count=args.shard_count,
        )
        _write_receipt(args.receipt, payload)
    elif args.command == "materialize-shards":
        payload = materialize_test1000_shards(
            plan_path=args.plan,
            source_root=args.source_root,
            output_root=args.output_root,
            expected_count=args.expected_count,
            shard_count=args.shard_count,
        )
    elif args.command == "stage":
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        payload = stage_selected_videos(
            args.predictions,
            args.staging,
            metadata=metadata,
            expected_count=args.expected_count,
            expected_frames=args.expected_frames,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
        )
    elif args.command == "select":
        payload = select_frozen_p0_candidates(
            model_path=args.model,
            policy_path=args.policy,
            candidates_path=args.candidates,
            output_predictions=args.predictions,
            output_receipt=args.receipt,
            expected_count=args.expected_count,
        )
    elif args.command == "package":
        payload = build_deterministic_submission_archive(args.staging, args.archive)
        payload["validation"] = validate_submission_archive_strict(
            args.archive, expected_count=args.expected_count
        )
        _write_receipt(args.receipt, payload)
    elif args.command == "hf-smoke":
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required for hf-smoke") from error
        payload = publish_hf_smoke(
            args.archive,
            repo_id=args.repo_id,
            api=HfApi(token=os.environ.get("HF_TOKEN")),
            expected_sha256=_sha256(args.archive),
        )
        _write_receipt(args.receipt, payload)
    elif args.command == "anonymous-verify":
        payload = _anonymous_verify(args)
        _write_receipt(args.receipt, payload)
    elif args.command == "finalize-verification":
        payload = finalize_release_verification(
            freeze_receipt=args.freeze_receipt,
            plan_receipt=args.plan_receipt,
            frozen_selector_receipt=args.frozen_selector_receipt,
            selector_replay_receipt=args.selector_replay_receipt,
            seed1_generation_receipt=args.seed1_generation_receipt,
            seed4_generation_receipt=args.seed4_generation_receipt,
            package_receipt=args.package_receipt,
            hf_upload_receipt=args.hf_upload_receipt,
            anonymous_receipt=args.anonymous_receipt,
            output_receipt=args.output,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
