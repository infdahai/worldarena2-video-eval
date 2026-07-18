from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .submission import validate_submission_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the validated Track 1 archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo-name", default="worldarena2-track1-oscar-baseline-v1")
    parser.add_argument("--verify-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_submission_archive(args.archive, expected_episodes=1000)
    api = HfApi()
    identity = api.whoami()
    owner = identity.get("name")
    if not owner:
        raise RuntimeError("Hugging Face user authentication is required")
    repo_id = f"{owner}/{args.repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    archives = [
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if "/" not in path and path.endswith((".tar.gz", ".tgz"))
    ]
    unexpected = [path for path in archives if path != "submission.tar.gz"]
    if unexpected:
        raise RuntimeError(f"dataset repo already contains other archives: {unexpected}")
    api.upload_file(
        path_or_fileobj=str(args.archive),
        path_in_repo="submission.tar.gz",
        repo_id=repo_id,
        repo_type="dataset",
    )
    args.verify_dir.mkdir(parents=True, exist_ok=True)
    public_copy = hf_hub_download(
        repo_id=repo_id,
        filename="submission.tar.gz",
        repo_type="dataset",
        token=False,
        cache_dir=args.verify_dir,
        force_download=True,
    )
    public_report = validate_submission_archive(public_copy, expected_episodes=1000)
    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "revision": "main",
                "uploaded_videos": report.video_count,
                "public_download_verified": public_report.video_count == 1000,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
