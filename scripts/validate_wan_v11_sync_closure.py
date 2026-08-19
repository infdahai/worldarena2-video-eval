#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldarena_baseline.wan_v11_sync_closure import (
    build_v11_source_receipt,
    validate_v11_source_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-receipt", type=Path)
    parser.add_argument("--require-git-clean", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_receipt:
        receipt = validate_v11_source_receipt(
            args.source_root,
            json.loads(args.expected_receipt.read_text()),
        )
    else:
        receipt = build_v11_source_receipt(
            args.source_root,
            require_git_clean=args.require_git_clean,
        )
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
