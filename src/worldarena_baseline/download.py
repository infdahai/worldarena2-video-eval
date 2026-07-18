from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DownloadError(RuntimeError):
    """A resumable download failed without invalidating existing bytes."""


def resume_download(
    url: str,
    destination: Path,
    expected_size: int,
    *,
    chunk_size: int = 8 * 1024 * 1024,
    timeout: float = 60.0,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = destination.stat().st_size if destination.exists() else 0
    if offset == expected_size:
        return
    if offset > expected_size:
        raise DownloadError(
            f"partial file is larger than expected: {offset} > {expected_size}"
        )

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code}") from None
    except URLError as exc:
        raise DownloadError(f"connection failed: {exc.reason}") from None

    with response:
        status = response.status
        if offset and status != 206:
            raise DownloadError(f"expected HTTP 206 for resume, got {status}")
        if not offset and status != 200:
            raise DownloadError(f"expected HTTP 200 for new download, got {status}")
        if offset:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
            if not match or int(match.group(1)) != offset:
                raise DownloadError(
                    f"invalid Content-Range for offset {offset}: {content_range!r}"
                )

        mode = "ab" if offset else "wb"
        try:
            with destination.open(mode) as output:
                while chunk := response.read(chunk_size):
                    output.write(chunk)
        except OSError as exc:
            raise DownloadError(f"transfer failed: {type(exc).__name__}") from None

    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise DownloadError(
            f"incomplete download preserved: {actual_size} of {expected_size} bytes"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely resume a signed weight download")
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-size", type=int, required=True)
    args = parser.parse_args()

    before = args.destination.stat().st_size if args.destination.exists() else 0
    print(f"resuming at {before} of {args.expected_size} bytes", flush=True)
    resume_download(args.url, args.destination, args.expected_size)
    print(f"complete: {args.destination} ({args.expected_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
