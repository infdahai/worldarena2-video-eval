from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from worldarena_baseline.download import DownloadError, resume_download


@contextmanager
def serve(handler: type[BaseHTTPRequestHandler]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/weight"
    finally:
        server.shutdown()
        thread.join()


class QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def test_expired_url_preserves_partial_file(tmp_path: Path) -> None:
    class ExpiredHandler(QuietHandler):
        def do_GET(self) -> None:
            self.send_response(403)
            self.end_headers()

    target = tmp_path / "weight.bin"
    target.write_bytes(b"partial")

    with serve(ExpiredHandler) as url:
        with pytest.raises(DownloadError, match="HTTP 403"):
            resume_download(url, target, expected_size=12)

    assert target.read_bytes() == b"partial"


def test_range_response_appends_to_partial_file(tmp_path: Path) -> None:
    payload = b"complete-weight"

    class RangeHandler(QuietHandler):
        def do_GET(self) -> None:
            assert self.headers["Range"] == "bytes=5-"
            self.send_response(206)
            self.send_header("Content-Range", f"bytes 5-{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(payload) - 5))
            self.end_headers()
            self.wfile.write(payload[5:])

    target = tmp_path / "weight.bin"
    target.write_bytes(payload[:5])

    with serve(RangeHandler) as url:
        resume_download(url, target, expected_size=len(payload))

    assert target.read_bytes() == payload


def test_server_ignoring_range_preserves_partial_file(tmp_path: Path) -> None:
    class NoRangeHandler(QuietHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "8")
            self.end_headers()
            self.wfile.write(b"new-data")

    target = tmp_path / "weight.bin"
    target.write_bytes(b"partial")

    with serve(NoRangeHandler) as url:
        with pytest.raises(DownloadError, match="expected HTTP 206"):
            resume_download(url, target, expected_size=12)

    assert target.read_bytes() == b"partial"
