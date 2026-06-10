#!/usr/bin/env python3
"""Tile downloader server: serves the Leaflet UI and fetches XYZ tiles.

Run from the repo root:

    python tile_fetcher/server.py [--port 8080] [--out data/raster]

Then open http://localhost:8080 - enter a tile provider URL template, drag a
bounding box, and the server downloads every touched tile into the output tree
as ``z/x/y.png`` (the layout ``run_experiment.py`` expects).

Endpoints:
    GET  /          - the single-page UI (index.html)
    POST /download  - {url_template, zoom, bbox, subdomains}
    GET  /progress  - {done, total, errors, finished, running}
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

import requests

# Allow running both from the repo root and from inside tile_fetcher/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tile_upscaler.tiles import Tile, tiles_for_bbox  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
USER_AGENT = "tile-upscaler-fetcher/1.0"
REQUEST_DELAY_S = 0.1          # polite pause between requests
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0          # doubled per retry on 429/5xx


class DownloadJob:
    """State for one bbox download, run on a background thread."""

    def __init__(self, tiles: List[Tile], url_template: str, out_root: str,
                 subdomains: Optional[List[str]] = None):
        self.tiles = tiles
        self.url_template = url_template
        self.out_root = out_root
        self.subdomains = itertools.cycle(subdomains or ["a"])
        self.total = len(tiles)
        self.done = 0
        self.skipped = 0
        self.errors: List[str] = []
        self.finished = False
        self.lock = threading.Lock()

    # -- URL building --------------------------------------------------------
    def url_for(self, tile: Tile) -> str:
        return (self.url_template
                .replace("{z}", str(tile.z))
                .replace("{x}", str(tile.x))
                .replace("{y}", str(tile.y))
                .replace("{s}", next(self.subdomains)))

    # -- fetching ------------------------------------------------------------
    def fetch_one(self, session: requests.Session, tile: Tile) -> Optional[bytes]:
        url = self.url_for(tile)
        backoff = RETRY_BACKOFF_S
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(url, timeout=30)
            except requests.RequestException as exc:
                print(f"  GET {url} -> {exc}", flush=True)
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"{tile.key}: {exc} [{url}]") from exc
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"  GET {url} -> {resp.status_code}", flush=True)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(f"{tile.key}: HTTP {resp.status_code} [{url}]")
        raise RuntimeError(f"{tile.key}: gave up after {MAX_RETRIES} retries [{url}]")

    def save(self, tile: Tile, data: bytes) -> None:
        out = os.path.join(self.out_root, str(tile.z), str(tile.x), f"{tile.y}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            with open(out, "wb") as fh:
                fh.write(data)
            return
        # Non-PNG (usually JPEG): convert so the z/x/y.png contract holds.
        from PIL import Image

        Image.open(io.BytesIO(data)).convert("RGB").save(out)

    def run(self) -> None:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        try:
            for tile in self.tiles:
                out = os.path.join(self.out_root, str(tile.z), str(tile.x), f"{tile.y}.png")
                if os.path.exists(out):
                    with self.lock:
                        self.done += 1
                        self.skipped += 1
                    continue
                try:
                    data = self.fetch_one(session, tile)
                    self.save(tile, data)
                except Exception as exc:
                    with self.lock:
                        self.errors.append(str(exc))
                with self.lock:
                    self.done += 1
                time.sleep(REQUEST_DELAY_S)
        finally:
            with self.lock:
                self.finished = True

    def progress(self) -> dict:
        with self.lock:
            return {
                "done": self.done,
                "total": self.total,
                "skipped": self.skipped,
                "errors": self.errors[-20:],
                "error_count": len(self.errors),
                "finished": self.finished,
                "running": not self.finished,
            }


class _State:
    job: Optional[DownloadJob] = None
    out_root: str = "data/raster"


def _start_download(body: dict) -> dict:
    if _State.job and not _State.job.finished:
        return {"error": "A download is already running"}

    template = (body.get("url_template") or "").strip()
    if "{z}" not in template or "{x}" not in template or "{y}" not in template:
        return {"error": "URL template must contain {z}, {x} and {y}"}
    try:
        zoom = int(body["zoom"])
        west, south, east, north = (float(v) for v in body["bbox"])
    except (KeyError, TypeError, ValueError):
        return {"error": "Need zoom (int) and bbox [west, south, east, north]"}
    if not 0 <= zoom <= 22:
        return {"error": "Zoom must be between 0 and 22"}

    subdomains = [s.strip() for s in (body.get("subdomains") or "").split(",") if s.strip()]
    tiles = tiles_for_bbox((west, south, east, north), zoom)
    if not tiles:
        return {"error": "Bounding box covers no tiles"}

    job = DownloadJob(tiles, template, _State.out_root, subdomains=subdomains or None)
    _State.job = job
    threading.Thread(target=job.run, daemon=True).start()
    return {"started": True, "total": job.total, "out": os.path.abspath(_State.out_root)}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/progress":
            if _State.job is None:
                self._send_json({"running": False, "finished": False, "done": 0, "total": 0})
            else:
                self._send_json(_State.job.progress())
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/download":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return
        result = _start_download(body)
        self._send_json(result, status=400 if "error" in result else 200)

    def log_message(self, fmt: str, *args) -> None:
        if "/progress" not in str(args[0] if args else ""):  # quiet the poll spam
            super().log_message(fmt, *args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Leaflet XYZ tile downloader")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out", default="data/raster",
                        help="Output tile tree root (default: data/raster)")
    args = parser.parse_args(argv)

    _State.out_root = args.out
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Tile fetcher UI:  http://localhost:{args.port}")
    print(f"Tiles saved to:   {os.path.abspath(args.out)} (z/x/y.png)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
