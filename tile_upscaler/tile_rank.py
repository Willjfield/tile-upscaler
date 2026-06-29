"""Rank input tiles by how much methods A, B, and C diverge in their outputs.

Used to pick showcase tiles for comparisons and to drive ``run_experiment.py
--best N``.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import tileio

DEFAULT_METHOD_DIRS = {
    "A": "A_realesrgan",
    "B": "B_controlnet_text",
    "C": "C_controlnet_osm",
}

RANKINGS_FILENAME = "tile_rankings.json"


@dataclass
class TileRankRow:
    rank: int
    tile: str
    score: float
    mean_pairwise_mae: float
    ab_mae: float
    ac_mae: float
    bc_mae: float
    consistency_spread: Optional[float] = None
    edge_coverage: Optional[float] = None


def _compare_size() -> int:
    return 512


def _to_array(image, size: int) -> np.ndarray:
    from PIL import Image

    if image.size != (size, size):
        image = image.resize((size, size), Image.BICUBIC)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _edge_coverage(edges_path: Optional[str]) -> Optional[float]:
    if not edges_path or not os.path.isfile(edges_path):
        return None
    arr = np.asarray(tileio.load_image(edges_path).convert("L"), dtype=np.float32)
    return float(np.mean(arr > 127.0))


def _load_consistency_mae(metrics_dir: str) -> Dict[str, Dict[str, float]]:
    """Return {method_dir: {tile_key: color_mae}} from consistency CSVs."""
    out: Dict[str, Dict[str, float]] = {}
    if not os.path.isdir(metrics_dir):
        return out
    for fname in os.listdir(metrics_dir):
        if not fname.endswith("_consistency.csv"):
            continue
        method = fname[: -len("_consistency.csv")]
        rows: Dict[str, float] = {}
        with open(os.path.join(metrics_dir, fname), newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("tile") and row.get("color_mae"):
                    rows[row["tile"]] = float(row["color_mae"])
        if rows:
            out[method] = rows
    return out


def _score_tile(
    tile_key: str,
    arrays: Dict[str, np.ndarray],
    consistency: Dict[str, Dict[str, float]],
    method_dirs: Dict[str, str],
    edges_path: Optional[str],
) -> Tuple[float, float, float, float, float, Optional[float], Optional[float]]:
    a, b, c = arrays["A"], arrays["B"], arrays["C"]
    ab = _mae(a, b)
    ac = _mae(a, c)
    bc = _mae(b, c)
    mean_pairwise = (ab + ac + bc) / 3.0

    spread: Optional[float] = None
    a_m = consistency.get(method_dirs["A"], {}).get(tile_key)
    b_m = consistency.get(method_dirs["B"], {}).get(tile_key)
    c_m = consistency.get(method_dirs["C"], {}).get(tile_key)
    if a_m is not None and b_m is not None and c_m is not None:
        spread = a_m - (b_m + c_m) / 2.0

    edge_cov = _edge_coverage(edges_path)

    # Higher = methods diverge more (better showcase tiles).
    score = mean_pairwise + 0.5 * bc + (2.0 * spread if spread is not None else 0.0)
    if edge_cov is not None:
        score += 8.0 * edge_cov

    return score, mean_pairwise, ab, ac, bc, spread, edge_cov


def rank_tiles(
    out_dir: str,
    *,
    method_dirs: Optional[Dict[str, str]] = None,
    compare_size: int = 512,
) -> List[TileRankRow]:
    """Rank tiles present in all three method output trees under ``out/up/``."""
    method_dirs = method_dirs or DEFAULT_METHOD_DIRS
    up_root = os.path.join(out_dir, "up")
    metrics_dir = os.path.join(out_dir, "metrics")
    edges_root = os.path.join(out_dir, "osm", "edges")
    consistency = _load_consistency_mae(metrics_dir)

    paths: Dict[str, Dict[str, str]] = {}
    for label, subdir in method_dirs.items():
        root = os.path.join(up_root, subdir)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Missing method output directory: {root}")
        paths[label] = {tf.tile.key: tf.path for tf in tileio.find_tiles(root)}

    common = sorted(set(paths["A"]) & set(paths["B"]) & set(paths["C"]))
    if not common:
        raise SystemExit(
            "No tiles found in all three method trees "
            f"({method_dirs['A']}, {method_dirs['B']}, {method_dirs['C']}). "
            "Run A, B, and C on overlapping tiles first."
        )

    scored: List[TileRankRow] = []
    for tile_key in common:
        arrays = {}
        for label in ("A", "B", "C"):
            img = tileio.load_image(paths[label][tile_key])
            arrays[label] = _to_array(img, compare_size)

        z, x, y = (int(v) for v in tile_key.split("/"))
        from .tiles import Tile

        edges_path = tileio.find_companion(edges_root, Tile(z, x, y))

        score, mean_p, ab, ac, bc, spread, edge_cov = _score_tile(
            tile_key, arrays, consistency, method_dirs, edges_path,
        )
        scored.append(
            TileRankRow(
                rank=0,
                tile=tile_key,
                score=score,
                mean_pairwise_mae=mean_p,
                ab_mae=ab,
                ac_mae=ac,
                bc_mae=bc,
                consistency_spread=spread,
                edge_coverage=edge_cov,
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    for i, row in enumerate(scored, 1):
        scored[i - 1] = TileRankRow(rank=i, **{k: v for k, v in asdict(row).items() if k != "rank"})
    return scored


def write_rankings(
    out_dir: str,
    rows: Sequence[TileRankRow],
    *,
    path: Optional[str] = None,
    method_dirs: Optional[Dict[str, str]] = None,
) -> str:
    method_dirs = method_dirs or DEFAULT_METHOD_DIRS
    dest = path or os.path.join(out_dir, RANKINGS_FILENAME)
    parent = os.path.dirname(os.path.abspath(dest))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": os.path.abspath(out_dir),
        "compare_size_px": _compare_size(),
        "methods": {
            label: os.path.join(out_dir, "up", subdir)
            for label, subdir in method_dirs.items()
        },
        "rankings": [asdict(r) for r in rows],
    }
    with open(dest, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return dest


def load_rankings(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def best_tile_keys(path: str, n: int) -> List[str]:
    """Return the top ``n`` tile keys from a rankings file, in rank order."""
    if n <= 0:
        raise ValueError("--best must be a positive integer")
    data = load_rankings(path)
    rankings = data.get("rankings") or []
    if not rankings:
        raise SystemExit(f"No rankings in {path}")
    keys = [row["tile"] for row in sorted(rankings, key=lambda r: r.get("rank", 999999))]
    if len(keys) < n:
        print(
            f"Warning: rankings file has only {len(keys)} tile(s); using all of them (--best {n})"
        )
    return keys[:n]


def default_rankings_path(out_dir: str) -> str:
    return os.path.join(out_dir, RANKINGS_FILENAME)
