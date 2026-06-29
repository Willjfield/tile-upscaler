#!/usr/bin/env python3
"""Rank tiles by how much methods A, B, and C differ in existing ``out/`` results.

Writes ``<out>/tile_rankings.json``. Use with ``run_experiment.py --best N`` to
process the top-ranked tiles instead of the first N.

Usage:
  python scripts/rank_tiles.py
  python scripts/rank_tiles.py --out out --output out/tile_rankings.json
  python scripts/rank_tiles.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tile_upscaler.tile_rank import (  # noqa: E402
    rank_tiles,
    write_rankings,
)


def _load_out_from_config(config_path: str) -> str:
    import yaml

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg["paths"]["out"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank tiles by A/B/C output divergence (writes tile_rankings.json)",
    )
    parser.add_argument("--config", default="config.yaml", help="Used to resolve paths.out")
    parser.add_argument("--out", default=None, help="Experiment output dir (default: paths.out from config)")
    parser.add_argument(
        "--output",
        default=None,
        help="Rankings JSON path (default: <out>/tile_rankings.json)",
    )
    parser.add_argument("--top", type=int, default=10, help="Print this many rows to stdout")
    args = parser.parse_args(argv)

    out_dir = args.out
    if out_dir is None:
        config_path = args.config
        if not os.path.isfile(config_path):
            raise SystemExit(f"Config not found: {config_path} (pass --out explicitly)")
        out_dir = _load_out_from_config(config_path)

    rows = rank_tiles(out_dir)
    dest = write_rankings(out_dir, rows, path=args.output)

    print(f"Ranked {len(rows)} tile(s) with outputs in A, B, and C")
    print(f"Wrote {dest}\n")
    print(f"{'rank':>4}  {'tile':<20}  {'score':>8}  {'mae(A,B,C)':>10}  {'bc_mae':>8}  spread")
    for row in rows[: args.top]:
        spread = f"{row.consistency_spread:.2f}" if row.consistency_spread is not None else "-"
        print(
            f"{row.rank:4d}  {row.tile:<20}  {row.score:8.2f}  "
            f"{row.mean_pairwise_mae:10.2f}  {row.bc_mae:8.2f}  {spread}"
        )
    if len(rows) > args.top:
        print(f"... and {len(rows) - args.top} more (see {dest})")
    print(f"\nRe-run experiment on top tiles: python run_experiment.py --best {min(5, len(rows))} --skip-osm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
