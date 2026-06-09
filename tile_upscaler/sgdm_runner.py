"""Stretch goal: run SGDM, the purpose-built vector-guided remote-sensing SR.

SGDM ("Semantic-Guided Large-Scale Factor Remote Sensing Image Super-Resolution
with Generative Diffusion Prior", ISPRS 2025, https://github.com/wwangcece/SGDM)
is the closest published match to this project's idea: a diffusion SR model that
takes a *vector map* as semantic guidance for remote-sensing imagery. It is the
strongest conceptual fit, but it is an external research repo with its own
weights and config, so we integrate it rather than reimplement it.

Important caveat (from the repo): the publicly released, ready-to-run weights are
the **no-map** variant (`sync_x32_no_map.ckpt`, set `use_map: False`). The fully
vector-conditioned variant may require additional setup/training of the style
normalising-flow model. So treat the with-map path as experimental.

This helper:
  - clones the repo into ``external/SGDM`` (``setup``),
  - converts our LR tile tree (and optionally OSM control tree) into the flat
    input folder SGDM expects (``prepare``),
  - invokes the repo's inference script via subprocess (``run``),
  - and reminds you how to fold the outputs back in for evaluation.

Because SGDM's exact CLI/config evolves, ``run`` shells out to a command you can
inspect/override; defaults follow the repo README at time of writing.
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional

from . import tileio

REPO_URL = "https://github.com/wwangcece/SGDM"
DEFAULT_DIR = os.path.join("external", "SGDM")


def setup(target_dir: str = DEFAULT_DIR) -> str:
    """Clone the SGDM repo if not already present. Returns its path."""
    if os.path.isdir(os.path.join(target_dir, ".git")):
        print(f"SGDM already present at {target_dir}")
        return target_dir
    os.makedirs(os.path.dirname(target_dir) or ".", exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, target_dir], check=True)
    print(
        f"Cloned SGDM -> {target_dir}\n"
        "Next: create its conda env and download checkpoints into "
        f"{target_dir}/checkpoints/ as per its README "
        "(start with the no-map weights: sync_x32_no_map.ckpt)."
    )
    return target_dir


def prepare(
    lr_root: str,
    dst_dir: str,
    osm_root: Optional[str] = None,
) -> int:
    """Flatten our z/x/y LR tiles into ``dst_dir/lr`` (and OSM into ``dst_dir/map``).

    File names encode the tile key as ``z_x_y.png`` so outputs can be mapped back
    to tiles by ``collect``.
    """
    from PIL import Image  # noqa: F401

    lr_out = os.path.join(dst_dir, "lr")
    os.makedirs(lr_out, exist_ok=True)
    n = 0
    for tf in tileio.find_tiles(lr_root):
        img = tileio.load_image(tf.path)
        img.save(os.path.join(lr_out, f"{tf.tile.z}_{tf.tile.x}_{tf.tile.y}.png"))
        n += 1

    if osm_root:
        map_out = os.path.join(dst_dir, "map")
        os.makedirs(map_out, exist_ok=True)
        for tf in tileio.find_tiles(osm_root):
            img = tileio.load_image(tf.path)
            img.save(os.path.join(map_out, f"{tf.tile.z}_{tf.tile.x}_{tf.tile.y}.png"))
    print(f"Prepared {n} LR tiles for SGDM under {dst_dir}")
    return n


def collect(flat_dir: str, out_root: str) -> int:
    """Map SGDM's flat ``z_x_y.png`` outputs back into a z/x/y tile tree."""
    from PIL import Image

    from .tiles import Tile

    n = 0
    for fname in os.listdir(flat_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        try:
            z, x, y = (int(v) for v in stem.split("_")[:3])
        except ValueError:
            continue
        img = Image.open(os.path.join(flat_dir, fname)).convert("RGB")
        tileio.write_tile(out_root, Tile(z, x, y), img)
        n += 1
    print(f"Collected {n} SGDM outputs -> {out_root}")
    return n


def run(
    repo_dir: str,
    input_dir: str,
    output_dir: str,
    use_map: bool = False,
    extra_args: Optional[List[str]] = None,
    python: str = "python",
) -> None:
    """Invoke SGDM inference via subprocess.

    The exact entry point/flags depend on the repo version; adjust ``extra_args``
    or this command to match its README. By default we point at an ``inference``
    script and pass input/output dirs and the map toggle.
    """
    cfg = "configs/model/refsr_real.yaml" if use_map else "configs/model/refsr_simu.yaml"
    cmd = [
        python, "inference.py",
        "--config", cfg,
        "--input", os.path.abspath(input_dir),
        "--output", os.path.abspath(output_dir),
        "--use_map", "true" if use_map else "false",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("Running SGDM:", " ".join(cmd))
    print(f"  (cwd={repo_dir})")
    subprocess.run(cmd, cwd=repo_dir, check=True)


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="SGDM (stretch) integration helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="Clone the SGDM repo")

    p = sub.add_parser("prepare", help="Flatten LR (+OSM) tiles for SGDM")
    p.add_argument("--lr", required=True)
    p.add_argument("--osm")
    p.add_argument("--dst", required=True)

    p = sub.add_parser("run", help="Run SGDM inference (subprocess)")
    p.add_argument("--repo", default=DEFAULT_DIR)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--use-map", action="store_true")
    p.add_argument("--python", default="python")

    p = sub.add_parser("collect", help="Map flat outputs back to a z/x/y tree")
    p.add_argument("--flat", required=True)
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "setup":
        setup()
    elif args.cmd == "prepare":
        prepare(args.lr, args.dst, args.osm)
    elif args.cmd == "run":
        run(args.repo, args.input, args.output, use_map=args.use_map, python=args.python)
    elif args.cmd == "collect":
        collect(args.flat, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
