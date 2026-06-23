#!/usr/bin/env python3
"""End-to-end experiment runner for vector-guided aerial tile upscaling.

Reads ``config.yaml`` (see ``config.example.yaml``) and runs the full comparison:

  1. (optional) Build a degraded LR set from genuine HR tiles for a ground-truth
     degradation test.
  2. Render OSM control images + text prompts for the tiles.
  3. Upscale with each enabled method:
       A  baseline Real-ESRGAN / Swin2SR        (no vector)
       B  SDXL + Tile ControlNet + OSM prompt    (text-guided vector)
       C  B + OSM-edge ControlNet                (spatial vector guidance)
  4. Re-cut each upscaled tree into deeper-zoom XYZ tiles for serving.
  5. Evaluate: PSNR/SSIM/LPIPS (if HR), cross-scale consistency, no-reference
     metrics, and side-by-side comparison sheets.

This is deliberately a thin orchestrator over the library modules so you can also
run any single step by hand (see the README).
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, Optional

import yaml

from tile_upscaler import eval as ev
from tile_upscaler import osm_render, retile, tileio
from tile_upscaler import upscale_baseline, upscale_controlnet


def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _first_tile_size(root: str) -> int:
    tiles = tileio.find_tiles(root)
    if not tiles:
        raise SystemExit(f"No tiles found under {root}")
    return tileio.load_image(tiles[0].path).width


def _render_osm(
    lr_root: str,
    out_osm: str,
    pbf: Optional[str],
    control_size: int,
    edge_line_width_px: float = 2.0,
) -> str:
    """Render OSM control images for every LR tile; return prompts.json path."""
    import json

    from PIL import Image

    if pbf and os.path.exists(pbf):
        source = osm_render.PbfOSMSource(pbf)
        print(f"OSM source: local pbf {pbf}")
    else:
        source = osm_render.OverpassOSMSource()
        print("OSM source: Overpass API (no local pbf configured)")

    prompts: Dict[str, str] = {}
    tiles = tileio.find_tiles(lr_root)
    edges_root = os.path.join(out_osm, "edges")
    for i, tf in enumerate(tiles, 1):
        try:
            control, edges, prompt = osm_render.render_tile(
                source, tf.tile, size=control_size, edge_line_width_px=edge_line_width_px,
            )
        except Exception as exc:  # keep going; a tile without OSM coverage is fine
            print(f"  [warn] OSM render failed for {tf.tile.key}: {exc}")
            continue
        tileio.write_tile(out_osm, tf.tile, Image.fromarray(control))
        tileio.write_tile(edges_root, tf.tile, Image.fromarray(edges))
        prompts[tf.tile.key] = prompt
        if i % 25 == 0:
            print(f"  rendered OSM {i}/{len(tiles)}")
    prompts_path = os.path.join(out_osm, "prompts.json")
    os.makedirs(out_osm, exist_ok=True)
    with open(prompts_path, "w") as fh:
        json.dump(prompts, fh, indent=2)
    print(f"Rendered OSM controls for {len(prompts)} tiles -> {out_osm} (edges -> {edges_root})")
    return prompts_path


def _archive_out(out_dir: str, archive_path: str) -> str:
    """Zip ``out_dir`` into ``archive_path`` (``.zip`` added if missing).

    The archive contains a single top-level ``out/`` folder (or whatever
    ``out_dir`` is named). Intended for one-shot download from RunPod via S3.
    """
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Cannot archive: {out_dir} is not a directory")
    base, ext = os.path.splitext(archive_path)
    if ext.lower() != ".zip":
        base = archive_path
    else:
        archive_path = base + ".zip"
    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    name = os.path.basename(out_dir.rstrip(os.sep))
    print(f"\nPacking {out_dir}/ -> {base}.zip ...")
    shutil.make_archive(base, "zip", root_dir=parent, base_dir=name)
    size_mb = os.path.getsize(base + ".zip") / (1024 * 1024)
    print(f"Archive ready: {base}.zip ({size_mb:.1f} MB)")
    return base + ".zip"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full upscaling experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Cap tiles per method (debug)")
    parser.add_argument("--skip-osm", action="store_true", help="Reuse existing OSM render")
    parser.add_argument(
        "--archive-out",
        metavar="PATH",
        help="Zip paths.out to PATH when done (overrides config paths.archive_out)",
    )
    parser.add_argument(
        "--archive-only",
        metavar="PATH",
        help="Only zip paths.out to PATH and exit (no upscaling)",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    paths = cfg["paths"]
    out = paths["out"]

    if args.archive_only:
        _archive_out(out, args.archive_only)
        return 0

    factor = int(cfg["upscale"]["factor"])
    device = cfg["upscale"].get("device")
    os.makedirs(out, exist_ok=True)

    # --- 1. resolve LR / HR roots ---------------------------------------------
    hr_root = (cfg.get("eval") or {}).get("hr_root")
    if hr_root:
        lr_root = os.path.join(out, "lr")
        print(f"Degradation test enabled: building LR from HR ({hr_root}) /{factor}")
        ev.make_degraded(hr_root, lr_root, factor=factor)
    else:
        lr_root = paths["raster"]
        print(f"No HR provided; upscaling source raster tiles directly ({lr_root})")

    lr_size = _first_tile_size(lr_root)
    target_size = lr_size * factor

    # --- 2. OSM render --------------------------------------------------------
    osm_tree = os.path.join(out, "osm")
    prompts_path = os.path.join(osm_tree, "prompts.json")
    need_osm = cfg["methods"].get("controlnet_text") or cfg["methods"].get("controlnet_osm")
    osm_cfg = cfg.get("osm") or {}
    if need_osm and not args.skip_osm:
        prompts_path = _render_osm(
            lr_root,
            osm_tree,
            paths.get("osm_pbf"),
            target_size,
            edge_line_width_px=float(osm_cfg.get("edge_line_width_px", 2.0)),
        )

    # --- 3. run methods -------------------------------------------------------
    produced: Dict[str, str] = {}
    m = cfg["methods"]
    d = cfg["diffusion"]

    if m.get("baseline_realesrgan"):
        dst = os.path.join(out, "up", "A_realesrgan")
        print("\n=== Method A: Real-ESRGAN (no vector) ===")
        upscale_baseline.run_tree(lr_root, dst, "realesrgan", factor, device, args.limit)
        produced["A_realesrgan"] = dst

    if m.get("baseline_swin2sr"):
        dst = os.path.join(out, "up", "A_swin2sr")
        print("\n=== Method A2: Swin2SR (no vector) ===")
        upscale_baseline.run_tree(lr_root, dst, "swin2sr", factor, device, args.limit)
        produced["A_swin2sr"] = dst

    def _diff_config(use_osm: bool) -> upscale_controlnet.UpscaleConfig:
        return upscale_controlnet.UpscaleConfig(
            outscale=factor,
            steps=int(d["steps"]),
            strength=float(d["strength"]),
            guidance_scale=float(d["guidance_scale"]),
            tile_cond_scale=float(d["tile_cond_scale"]),
            osm_cond_scale=float(d["osm_cond_scale"]),
            use_osm=use_osm,
            seed=int(d["seed"]),
            color_fix=d.get("color_fix", "wavelet"),
            device=device,
        )

    if m.get("controlnet_text"):
        dst = os.path.join(out, "up", "B_controlnet_text")
        print("\n=== Method B: SDXL + Tile ControlNet + OSM text prompt ===")
        upscale_controlnet.run_tree(
            lr_root, dst, _diff_config(use_osm=False),
            osm_root=osm_tree, prompts_path=prompts_path, limit=args.limit,
        )
        produced["B_controlnet_text"] = dst

    if m.get("controlnet_osm"):
        dst = os.path.join(out, "up", "C_controlnet_osm")
        print("\n=== Method C: + OSM-edge ControlNet (spatial vector guidance) ===")
        upscale_controlnet.run_tree(
            lr_root, dst, _diff_config(use_osm=True),
            osm_root=osm_tree, prompts_path=prompts_path, limit=args.limit,
        )
        produced["C_controlnet_osm"] = dst

    # --- 4. retile for serving ------------------------------------------------
    for name, src in produced.items():
        serve = os.path.join(out, "serve", name)
        retile.run_tree(src, serve, factor=factor)

    # --- 5. evaluate ----------------------------------------------------------
    print("\n#### EVALUATION ####")
    metrics_dir = os.path.join(out, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    for name, src in produced.items():
        if hr_root:
            ev.score_against_hr(hr_root, src, os.path.join(metrics_dir, f"{name}_score.csv"))
        ev.consistency(lr_root, src, os.path.join(metrics_dir, f"{name}_consistency.csv"))
        if (cfg.get("eval") or {}).get("no_reference"):
            ev.no_reference(src, csv_path=os.path.join(metrics_dir, f"{name}_nr.csv"))

    if (cfg.get("eval") or {}).get("comparison_sheets") and produced:
        ev.comparison_sheets(
            lr_root, produced, os.path.join(out, "sheets"),
            hr_root=hr_root, limit=args.limit,
        )

    print("\nDone. Upscaled trees in out/up/, servable tiles in out/serve/, "
          "metrics in out/metrics/, sheets in out/sheets/.")

    archive_path = args.archive_out or paths.get("archive_out")
    if archive_path:
        _archive_out(out, archive_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
