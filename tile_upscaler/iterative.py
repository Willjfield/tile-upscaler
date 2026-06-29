"""Iterative zoom-layer upscaling: z18 -> z22 -> z25, etc.

Each layer upscales the previous layer's tile tree, retiles to the next zoom
level, and feeds that output into the next pass. Method settings in
``config.yaml`` are reused; only the per-layer pixel factor (``2 ** zoom_delta``)
changes.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Dict, List, Optional, Sequence, Set

from . import retile, tileio
from . import upscale_baseline, upscale_controlnet
from .tiles import ancestor_tile, pixel_factor_for_zoom_delta, plan_iterative_zoom_ladder

RenderOsmFn = Callable[
    [str, str, int, float, Optional[Sequence[str]]],
    str,
]
RunEvaluationFn = Callable[..., None]


def _detect_raster_zoom(raster_root: str) -> int:
    zooms = {tf.tile.z for tf in tileio.find_tiles(raster_root)}
    if len(zooms) != 1:
        raise SystemExit(
            f"Expected a single zoom level under {raster_root}, found {sorted(zooms)}"
        )
    return next(iter(zooms))


def _iterative_cfg(upscale: dict) -> dict:
    return upscale.get("iterative") or {}


def resolve_zoom_ladder(upscale: dict, raster_root: str) -> List[int]:
    icfg = _iterative_cfg(upscale)
    initial_z = icfg.get("initial_zoom")
    if initial_z is None:
        initial_z = _detect_raster_zoom(raster_root)
    final_z = icfg.get("final_zoom")
    if final_z is None:
        raise SystemExit("upscale.iterative.final_zoom is required when iterative mode is enabled")
    zoom_step = int(icfg.get("zoom_step", upscale.get("factor", 4)))
    return plan_iterative_zoom_ladder(int(initial_z), int(final_z), zoom_step)


def _enabled_methods(cfg: dict) -> List[str]:
    m = cfg["methods"]
    names: List[str] = []
    if m.get("baseline_realesrgan"):
        names.append("A_realesrgan")
    if m.get("baseline_swin2sr"):
        names.append("A_swin2sr")
    if m.get("controlnet_text"):
        names.append("B_controlnet_text")
    if m.get("controlnet_osm"):
        names.append("C_controlnet_osm")
    return names


def _layer_tile_keys(
    lr_root: str,
    z_from: int,
    initial_z: int,
    ancestor_keys: Optional[Set[str]],
    first_layer_keys: Optional[Sequence[str]],
) -> List[str]:
    tiles = tileio.find_tiles(lr_root)
    if ancestor_keys is not None:
        tiles = [
            tf for tf in tiles
            if ancestor_tile(tf.tile, initial_z).key in ancestor_keys
        ]
    if z_from == initial_z and first_layer_keys is not None:
        tiles = tileio.select_tiles(tiles, tile_keys=first_layer_keys)
    return [tf.tile.key for tf in tiles]


def _run_one_method(
    cfg: dict,
    method_name: str,
    lr_root: str,
    method_dir: str,
    pixel_factor: int,
    device: Optional[str],
    layer_tile_keys: List[str],
    osm_tree: str,
    prompts_path: str,
) -> str:
    """Upscale one method for the current layer; return the upscaled tree path."""
    up_root = os.path.join(method_dir, "up")
    m = cfg["methods"]
    d = cfg["diffusion"]

    if method_name == "A_realesrgan":
        print("\n=== Method A: Real-ESRGAN (no vector) ===")
        upscale_baseline.run_tree(
            lr_root, up_root, "realesrgan", pixel_factor, device,
            tile_keys=layer_tile_keys,
        )
        return up_root

    if method_name == "A_swin2sr":
        print("\n=== Method A2: Swin2SR (no vector) ===")
        upscale_baseline.run_tree(
            lr_root, up_root, "swin2sr", pixel_factor, device,
            tile_keys=layer_tile_keys,
        )
        return up_root

    def _diff_config(use_osm: bool) -> upscale_controlnet.UpscaleConfig:
        return upscale_controlnet.UpscaleConfig(
            outscale=pixel_factor,
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

    if method_name == "B_controlnet_text":
        print("\n=== Method B: SDXL + Tile ControlNet + OSM text prompt ===")
        upscale_controlnet.run_tree(
            lr_root, up_root, _diff_config(use_osm=False),
            osm_root=osm_tree, prompts_path=prompts_path,
            tile_keys=layer_tile_keys,
        )
        return up_root

    if method_name == "C_controlnet_osm":
        print("\n=== Method C: + OSM-edge ControlNet (spatial vector guidance) ===")
        upscale_controlnet.run_tree(
            lr_root, up_root, _diff_config(use_osm=True),
            osm_root=osm_tree, prompts_path=prompts_path,
            tile_keys=layer_tile_keys,
        )
        return up_root

    raise ValueError(f"Unknown method: {method_name}")


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copytree(src, dst)


def run_iterative(
    cfg: dict,
    *,
    paths: dict,
    device: Optional[str],
    tile_keys: Optional[Sequence[str]],
    skip_osm: bool,
    render_osm_fn: RenderOsmFn,
    run_evaluation_fn: RunEvaluationFn,
) -> List[int]:
    """Run the full iterative ladder; return the zoom checkpoint list."""
    upscale = cfg["upscale"]
    out = paths["out"]
    raster_root = paths["raster"]
    ladder = resolve_zoom_ladder(upscale, raster_root)
    initial_z = ladder[0]
    ancestor_keys: Optional[Set[str]] = set(tile_keys) if tile_keys else None

    iter_root = os.path.join(out, "iterative")
    os.makedirs(iter_root, exist_ok=True)
    ladder_path = os.path.join(iter_root, "zoom_ladder.json")

    print(f"Iterative upscaling ladder: {' -> '.join(str(z) for z in ladder)}")

    method_names = _enabled_methods(cfg)
    if not method_names:
        raise SystemExit("No methods enabled in config.yaml")

    method_inputs: Dict[str, str] = {name: raster_root for name in method_names}
    layer_records: List[dict] = []
    final_up: Dict[str, str] = {}
    final_serve: Dict[str, str] = {}
    final_lr: Dict[str, str] = {}

    m = cfg["methods"]
    need_osm = m.get("controlnet_text") or m.get("controlnet_osm")
    osm_cfg = cfg.get("osm") or {}

    for z_from, z_to in zip(ladder, ladder[1:]):
        pixel_factor = pixel_factor_for_zoom_delta(z_to - z_from)
        layer_name = f"z{z_from}_z{z_to}"
        layer_dir = os.path.join(iter_root, "layers", layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        print(
            f"\n{'=' * 72}\n"
            f"LAYER z{z_from} -> z{z_to}  (pixel factor x{pixel_factor})\n"
            f"{'=' * 72}"
        )

        layer_records.append({
            "name": layer_name,
            "z_from": z_from,
            "z_to": z_to,
            "pixel_factor": pixel_factor,
        })

        sample_lr = method_inputs[method_names[0]]
        layer_tile_keys = _layer_tile_keys(
            sample_lr, z_from, initial_z, ancestor_keys, tile_keys,
        )
        if not layer_tile_keys:
            raise SystemExit(
                f"No tiles for layer z{z_from}->z{z_to} under {sample_lr}"
            )

        first_tf = next(
            tf for tf in tileio.find_tiles(sample_lr) if tf.tile.key in set(layer_tile_keys)
        )
        target_size = tileio.load_image(first_tf.path).width * pixel_factor

        layer_osm = os.path.join(layer_dir, "osm")
        prompts_path = os.path.join(layer_osm, "prompts.json")
        if need_osm and not skip_osm:
            prompts_path = render_osm_fn(
                sample_lr,
                layer_osm,
                target_size,
                float(osm_cfg.get("edge_line_width_px", 2.0)),
                layer_tile_keys,
            )

        for method_name in method_names:
            lr_root = method_inputs[method_name]
            method_dir = os.path.join(layer_dir, method_name)
            os.makedirs(method_dir, exist_ok=True)

            up_path = _run_one_method(
                cfg, method_name, lr_root, method_dir, pixel_factor, device,
                layer_tile_keys, layer_osm, prompts_path,
            )
            serve_path = os.path.join(method_dir, "serve")
            retile.run_tree(up_path, serve_path, factor=pixel_factor, tile_keys=layer_tile_keys)

            method_inputs[method_name] = serve_path
            final_up[method_name] = up_path
            final_serve[method_name] = serve_path
            final_lr[method_name] = lr_root

    with open(ladder_path, "w") as fh:
        json.dump({"zoom_levels": ladder, "layers": layer_records}, fh, indent=2)
        fh.write("\n")
    print(f"Wrote ladder manifest -> {ladder_path}")

    for method_name in method_names:
        _link_or_copy(final_up[method_name], os.path.join(out, "up", method_name))
        _link_or_copy(final_serve[method_name], os.path.join(out, "serve", method_name))

    if cfg.get("eval"):
        hr_root = (cfg.get("eval") or {}).get("hr_root")
        # Use the first method's pre-final raster as LR reference for sheets.
        ref_lr = final_lr[method_names[0]]
        run_evaluation_fn(
            cfg,
            paths,
            final_up,
            ref_lr,
            hr_root,
            tile_keys=tile_keys,
        )

    print(
        f"\nIterative run complete. Final zoom z{ladder[-1]}. "
        f"Artifacts under {iter_root}/layers/; finals mirrored to {out}/up/ and {out}/serve/."
    )
    return ladder
