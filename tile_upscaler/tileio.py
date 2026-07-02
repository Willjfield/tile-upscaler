"""Shared helpers for reading/writing XYZ tile trees.

A tile tree is a directory laid out as ``<root>/<z>/<x>/<y>.<ext>`` - the
standard slippy-map structure. These helpers let the upscalers and eval code
discover, read and write tiles without each re-implementing the path logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from .tiles import Bounds, Tile, tile_intersects_bbox

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class TileFile:
    tile: Tile
    path: str


def find_tiles(root: str) -> List[TileFile]:
    """Discover all ``z/x/y.ext`` tiles under ``root``, sorted by z,x,y."""
    found: List[TileFile] = []
    for z_name in _int_dirs(root):
        z_path = os.path.join(root, z_name)
        for x_name in _int_dirs(z_path):
            x_path = os.path.join(z_path, x_name)
            for fname in os.listdir(x_path):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in IMAGE_EXTS or not stem.isdigit():
                    continue
                tile = Tile(int(z_name), int(x_name), int(stem))
                found.append(TileFile(tile, os.path.join(x_path, fname)))
    found.sort(key=lambda tf: (tf.tile.z, tf.tile.x, tf.tile.y))
    return found


def _int_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        (d for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))),
        key=int,
    )


def tile_path(root: str, tile: Tile, ext: str = "png") -> str:
    """Path for a tile in ``root`` (creating parent dirs is the caller's job)."""
    return os.path.join(root, str(tile.z), str(tile.x), f"{tile.y}.{ext}")


def write_tile(root: str, tile: Tile, image, ext: str = "png") -> str:
    """Write a PIL image to the correct ``z/x/y`` path, making dirs as needed."""
    out = tile_path(root, tile, ext)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    image.save(out)
    return out


def load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def iter_tiles(root: str) -> Iterator[TileFile]:
    yield from find_tiles(root)


def filter_tiles_by_bbox(tiles: List[TileFile], bbox: Bounds) -> List[TileFile]:
    """Keep tiles whose geographic bounds intersect ``bbox`` (preserves order)."""
    return [tf for tf in tiles if tile_intersects_bbox(tf.tile, bbox)]


def select_tiles(
    tiles: List[TileFile],
    *,
    limit: Optional[int] = None,
    tile_keys: Optional[Sequence[str]] = None,
    bbox: Optional[Bounds] = None,
) -> List[TileFile]:
    """Subset ``tiles`` by bbox, explicit keys (preserving key order), or a leading limit."""
    if bbox is not None:
        tiles = filter_tiles_by_bbox(tiles, bbox)
    if tile_keys is not None:
        order = {key: idx for idx, key in enumerate(tile_keys)}
        selected = [tf for tf in tiles if tf.tile.key in order]
        selected.sort(key=lambda tf: order[tf.tile.key])
        return selected
    if limit:
        return tiles[:limit]
    return tiles


def find_companion(root: Optional[str], tile: Tile) -> Optional[str]:
    """Find a tile's file in another tree (e.g. OSM control matching a raster)."""
    if not root:
        return None
    for ext in IMAGE_EXTS:
        cand = tile_path(root, tile, ext.lstrip("."))
        if os.path.exists(cand):
            return cand
    return None
