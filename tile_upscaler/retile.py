"""Cut upscaled images into a deeper-zoom XYZ tile tree for serving.

After super-resolution each source tile ``(z, x, y)`` is a larger image (e.g.
256 -> 1024 px for 4x) covering the *same* geographic extent. To serve it on a
slippy map as sharper, higher-zoom tiles we slice that image into the standard
256 px grid at a deeper zoom:

  - 2x upscale  -> z+1, a 2x2 grid of child tiles
  - 4x upscale  -> z+2, a 4x4 grid of child tiles

Child tile coordinates follow the standard quadtree mapping (see
``tiles.child_tiles``): a child ``(z+L, X, Y)`` maps to the pixel block
``((X - x*2^L) * 256, (Y - y*2^L) * 256)`` of the upscaled image.

Optionally emit 512 px "retina"/@2x tiles instead of 256 px (``--tile-size``).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from . import tileio
from .tiles import Tile, child_tiles, upscale_levels


def retile_image(
    upscaled_image,
    source_tile: Tile,
    out_root: str,
    factor: int = 4,
    tile_size: int = 256,
) -> int:
    """Slice one upscaled image into child tiles written under ``out_root``."""
    from PIL import Image

    levels = upscale_levels(factor)
    grid = 2 ** levels  # children per axis
    expected = tile_size * grid
    if upscaled_image.size != (expected, expected):
        upscaled_image = upscaled_image.resize((expected, expected), Image.BICUBIC)

    written = 0
    base_x = source_tile.x * grid
    base_y = source_tile.y * grid
    for dy in range(grid):
        for dx in range(grid):
            box = (dx * tile_size, dy * tile_size, (dx + 1) * tile_size, (dy + 1) * tile_size)
            crop = upscaled_image.crop(box)
            child = Tile(source_tile.z + levels, base_x + dx, base_y + dy)
            tileio.write_tile(out_root, child, crop)
            written += 1
    return written


def run_tree(
    src_root: str,
    out_root: str,
    factor: int = 4,
    tile_size: int = 256,
    limit: Optional[int] = None,
    tile_keys: Optional[Sequence[str]] = None,
) -> int:
    """Retile every upscaled image under ``src_root`` into ``out_root``."""
    tiles = tileio.find_tiles(src_root)
    tiles = tileio.select_tiles(tiles, limit=limit, tile_keys=tile_keys)
    total = 0
    for tf in tiles:
        img = tileio.load_image(tf.path)
        total += retile_image(img, tf.tile, out_root, factor=factor, tile_size=tile_size)
    levels = upscale_levels(factor)
    print(
        f"Retiled {len(tiles)} images -> {total} tiles at z+{levels} "
        f"({tile_size}px) under {out_root}"
    )
    return total


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Cut upscaled images into XYZ tiles")
    parser.add_argument("--src", required=True, help="Upscaled image tree (z/x/y)")
    parser.add_argument("--out", required=True, help="Output XYZ tile tree (deeper zoom)")
    parser.add_argument("--factor", type=int, default=4, help="Upscale factor used (2 or 4)")
    parser.add_argument("--tile-size", type=int, default=256, help="Output tile px (256 or 512)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    run_tree(
        src_root=args.src,
        out_root=args.out,
        factor=args.factor,
        tile_size=args.tile_size,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
