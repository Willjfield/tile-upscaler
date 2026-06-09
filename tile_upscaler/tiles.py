"""Slippy-map tile math and AOI enumeration.

XYZ / "slippy map" tiles follow the Web Mercator (EPSG:3857) scheme used by
OpenStreetMap, Google, Mapbox, etc. This module provides the coordinate math to:

- convert between lon/lat and tile x/y/z,
- compute a tile's geographic bounds (in lon/lat *and* Web Mercator meters),
- enumerate all tiles covering an area of interest (AOI),
- build a padded "context window" of neighbour tiles (to reduce diffusion seams),
- map a parent tile to its child tiles at a deeper zoom (for re-tiling after
  super-resolution).

Rasterising OSM data needs Web Mercator bounds (``tile_bounds_3857``) so that the
control image lines up pixel-for-pixel with the raster tile, which is itself
rendered in Web Mercator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

# Web Mercator constants (EPSG:3857)
EARTH_RADIUS_M = 6378137.0
ORIGIN_SHIFT_M = math.pi * EARTH_RADIUS_M  # 20037508.342789244

LonLat = Tuple[float, float]
Bounds = Tuple[float, float, float, float]  # (west, south, east, north)


@dataclass(frozen=True)
class Tile:
    """An XYZ tile coordinate."""

    z: int
    x: int
    y: int

    def __iter__(self) -> Iterator[int]:
        yield self.z
        yield self.x
        yield self.y

    @property
    def key(self) -> str:
        return f"{self.z}/{self.x}/{self.y}"


# ---------------------------------------------------------------------------
# lon/lat <-> tile
# ---------------------------------------------------------------------------
def lonlat_to_tile(lon: float, lat: float, z: int) -> Tile:
    """Return the tile that contains the given lon/lat at zoom ``z``."""
    lat = max(min(lat, 85.0511287798), -85.0511287798)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = min(max(x, 0), n - 1)
    y = min(max(y, 0), n - 1)
    return Tile(z, x, y)


def tile_nw_lonlat(tile: Tile) -> LonLat:
    """North-west corner (lon, lat) of a tile."""
    n = 2 ** tile.z
    lon = tile.x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile.y / n)))
    return lon, math.degrees(lat_rad)


def tile_bounds_lonlat(tile: Tile) -> Bounds:
    """Geographic bounds (west, south, east, north) of a tile in lon/lat."""
    west, north = tile_nw_lonlat(tile)
    east, south = tile_nw_lonlat(Tile(tile.z, tile.x + 1, tile.y + 1))
    return west, south, east, north


# ---------------------------------------------------------------------------
# Web Mercator (meters) helpers
# ---------------------------------------------------------------------------
def tile_bounds_3857(tile: Tile) -> Bounds:
    """Bounds (west, south, east, north) of a tile in Web Mercator meters.

    Use these bounds to build the affine transform for rasterising OSM data so
    that the control image aligns pixel-for-pixel with the raster tile.
    """
    n = 2 ** tile.z
    tile_size_m = (2 * ORIGIN_SHIFT_M) / n
    west = -ORIGIN_SHIFT_M + tile.x * tile_size_m
    east = west + tile_size_m
    north = ORIGIN_SHIFT_M - tile.y * tile_size_m
    south = north - tile_size_m
    return west, south, east, north


def lonlat_to_3857(lon: float, lat: float) -> Tuple[float, float]:
    """Project lon/lat (EPSG:4326) to Web Mercator meters (EPSG:3857)."""
    x = math.radians(lon) * EARTH_RADIUS_M
    lat = max(min(lat, 85.0511287798), -85.0511287798)
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * EARTH_RADIUS_M
    return x, y


# ---------------------------------------------------------------------------
# AOI enumeration
# ---------------------------------------------------------------------------
def tiles_for_bbox(bbox: Bounds, z: int) -> List[Tile]:
    """Enumerate every tile at zoom ``z`` covering a lon/lat bbox.

    ``bbox`` is (west, south, east, north) in degrees.
    """
    west, south, east, north = bbox
    nw = lonlat_to_tile(west, north, z)
    se = lonlat_to_tile(east, south, z)
    x0, x1 = min(nw.x, se.x), max(nw.x, se.x)
    y0, y1 = min(nw.y, se.y), max(nw.y, se.y)
    return [Tile(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def neighbours(tile: Tile, radius: int = 1) -> List[Tile]:
    """Return the (2*radius+1)^2 block of tiles centred on ``tile``.

    Used for "context padding": process the centre tile with surrounding pixels
    available so a diffusion model produces consistent edges, then keep only the
    centre region. Out-of-range tiles are clamped to the valid range.
    """
    n = 2 ** tile.z
    out: List[Tile] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = (tile.x + dx) % n  # wrap horizontally (longitude is cyclic)
            y = min(max(tile.y + dy, 0), n - 1)
            out.append(Tile(tile.z, x, y))
    return out


def child_tiles(tile: Tile, levels: int = 2) -> List[Tile]:
    """Return the child tiles covering ``tile`` at zoom ``tile.z + levels``.

    Each step deeper doubles linear resolution, so ``levels`` children form a
    ``2**levels`` x ``2**levels`` grid. This is the inverse of the upscale: a 4x
    (levels=2) super-resolved image re-cuts into these tiles.
    """
    factor = 2 ** levels
    base_x = tile.x * factor
    base_y = tile.y * factor
    return [
        Tile(tile.z + levels, base_x + dx, base_y + dy)
        for dy in range(factor)
        for dx in range(factor)
    ]


def upscale_levels(factor: int) -> int:
    """Number of zoom levels gained by a linear upscale ``factor`` (2->1, 4->2)."""
    levels = int(round(math.log2(factor)))
    if 2 ** levels != factor:
        raise ValueError(f"Upscale factor {factor} is not a power of two")
    return levels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_bbox(text: str) -> Bounds:
    parts = [float(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'west,south,east,north'")
    return parts[0], parts[1], parts[2], parts[3]


def main(argv: Iterable[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Slippy-map tile utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List tiles covering a bbox at a zoom")
    p_list.add_argument("--bbox", required=True, help="west,south,east,north (deg)")
    p_list.add_argument("--zoom", type=int, required=True)
    p_list.add_argument("--json", action="store_true", help="Emit JSON")

    p_bounds = sub.add_parser("bounds", help="Print bounds of a z/x/y tile")
    p_bounds.add_argument("tile", help="z/x/y")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "list":
        tiles = tiles_for_bbox(_parse_bbox(args.bbox), args.zoom)
        if args.json:
            print(json.dumps([t.key for t in tiles]))
        else:
            for t in tiles:
                print(t.key)
            print(f"# {len(tiles)} tiles", flush=True)
    elif args.cmd == "bounds":
        z, x, y = (int(v) for v in args.tile.split("/"))
        t = Tile(z, x, y)
        print("lonlat:", tile_bounds_lonlat(t))
        print("3857  :", tile_bounds_3857(t))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
