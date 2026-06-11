"""Render OpenStreetMap vector data into ControlNet control images + text prompts.

For each raster tile we produce two vector-derived guidance signals:

1. A **control image**: the OSM features for the tile's extent rasterised into a
   clean, fixed-palette image (water, vegetation/landuse fills; building
   polygons; roads as lines). This is fed to a spatial ControlNet so the model's
   structure follows real-world boundaries (field edges, roads, water).

2. A **text prompt**: a short natural-language summary of the dominant OSM tags
   in the tile (e.g. "dense residential buildings, a road, trees"), used to
   condition the diffusion model semantically.

The control image is rendered in Web Mercator (EPSG:3857) using the tile's
mercator bounds so it aligns pixel-for-pixel with the raster tile.

OSM sources
-----------
- ``PbfOSMSource``: a local ``.osm.pbf`` extract via ``pyrosm`` (fast, offline,
  reproducible - recommended for batch runs).
- ``OverpassOSMSource``: live Overpass API queries per bbox (handy for tiny
  AOIs / quick experiments; rate-limited, needs network).

Both expose ``features_for_bounds(bounds_lonlat)`` returning GeoDataFrames keyed
by semantic class. Rendering and prompt generation are source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tiles import Bounds, Tile, tile_bounds_3857, tile_bounds_lonlat

# Semantic classes -> RGB color used in the control image. A fixed, distinct
# palette so the ControlNet sees consistent "meanings". Background is black.
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "background": (0, 0, 0),
    "water": (40, 90, 200),
    "vegetation": (60, 160, 60),
    "residential": (170, 130, 90),
    "commercial": (200, 120, 60),
    "building": (220, 60, 60),
    "road_major": (240, 240, 240),
    "road_minor": (160, 160, 160),
    "path": (110, 110, 110),
    "rail": (200, 60, 200),
}

# Road classes drawn as lines, with a render width in *meters* (so line width is
# zoom-consistent). Major roads are drawn wider.
ROAD_WIDTHS_M: Dict[str, float] = {
    "motorway": 18.0,
    "trunk": 16.0,
    "primary": 14.0,
    "secondary": 11.0,
    "tertiary": 9.0,
    "residential": 7.0,
    "service": 5.0,
    "unclassified": 6.0,
    "living_street": 6.0,
    "pedestrian": 4.0,
    "footway": 2.0,
    "path": 2.0,
    "cycleway": 2.0,
}
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}


@dataclass
class TileFeatures:
    """OSM geometries for a tile, grouped by semantic class (in EPSG:4326)."""

    bounds_lonlat: Bounds
    polygons: Dict[str, list] = field(default_factory=dict)  # class -> [shapely geom]
    roads: Dict[str, list] = field(default_factory=dict)  # highway -> [shapely line]
    counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OSM sources
# ---------------------------------------------------------------------------
class OSMSource:
    """Base class. Subclasses return shapely geometries for a lon/lat bbox."""

    def features_for_bounds(self, bounds_lonlat: Bounds) -> TileFeatures:
        raise NotImplementedError


class PbfOSMSource(OSMSource):
    """Load a local ``.osm.pbf`` extract once, then query per tile bbox.

    Requires ``pyrosm`` and ``geopandas``. The whole extract's relevant layers
    are read up front and spatially indexed; per-tile queries use a bbox clip.
    """

    def __init__(self, pbf_path: str):
        try:
            from pyrosm import OSM  # noqa: F401
            import geopandas as gpd  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PbfOSMSource needs pyrosm + geopandas "
                "(pip install pyrosm geopandas)"
            ) from exc
        from pyrosm import OSM

        self._osm = OSM(pbf_path)
        self._buildings = None
        self._landuse = None
        self._natural = None
        self._roads = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # These calls can be slow on big extracts; do them once.
        self._buildings = self._osm.get_buildings()
        self._roads = self._osm.get_network(network_type="all")
        try:
            self._landuse = self._osm.get_landuse()
        except Exception:
            self._landuse = None
        try:
            self._natural = self._osm.get_natural()
        except Exception:
            self._natural = None
        self._loaded = True

    def features_for_bounds(self, bounds_lonlat: Bounds) -> TileFeatures:
        self._ensure_loaded()
        west, south, east, north = bounds_lonlat
        feats = TileFeatures(bounds_lonlat=bounds_lonlat)

        def clip(gdf):
            if gdf is None or len(gdf) == 0:
                return None
            return gdf.cx[west:east, south:north]

        # Polygonal land cover
        for gdf, classifier in (
            (clip(self._natural), _classify_natural),
            (clip(self._landuse), _classify_landuse),
            (clip(self._buildings), lambda row: "building"),
        ):
            if gdf is None:
                continue
            for _, row in gdf.iterrows():
                cls = classifier(row)
                if cls is None or row.geometry is None:
                    continue
                feats.polygons.setdefault(cls, []).append(row.geometry)
                feats.counts[cls] = feats.counts.get(cls, 0) + 1

        roads = clip(self._roads)
        if roads is not None:
            for _, row in roads.iterrows():
                hw = row.get("highway")
                if not hw or row.geometry is None:
                    continue
                feats.roads.setdefault(str(hw), []).append(row.geometry)
                feats.counts["road"] = feats.counts.get("road", 0) + 1
        return feats


class OverpassOSMSource(OSMSource):
    """Query the Overpass API per bbox. Good for tiny AOIs / quick tests."""

    def __init__(self, endpoint: str = "https://overpass-api.de/api/interpreter", timeout: int = 60):
        self.endpoint = endpoint
        self.timeout = timeout

    def features_for_bounds(self, bounds_lonlat: Bounds) -> TileFeatures:
        import json

        import requests
        from shapely.geometry import LineString, Polygon

        west, south, east, north = bounds_lonlat
        bbox = f"{south},{west},{north},{east}"
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["building"]({bbox});
          way["highway"]({bbox});
          way["natural"]({bbox});
          way["landuse"]({bbox});
          way["waterway"]({bbox});
        );
        out geom;
        """
        # overpass-api.de rejects stock library User-Agents (HTTP 406); a
        # descriptive UA identifying the script is required by their usage policy.
        headers = {
            "User-Agent": "tile-upscaler-osm-render/1.0 (https://github.com/Willjfield/tile-upscaler)",
            "Accept": "application/json",
        }
        resp = requests.post(self.endpoint, data={"data": query}, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = json.loads(resp.text)

        feats = TileFeatures(bounds_lonlat=bounds_lonlat)
        for el in data.get("elements", []):
            geom_pts = el.get("geometry")
            if not geom_pts:
                continue
            coords = [(p["lon"], p["lat"]) for p in geom_pts]
            tags = el.get("tags", {})
            if "highway" in tags and len(coords) >= 2:
                feats.roads.setdefault(str(tags["highway"]), []).append(LineString(coords))
                feats.counts["road"] = feats.counts.get("road", 0) + 1
                continue
            if len(coords) < 3:
                continue
            poly = Polygon(coords)
            cls = _classify_tags(tags)
            if cls is None:
                continue
            feats.polygons.setdefault(cls, []).append(poly)
            feats.counts[cls] = feats.counts.get(cls, 0) + 1
        return feats


# ---------------------------------------------------------------------------
# Tag classification
# ---------------------------------------------------------------------------
def _classify_natural(row) -> Optional[str]:
    val = str(row.get("natural") or "").lower()
    if val in {"water", "wetland", "bay"}:
        return "water"
    if val in {"wood", "scrub", "grassland", "heath"}:
        return "vegetation"
    return None


def _classify_landuse(row) -> Optional[str]:
    val = str(row.get("landuse") or "").lower()
    if val in {"forest", "grass", "meadow", "farmland", "orchard", "vineyard", "recreation_ground"}:
        return "vegetation"
    if val in {"residential"}:
        return "residential"
    if val in {"commercial", "retail", "industrial"}:
        return "commercial"
    if val in {"reservoir", "basin"}:
        return "water"
    return None


def _classify_tags(tags: dict) -> Optional[str]:
    if "building" in tags:
        return "building"
    if tags.get("natural") in {"water", "wetland", "bay"} or "waterway" in tags:
        return "water"
    lu = tags.get("landuse")
    if lu in {"forest", "grass", "meadow", "farmland", "orchard", "vineyard"}:
        return "vegetation"
    if lu == "residential":
        return "residential"
    if lu in {"commercial", "retail", "industrial"}:
        return "commercial"
    return None


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------
def render_control_image(
    tile: Tile,
    features: TileFeatures,
    size: int = 1024,
) -> "np.ndarray":
    """Rasterise ``features`` into an RGB control image of ``size`` x ``size``.

    Rendered in Web Mercator so it aligns with the raster tile. Returns a uint8
    HxWx3 array (use ``Image.fromarray`` to save).
    """
    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from shapely.geometry import LineString
        from shapely.ops import transform as shp_transform
        import pyproj
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "render_control_image needs rasterio, shapely and pyproj "
            "(pip install rasterio shapely pyproj)"
        ) from exc

    west, south, east, north = tile_bounds_3857(tile)
    transform = from_bounds(west, south, east, north, size, size)
    meters_per_px = (east - west) / size

    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    def burn(geoms, color, buffer_m: float = 0.0):
        if not geoms:
            return
        shapes = []
        for g in geoms:
            gm = shp_transform(project, g)
            if buffer_m > 0:
                gm = gm.buffer(buffer_m)
            shapes.append((gm, 1))
        if not shapes:
            return
        mask = rasterize(
            shapes,
            out_shape=(size, size),
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=True,
            dtype="uint8",
        )
        canvas[mask.astype(bool)] = color

    # Draw order: land cover first, then buildings, then roads on top.
    burn(features.polygons.get("vegetation"), PALETTE["vegetation"])
    burn(features.polygons.get("residential"), PALETTE["residential"])
    burn(features.polygons.get("commercial"), PALETTE["commercial"])
    burn(features.polygons.get("water"), PALETTE["water"])
    burn(features.polygons.get("building"), PALETTE["building"])

    for hw, lines in features.roads.items():
        width_m = ROAD_WIDTHS_M.get(hw, 4.0)
        buf = max(width_m / 2.0, meters_per_px)  # at least one pixel wide
        if hw in MAJOR_ROADS:
            color = PALETTE["road_major"]
        elif hw in {"footway", "path", "cycleway", "steps"}:
            color = PALETTE["path"]
        else:
            color = PALETTE["road_minor"]
        burn(lines, color, buffer_m=buf)

    return canvas


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------
def build_prompt(features: TileFeatures, base: str = "high-resolution aerial orthophoto, top-down satellite view") -> str:
    """Summarise the tile's OSM content into a text prompt for the diffuser."""
    counts = features.counts
    phrases: List[str] = []

    n_buildings = counts.get("building", 0)
    if n_buildings >= 25:
        phrases.append("dense buildings and rooftops")
    elif n_buildings >= 5:
        phrases.append("scattered buildings with rooftops")
    elif n_buildings >= 1:
        phrases.append("a few isolated buildings")

    if counts.get("residential"):
        phrases.append("residential housing")
    if counts.get("commercial"):
        phrases.append("commercial or industrial structures")
    if counts.get("road"):
        phrases.append("roads and pavement")
    if counts.get("vegetation"):
        phrases.append("trees, grass and vegetation")
    if counts.get("water"):
        phrases.append("water")

    if not phrases:
        phrases.append("open natural terrain")

    return f"{base}, {', '.join(phrases)}, sharp detail, realistic textures"


def render_tile(
    source: OSMSource,
    tile: Tile,
    size: int = 1024,
) -> Tuple["np.ndarray", str]:
    """Convenience: fetch features and return (control_image, prompt)."""
    feats = source.features_for_bounds(tile_bounds_lonlat(tile))
    control = render_control_image(tile, feats, size=size)
    prompt = build_prompt(feats)
    return control, prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _make_source(args) -> OSMSource:
    if args.pbf:
        return PbfOSMSource(args.pbf)
    return OverpassOSMSource()


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse
    import json
    import os

    from PIL import Image

    parser = argparse.ArgumentParser(
        description="Render OSM control images + prompts for tiles"
    )
    parser.add_argument("--pbf", help="Path to a local .osm.pbf (else Overpass)")
    parser.add_argument("--tile", help="Single tile z/x/y")
    parser.add_argument("--tiles-file", help="File with one z/x/y per line")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args(argv)

    source = _make_source(args)

    keys: List[str] = []
    if args.tile:
        keys.append(args.tile)
    if args.tiles_file:
        with open(args.tiles_file) as fh:
            keys.extend(line.strip() for line in fh if line.strip())
    if not keys:
        parser.error("provide --tile or --tiles-file")

    os.makedirs(args.out, exist_ok=True)
    prompts = {}
    for key in keys:
        z, x, y = (int(v) for v in key.split("/"))
        tile = Tile(z, x, y)
        control, prompt = render_tile(source, tile, size=args.size)
        out_dir = os.path.join(args.out, str(z), str(x))
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(control).save(os.path.join(out_dir, f"{y}.png"))
        prompts[key] = prompt
        print(f"{key}: {prompt}")

    with open(os.path.join(args.out, "prompts.json"), "w") as fh:
        json.dump(prompts, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
