"""Render OpenStreetMap vector data into ControlNet control images + text prompts.

For each raster tile we produce two vector-derived guidance signals:

1. A **control image**: the OSM features for the tile's extent rasterised into a
   clean, fixed-palette image (water, vegetation/landuse fills; building
   polygons; roads as lines). Kept for debugging and comparison.

2. A **building-edge image**: building footprint outlines only (white on black),
   written to ``edges/`` and fed directly to the OSM ControlNet (variant C).

3. A **text prompt**: a natural-language summary of OSM tags in the tile —
   building types, roof shape/material/colour, materials, levels, named roads,
   addr:street, landuse, surfaces, sport, man_made, and other aerial cues.

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

# Human-readable labels for common OSM tag values used in text prompts.
BUILDING_LABELS: Dict[str, str] = {
    "yes": "buildings",
    "house": "houses",
    "residential": "residential buildings",
    "apartments": "apartment blocks",
    "terrace": "terraced houses",
    "detached": "detached houses",
    "semidetached_house": "semi-detached houses",
    "bungalow": "bungalows",
    "commercial": "commercial buildings",
    "industrial": "industrial buildings",
    "retail": "retail buildings",
    "warehouse": "warehouse buildings",
    "church": "a church",
    "school": "a school building",
    "hospital": "a hospital building",
    "university": "university buildings",
    "garage": "garages",
    "shed": "sheds",
    "roof": "roof structures",
}
LANDUSE_LABELS: Dict[str, str] = {
    "residential": "residential land",
    "commercial": "commercial land",
    "industrial": "industrial land",
    "retail": "retail land",
    "grass": "grassy areas",
    "meadow": "meadow",
    "farmland": "farmland",
    "forest": "forest",
    "orchard": "orchard",
    "vineyard": "vineyard",
    "recreation_ground": "recreation ground",
    "cemetery": "cemetery",
    "construction": "construction site",
}
NATURAL_LABELS: Dict[str, str] = {
    "wood": "woodland",
    "scrub": "scrubland",
    "grassland": "grassland",
    "heath": "heathland",
    "water": "open water",
    "wetland": "wetland",
    "bay": "bay",
}
HIGHWAY_LABELS: Dict[str, str] = {
    "motorway": "motorway",
    "trunk": "trunk road",
    "primary": "primary road",
    "secondary": "secondary road",
    "tertiary": "tertiary road",
    "residential": "residential street",
    "living_street": "living street",
    "service": "service road",
    "pedestrian": "pedestrian area",
    "footway": "footpath",
    "path": "path",
    "cycleway": "cycleway",
}
AMENITY_LABELS: Dict[str, str] = {
    "parking": "parking area",
    "school": "school grounds",
    "hospital": "hospital grounds",
    "place_of_worship": "place of worship",
    "university": "university campus",
}
LEISURE_LABELS: Dict[str, str] = {
    "park": "park",
    "garden": "garden",
    "pitch": "sports pitch",
    "playground": "playground",
    "golf_course": "golf course",
    "stadium": "stadium",
    "track": "running track",
    "nature_reserve": "nature reserve",
}
ROOF_SHAPE_LABELS: Dict[str, str] = {
    "flat": "flat rooftops",
    "gabled": "gabled rooftops",
    "hipped": "hipped rooftops",
    "pyramidal": "pyramidal rooftops",
    "mansard": "mansard rooftops",
    "gambrel": "gambrel rooftops",
    "skillion": "skillion rooftops",
    "round": "round rooftops",
    "dome": "domed rooftops",
    "onion": "onion domes",
    "saltbox": "saltbox rooftops",
    "butterfly": "butterfly rooftops",
}
ROOF_MATERIAL_LABELS: Dict[str, str] = {
    "slate": "slate rooftops",
    "tiles": "tiled rooftops",
    "tile": "tiled rooftops",
    "metal": "metal rooftops",
    "steel": "metal rooftops",
    "concrete": "concrete rooftops",
    "tar_paper": "tar paper rooftops",
    "thatch": "thatched rooftops",
    "glass": "glass rooftops",
    "copper": "copper rooftops",
    "asbestos": "corrugated rooftops",
    "roof_tiles": "clay tile rooftops",
    "shingle": "shingle rooftops",
}
BUILDING_MATERIAL_LABELS: Dict[str, str] = {
    "brick": "brick buildings",
    "stone": "stone buildings",
    "concrete": "concrete buildings",
    "wood": "wooden buildings",
    "metal": "metal-clad buildings",
    "glass": "glass-fronted buildings",
    "steel": "steel-framed buildings",
    "sandstone": "sandstone buildings",
    "limestone": "limestone buildings",
    "marble": "stone buildings",
    "plaster": "rendered buildings",
    "masonry": "masonry buildings",
}
COLOUR_LABELS: Dict[str, str] = {
    "red": "red rooftops",
    "grey": "grey rooftops",
    "gray": "grey rooftops",
    "brown": "brown rooftops",
    "black": "dark rooftops",
    "white": "light rooftops",
    "blue": "blue rooftops",
    "green": "green rooftops",
    "orange": "orange rooftops",
    "yellow": "yellow rooftops",
    "beige": "beige rooftops",
    "tan": "tan rooftops",
}
SURFACE_LABELS: Dict[str, str] = {
    "asphalt": "asphalt roads",
    "paved": "paved surfaces",
    "concrete": "concrete pavement",
    "paving_stones": "cobbled streets",
    "sett": "cobbled streets",
    "gravel": "gravel paths",
    "compacted": "compacted gravel",
    "grass": "grass surfaces",
    "dirt": "dirt paths",
    "unpaved": "unpaved tracks",
    "wood": "wooden boardwalk",
}
MAN_MADE_LABELS: Dict[str, str] = {
    "silo": "silos",
    "storage_tank": "storage tanks",
    "chimney": "chimneys",
    "bridge": "bridge",
    "pier": "pier",
    "tower": "tower",
    "water_tower": "water tower",
    "works": "industrial works",
    "pipeline": "pipeline",
    "mast": "communications mast",
    "surveillance": "surveillance mast",
    "crane": "construction crane",
    "gasometer": "gas holder",
}
SPORT_LABELS: Dict[str, str] = {
    "soccer": "football pitch",
    "football": "football pitch",
    "tennis": "tennis courts",
    "basketball": "basketball court",
    "baseball": "baseball field",
    "cricket": "cricket pitch",
    "rugby": "rugby pitch",
    "golf": "golf course",
    "swimming": "swimming pool",
    "athletics": "athletics track",
}
SHOP_LABELS: Dict[str, str] = {
    "supermarket": "supermarket",
    "convenience": "convenience store",
    "mall": "shopping centre",
    "department_store": "department store",
    "retail": "retail units",
}
TOURISM_LABELS: Dict[str, str] = {
    "hotel": "hotel",
    "hostel": "hostel",
    "museum": "museum",
    "attraction": "tourist attraction",
    "viewpoint": "viewpoint",
}
HISTORIC_LABELS: Dict[str, str] = {
    "building": "historic building",
    "castle": "castle",
    "ruins": "ruins",
    "monument": "monument",
    "church": "historic church",
    "yes": "historic structure",
}
WATER_TYPE_LABELS: Dict[str, str] = {
    "pond": "pond",
    "reservoir": "reservoir",
    "lake": "lake",
    "river": "river",
    "canal": "canal",
    "stream": "stream",
    "basin": "basin",
    "dock": "dock",
    "marina": "marina",
}

# OSM tag keys ingested into prompt rollups (documentation + tests).
PROMPT_TAG_KEYS = (
    "building", "building:levels", "building:material", "building:colour", "building:color",
    "roof:shape", "roof:material", "roof:colour", "roof:color",
    "addr:street", "addr:place", "name",
    "highway", "surface", "landuse", "natural", "water", "waterway",
    "leisure", "amenity", "man_made", "sport", "shop", "tourism", "historic",
)


@dataclass
class TagRollup:
    """Aggregated OSM tag values in a tile, used for rich text prompts."""

    building_types: Dict[str, int] = field(default_factory=dict)
    building_levels: List[int] = field(default_factory=list)
    road_names: Dict[str, int] = field(default_factory=dict)
    feature_names: Dict[str, int] = field(default_factory=dict)
    highway_types: Dict[str, int] = field(default_factory=dict)
    landuse: Dict[str, int] = field(default_factory=dict)
    natural: Dict[str, int] = field(default_factory=dict)
    waterway: Dict[str, int] = field(default_factory=dict)
    amenity: Dict[str, int] = field(default_factory=dict)
    leisure: Dict[str, int] = field(default_factory=dict)
    roof_shapes: Dict[str, int] = field(default_factory=dict)
    roof_materials: Dict[str, int] = field(default_factory=dict)
    building_materials: Dict[str, int] = field(default_factory=dict)
    roof_colours: Dict[str, int] = field(default_factory=dict)
    building_colours: Dict[str, int] = field(default_factory=dict)
    addr_streets: Dict[str, int] = field(default_factory=dict)
    addr_places: Dict[str, int] = field(default_factory=dict)
    surfaces: Dict[str, int] = field(default_factory=dict)
    man_made: Dict[str, int] = field(default_factory=dict)
    sport: Dict[str, int] = field(default_factory=dict)
    shop: Dict[str, int] = field(default_factory=dict)
    tourism: Dict[str, int] = field(default_factory=dict)
    historic: Dict[str, int] = field(default_factory=dict)
    water_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class TileFeatures:
    """OSM geometries for a tile, grouped by semantic class (in EPSG:4326)."""

    bounds_lonlat: Bounds
    polygons: Dict[str, list] = field(default_factory=dict)  # class -> [shapely geom]
    roads: Dict[str, list] = field(default_factory=dict)  # highway -> [shapely line]
    counts: Dict[str, int] = field(default_factory=dict)
    tag_rollup: TagRollup = field(default_factory=TagRollup)


# ---------------------------------------------------------------------------
# Tag ingestion (for rich prompts)
# ---------------------------------------------------------------------------
def _row_tags(row) -> dict:
    """Extract OSM tags from a GeoDataFrame row (geometry excluded)."""
    tags = {}
    for key, val in row.items():
        if key == "geometry" or val is None:
            continue
        text = str(val).strip()
        if text and text.lower() != "nan":
            tags[str(key)] = text
    return tags


def _parse_levels(raw: str) -> Optional[int]:
    token = str(raw).split(";")[0].strip()
    if not token:
        return None
    try:
        return int(float(token))
    except ValueError:
        return None


def _norm_tag_val(raw: object) -> str:
    return str(raw).strip().lower()


def _count_tag(store: Dict[str, int], raw: object) -> None:
    val = _norm_tag_val(raw)
    if val and val != "nan":
        store[val] = store.get(val, 0) + 1


def _ingest_tags(feats: TileFeatures, tags: dict, *, is_road: bool = False) -> None:
    """Accumulate tag values from one OSM feature into ``feats.tag_rollup``."""
    if not tags:
        return
    rollup = feats.tag_rollup

    if "building" in tags:
        _count_tag(rollup.building_types, tags["building"])
    levels = _parse_levels(tags.get("building:levels", ""))
    if levels is not None:
        rollup.building_levels.append(levels)

    for key, store in (
        ("building:material", rollup.building_materials),
        ("roof:shape", rollup.roof_shapes),
        ("roof:material", rollup.roof_materials),
        ("highway", rollup.highway_types),
        ("landuse", rollup.landuse),
        ("natural", rollup.natural),
        ("waterway", rollup.waterway),
        ("water", rollup.water_types),
        ("amenity", rollup.amenity),
        ("leisure", rollup.leisure),
        ("man_made", rollup.man_made),
        ("sport", rollup.sport),
        ("shop", rollup.shop),
        ("tourism", rollup.tourism),
        ("historic", rollup.historic),
    ):
        if key in tags:
            _count_tag(store, tags[key])

    for colour_key, store in (
        ("roof:colour", rollup.roof_colours),
        ("roof:color", rollup.roof_colours),
        ("building:colour", rollup.building_colours),
        ("building:color", rollup.building_colours),
    ):
        if colour_key in tags:
            _count_tag(store, tags[colour_key])

    if "surface" in tags:
        _count_tag(rollup.surfaces, tags["surface"])

    name = str(tags.get("name", "")).strip()
    if name:
        if is_road:
            rollup.road_names[name] = rollup.road_names.get(name, 0) + 1
        else:
            rollup.feature_names[name] = rollup.feature_names.get(name, 0) + 1

    if not is_road:
        street = str(tags.get("addr:street", "")).strip()
        if street:
            rollup.addr_streets[street] = rollup.addr_streets.get(street, 0) + 1
        place = str(tags.get("addr:place", "")).strip()
        if place:
            rollup.addr_places[place] = rollup.addr_places.get(place, 0) + 1


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
                _ingest_tags(feats, _row_tags(row))

        roads = clip(self._roads)
        if roads is not None:
            for _, row in roads.iterrows():
                hw = row.get("highway")
                if not hw or row.geometry is None:
                    continue
                feats.roads.setdefault(str(hw), []).append(row.geometry)
                feats.counts["road"] = feats.counts.get("road", 0) + 1
                _ingest_tags(feats, _row_tags(row), is_road=True)
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
          way["leisure"]({bbox});
          way["sport"]({bbox});
          way["man_made"]({bbox});
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
                _ingest_tags(feats, tags, is_road=True)
                continue
            # Tag-only or area features (man_made lines, sport pitches, parks).
            if any(k in tags for k in ("man_made", "sport", "leisure", "historic")) and "building" not in tags:
                _ingest_tags(feats, tags)
                if len(coords) < 3:
                    continue
            if len(coords) < 3:
                continue
            poly = Polygon(coords)
            cls = _classify_tags(tags)
            if cls is None:
                continue
            feats.polygons.setdefault(cls, []).append(poly)
            feats.counts[cls] = feats.counts.get(cls, 0) + 1
            _ingest_tags(feats, tags)
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
    if tags.get("water") in WATER_TYPE_LABELS:
        return "water"
    lu = tags.get("landuse")
    if lu in {"forest", "grass", "meadow", "farmland", "orchard", "vineyard"}:
        return "vegetation"
    if lu == "residential":
        return "residential"
    if lu in {"commercial", "retail", "industrial"}:
        return "commercial"
    leisure = tags.get("leisure")
    if leisure in {"park", "garden", "playground", "pitch", "golf_course", "stadium", "track", "nature_reserve"}:
        return "vegetation"
    if tags.get("sport"):
        return "vegetation"
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


def render_building_edge_control_image(
    tile: Tile,
    features: TileFeatures,
    size: int = 1024,
    line_width_px: float = 2.0,
) -> "np.ndarray":
    """Rasterise building footprint outlines only (white on black).

    Intended for the OSM ControlNet path: Canny on the full palette image mixes
    roads, landcover and buildings; building outlines alone give cleaner spatial
    guidance for rooflines and block edges.
    """
    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from shapely.ops import transform as shp_transform
        import pyproj
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "render_building_edge_control_image needs rasterio, shapely and pyproj "
            "(pip install rasterio shapely pyproj)"
        ) from exc

    buildings = features.polygons.get("building", [])
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if not buildings:
        return canvas

    west, south, east, north = tile_bounds_3857(tile)
    transform = from_bounds(west, south, east, north, size, size)
    meters_per_px = (east - west) / size
    buffer_m = max(line_width_px * meters_per_px / 2.0, meters_per_px * 0.5)

    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    shapes = []
    for geom in buildings:
        gm = shp_transform(project, geom)
        outline = gm.boundary if hasattr(gm, "boundary") else gm
        if buffer_m > 0:
            outline = outline.buffer(buffer_m, cap_style=2, join_style=2)
        shapes.append((outline, 1))

    mask = rasterize(
        shapes,
        out_shape=(size, size),
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype="uint8",
    )
    canvas[mask.astype(bool)] = (255, 255, 255)
    return canvas


# SDXL conditions on OpenAI CLIP ViT-L/14 text (77 tokens including specials).
CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"
CLIP_MAX_TOKENS = 77

_CLIP_TOKENIZER = None


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------
def _clip_tokenizer():
    """Lazy-load the CLIP tokenizer used by SDXL (cached after first call)."""
    global _CLIP_TOKENIZER
    if _CLIP_TOKENIZER is None:
        from transformers import CLIPTokenizer

        _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_ID)
    return _CLIP_TOKENIZER


def _clip_token_count(text: str) -> int:
    """Return CLIP token length for ``text`` (including special tokens)."""
    try:
        tok = _clip_tokenizer()
        return len(tok(text, add_special_tokens=True, truncation=False).input_ids)
    except Exception:
        # Offline / no cache: conservative estimate so prompts stay under the limit.
        words = text.replace(",", " ,").split()
        return int(len(words) * 1.35) + 2


def _top_labels(counter: Dict[str, int], label_map: Dict[str, str], limit: int = 3) -> List[str]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    out: List[str] = []
    for key, _count in ranked[:limit]:
        label = label_map.get(key, key.replace("_", " "))
        if label not in out:
            out.append(label)
    return out


def _dedupe_phrases(phrases: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _names_overlap(a: str, b: str) -> bool:
    al, bl = a.lower(), b.lower()
    return al == bl or al in bl or bl in al


def _street_phrases(rollup: TagRollup) -> List[str]:
    """Named roads and addr:street values, deduped."""
    phrases: List[str] = []
    used: List[str] = []

    ranked_roads = sorted(rollup.road_names.items(), key=lambda item: (-item[1], item[0]))
    for name, _ in ranked_roads[:2]:
        phrases.append(f"along {name}")
        used.append(name)

    ranked_addr = sorted(rollup.addr_streets.items(), key=lambda item: (-item[1], item[0]))
    for street, _ in ranked_addr[:2]:
        if any(_names_overlap(street, known) for known in used):
            continue
        phrases.append(f"on {street}")
        used.append(street)
        if len(phrases) >= 3:
            break

    return phrases[:3]


def _finalize_prompt(
    base: str,
    phrases: List[str],
    max_tokens: int = CLIP_MAX_TOKENS,
) -> str:
    """Join ``phrases`` into a CLIP-safe SDXL prompt (<= ``max_tokens`` tokens)."""
    phrases = _dedupe_phrases(phrases)
    suffix = ", sharp detail, realistic textures"
    kept = list(phrases)
    with_suffix = True

    while True:
        body = ", ".join(kept)
        if body:
            prompt = f"{base}, {body}"
        else:
            prompt = base
        if with_suffix:
            prompt = f"{prompt}{suffix}"

        if _clip_token_count(prompt) <= max_tokens:
            return prompt

        if with_suffix:
            with_suffix = False
            continue
        if kept:
            kept.pop()
            continue
        return base


def build_prompt(features: TileFeatures, base: str = "high-resolution aerial orthophoto, top-down satellite view") -> str:
    """Summarise the tile's OSM content into a text prompt for the diffuser."""
    counts = features.counts
    rollup = features.tag_rollup
    phrases: List[str] = []

    n_buildings = counts.get("building", 0)
    if n_buildings >= 25:
        phrases.append("dense buildings and rooftops")
    elif n_buildings >= 5:
        phrases.append("scattered buildings with rooftops")
    elif n_buildings >= 1:
        phrases.append("a few isolated buildings")

    for label in _top_labels(rollup.building_types, BUILDING_LABELS, limit=3):
        if label.endswith("buildings") and any("buildings" in p for p in phrases):
            continue
        phrases.append(label)

    for label in _top_labels(rollup.roof_shapes, ROOF_SHAPE_LABELS, limit=2):
        phrases.append(label)
    for label in _top_labels(rollup.roof_materials, ROOF_MATERIAL_LABELS, limit=2):
        phrases.append(label)
    for label in _top_labels(rollup.building_materials, BUILDING_MATERIAL_LABELS, limit=2):
        phrases.append(label)
    for label in _top_labels(rollup.roof_colours, COLOUR_LABELS, limit=1):
        phrases.append(label)
    for label in _top_labels(rollup.building_colours, COLOUR_LABELS, limit=1):
        if label not in phrases:
            phrases.append(label.replace("rooftops", "building facades"))

    if rollup.building_levels:
        max_levels = max(rollup.building_levels)
        if max_levels >= 6:
            phrases.append(f"high-rise buildings up to {max_levels} floors")
        elif max_levels >= 4:
            phrases.append(f"multi-storey buildings up to {max_levels} floors")
        elif max_levels >= 2:
            phrases.append("low-rise multi-storey buildings")

    for label in _top_labels(rollup.highway_types, HIGHWAY_LABELS, limit=2):
        phrases.append(label)
    phrases.extend(_street_phrases(rollup))
    for label in _top_labels(rollup.surfaces, SURFACE_LABELS, limit=2):
        phrases.append(label)

    if counts.get("residential") or rollup.landuse.get("residential"):
        phrases.append("residential housing")
    if counts.get("commercial") or rollup.landuse.get("commercial"):
        phrases.append("commercial or industrial structures")

    for label in _top_labels(rollup.landuse, LANDUSE_LABELS, limit=2):
        phrases.append(label)
    for label in _top_labels(rollup.natural, NATURAL_LABELS, limit=2):
        phrases.append(label)
    for label in _top_labels(rollup.water_types, WATER_TYPE_LABELS, limit=2):
        phrases.append(label)
    for label in _top_labels(rollup.leisure, LEISURE_LABELS, limit=1):
        phrases.append(label)
    for label in _top_labels(rollup.sport, SPORT_LABELS, limit=1):
        phrases.append(label)
    for label in _top_labels(rollup.amenity, AMENITY_LABELS, limit=1):
        phrases.append(label)
    for label in _top_labels(rollup.man_made, MAN_MADE_LABELS, limit=2):
        phrases.append(label)
    for label in _top_labels(rollup.shop, SHOP_LABELS, limit=1):
        phrases.append(label)
    for label in _top_labels(rollup.tourism, TOURISM_LABELS, limit=1):
        phrases.append(label)
    for label in _top_labels(rollup.historic, HISTORIC_LABELS, limit=1):
        phrases.append(label)

    if counts.get("road") and not rollup.highway_types:
        phrases.append("roads and pavement")
    if counts.get("vegetation"):
        phrases.append("trees, grass and vegetation")
    if counts.get("water") or rollup.waterway:
        phrases.append("water")

    if rollup.addr_places:
        top_places = sorted(rollup.addr_places.items(), key=lambda item: (-item[1], item[0]))[:1]
        phrases.append(f"in {top_places[0][0]}")

    if rollup.feature_names:
        top_names = sorted(rollup.feature_names.items(), key=lambda item: (-item[1], item[0]))[:2]
        named = [name for name, _ in top_names]
        if len(named) == 1:
            phrases.append(f"including {named[0]}")
        else:
            phrases.append(f"including {named[0]} and {named[1]}")

    if not phrases:
        phrases.append("open natural terrain")

    return _finalize_prompt(base, phrases)


def render_tile(
    source: OSMSource,
    tile: Tile,
    size: int = 1024,
    edge_line_width_px: float = 2.0,
) -> Tuple["np.ndarray", "np.ndarray", str]:
    """Fetch features and return (palette_control, building_edges, prompt)."""
    feats = source.features_for_bounds(tile_bounds_lonlat(tile))
    control = render_control_image(tile, feats, size=size)
    edges = render_building_edge_control_image(
        tile, feats, size=size, line_width_px=edge_line_width_px,
    )
    prompt = build_prompt(feats)
    return control, edges, prompt


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
    edges_root = os.path.join(args.out, "edges")
    prompts = {}
    for key in keys:
        z, x, y = (int(v) for v in key.split("/"))
        tile = Tile(z, x, y)
        control, edges, prompt = render_tile(source, tile, size=args.size)
        out_dir = os.path.join(args.out, str(z), str(x))
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(control).save(os.path.join(out_dir, f"{y}.png"))
        edge_dir = os.path.join(edges_root, str(z), str(x))
        os.makedirs(edge_dir, exist_ok=True)
        Image.fromarray(edges).save(os.path.join(edge_dir, f"{y}.png"))
        prompts[key] = prompt
        print(f"{key}: {prompt}")

    with open(os.path.join(args.out, "prompts.json"), "w") as fh:
        json.dump(prompts, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
