"""Lightweight smoke tests for the CPU-only logic (no torch/GPU needed).

Run with: python tests/test_core.py
Covers tile math, color-fix, retiling, tiled-blend helpers, and consistency
metrics on synthetic data so logic regressions are caught without model weights.
"""

import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tile_upscaler import colorfix, eval as ev, retile, tileio  # noqa: E402
from tile_upscaler import tile_rank, upscale_controlnet as uc  # noqa: E402
from tile_upscaler.tiles import (  # noqa: E402
    Tile, ancestor_tile, child_tiles, lonlat_to_tile, neighbours,
    pixel_factor_for_zoom_delta, plan_iterative_zoom_ladder, tile_bounds_3857,
    tile_bounds_lonlat, upscale_levels,
)


def _rand_image(size, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _smooth_image(size, seed=0):
    """A low-frequency (band-limited) image that survives bicubic round-trips,
    appropriate for cross-scale consistency assertions."""
    yy, xx = np.mgrid[0:size, 0:size] / max(size - 1, 1)
    r = (np.sin(2 * np.pi * xx + seed) * 0.5 + 0.5)
    g = (np.cos(2 * np.pi * yy + seed) * 0.5 + 0.5)
    b = ((xx + yy) / 2.0)
    arr = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_tile_math():
    t = lonlat_to_tile(-122.4194, 37.7849, 16)
    assert t.z == 16
    w, s, e, n = tile_bounds_lonlat(t)
    assert w < e and s < n
    mw, ms, me, mn = tile_bounds_3857(t)
    assert mw < me and ms < mn
    assert len(child_tiles(t, 2)) == 16
    assert child_tiles(t, 2)[0].z == 18
    assert len(neighbours(t, 1)) == 9
    assert upscale_levels(4) == 2 and upscale_levels(2) == 1
    print("ok test_tile_math")


def test_iterative_zoom_ladder():
    assert plan_iterative_zoom_ladder(18, 25, 4) == [18, 22, 25]
    assert plan_iterative_zoom_ladder(18, 22, 4) == [18, 22]
    assert plan_iterative_zoom_ladder(18, 26, 4) == [18, 22, 26]
    assert pixel_factor_for_zoom_delta(4) == 16
    assert pixel_factor_for_zoom_delta(3) == 8
    t = Tile(22, 100, 200)
    assert ancestor_tile(t, 18).key == Tile(18, 6, 12).key
    print("ok test_iterative_zoom_ladder")


def test_colorfix_reduces_drift():
    # reference is darker; target is brighter + noisy. After color-fix the
    # low-frequency color should move toward the reference.
    ref = Image.fromarray(np.full((128, 128, 3), 80, dtype=np.uint8))
    tgt = _rand_image(128, seed=1)
    tgt = Image.fromarray(np.clip(np.asarray(tgt).astype(int) + 120, 0, 255).astype(np.uint8))

    before = abs(np.asarray(tgt).mean() - 80)
    for method in ("wavelet", "adain"):
        out = colorfix.apply_color_fix(tgt, ref, method)
        assert out.size == tgt.size
        after = abs(np.asarray(out).mean() - 80)
        assert after < before, f"{method}: drift not reduced ({after} !< {before})"
    print("ok test_colorfix_reduces_drift")


def test_retile_grid():
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "src"), os.path.join(d, "out")
        tile = Tile(16, 100, 200)
        tileio.write_tile(src, tile, _rand_image(1024, seed=2))
        n = retile.run_tree(src, out, factor=4, tile_size=256)
        assert n == 16
        children = tileio.find_tiles(out)
        assert len(children) == 16
        assert all(c.tile.z == 18 for c in children)
        xs = {c.tile.x for c in children}
        assert xs == set(range(400, 404))  # 100*4 .. +3
    print("ok test_retile_grid")


def test_blend_helpers():
    assert uc._window_starts(256, 256, 128) == [0]
    starts = uc._window_starts(1280, 1024, 896)
    assert starts[0] == 0 and starts[-1] == 1280 - 1024
    h = uc._hann2d(64)
    assert h.shape == (64, 64) and h.max() <= 1.0 and h.min() > 0
    print("ok test_blend_helpers")


def test_consistency_metric():
    with tempfile.TemporaryDirectory() as d:
        lr, sr = os.path.join(d, "lr"), os.path.join(d, "sr")
        tile = Tile(16, 1, 2)
        base = _smooth_image(64, seed=3)
        tileio.write_tile(lr, tile, base)
        # SR = a faithful 4x bicubic of the LR -> should be highly consistent.
        tileio.write_tile(sr, tile, base.resize((256, 256), Image.BICUBIC))
        summary = ev.consistency(lr, sr)
        assert summary["color_mae"] < 5.0, summary
        assert summary["contour_corr"] > 0.9, summary
    print("ok test_consistency_metric")


def test_degradation_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        hr, lr = os.path.join(d, "hr"), os.path.join(d, "lr")
        tileio.write_tile(hr, Tile(18, 5, 6), _rand_image(256, seed=4))
        ev.make_degraded(hr, lr, factor=4)
        lr_tiles = tileio.find_tiles(lr)
        assert len(lr_tiles) == 1
        assert tileio.load_image(lr_tiles[0].path).size == (64, 64)
    print("ok test_degradation_roundtrip")


def test_build_prompt_rich_tags():
    from tile_upscaler import osm_render as orm

    feats = orm.TileFeatures(bounds_lonlat=(-4.28, 55.86, -4.27, 55.87))
    feats.counts = {"building": 12, "road": 3, "vegetation": 5}
    feats.tag_rollup.building_types = {"terrace": 8, "apartments": 4}
    feats.tag_rollup.building_levels = [2, 3, 4, 4]
    feats.tag_rollup.roof_shapes = {"gabled": 6, "flat": 2}
    feats.tag_rollup.roof_materials = {"slate": 7}
    feats.tag_rollup.building_materials = {"brick": 9}
    feats.tag_rollup.roof_colours = {"grey": 5}
    feats.tag_rollup.road_names = {"Sauchiehall Street": 2, "Renfrew Street": 1}
    feats.tag_rollup.addr_streets = {"Sauchiehall Street": 4, "Rose Street": 2}
    feats.tag_rollup.highway_types = {"residential": 2, "secondary": 1}
    feats.tag_rollup.surfaces = {"asphalt": 4, "paving_stones": 1}
    feats.tag_rollup.landuse = {"grass": 3}
    feats.tag_rollup.sport = {"tennis": 1}
    feats.tag_rollup.historic = {"building": 1}
    prompt = orm.build_prompt(feats)
    assert "terraced houses" in prompt
    assert "gabled rooftops" in prompt
    assert "slate rooftops" in prompt
    assert "brick buildings" in prompt
    assert "grey rooftops" in prompt
    assert "along Sauchiehall Street" in prompt
    assert "multi-storey" in prompt
    assert orm._clip_token_count(prompt) <= orm.CLIP_MAX_TOKENS
    print("ok test_build_prompt_rich_tags")


def test_finalize_prompt_clip_limit():
    from tile_upscaler import osm_render as orm

    base = "high-resolution aerial orthophoto, top-down satellite view"
    long_phrases = [
        "scattered buildings with rooftops",
        "apartment blocks",
        "low-rise multi-storey buildings",
        "residential street",
        "service road",
        "along Derby Street",
        "along Craigmaddie Terrace Lane",
        "on Parkgrove Terrace",
        "on Bentinck Street",
        "asphalt roads",
        "cobblestone",
        "residential housing",
        "grassy areas",
        "trees, grass and vegetation",
        "including Dom Polski and Kelvingrove House apartment",
        "tennis courts",
        "historic building",
        "place of worship",
    ]
    prompt = orm._finalize_prompt(base, long_phrases)
    assert orm._clip_token_count(prompt) <= orm.CLIP_MAX_TOKENS
    assert prompt.startswith(base)
    print("ok test_finalize_prompt_clip_limit")


def test_ingest_tags():
    from tile_upscaler import osm_render as orm

    feats = orm.TileFeatures(bounds_lonlat=(0, 0, 1, 1))
    orm._ingest_tags(
        feats,
        {
            "building": "terrace",
            "building:levels": "3",
            "building:material": "brick",
            "roof:shape": "gabled",
            "roof:material": "slate",
            "roof:colour": "grey",
            "addr:street": "High Street",
            "addr:place": "Old Town",
            "shop": "convenience",
            "tourism": "hotel",
        },
    )
    orm._ingest_tags(
        feats,
        {"highway": "residential", "name": "High Street", "surface": "asphalt"},
        is_road=True,
    )
    rollup = feats.tag_rollup
    assert rollup.building_types["terrace"] == 1
    assert rollup.building_levels == [3]
    assert rollup.roof_shapes["gabled"] == 1
    assert rollup.addr_streets["High Street"] == 1
    assert rollup.addr_places["Old Town"] == 1
    assert rollup.road_names["High Street"] == 1
    assert rollup.surfaces["asphalt"] == 1
    assert rollup.shop["convenience"] == 1
    print("ok test_ingest_tags")


def test_building_edge_control_image():
    from shapely.geometry import box

    from tile_upscaler import osm_render as orm

    tile = Tile(18, 127955, 81803)
    west, south, east, north = tile_bounds_lonlat(tile)
    cx, cy = (west + east) / 2, (south + north) / 2
    delta = (east - west) * 0.05
    poly = box(cx - delta, cy - delta, cx + delta, cy + delta)
    feats = orm.TileFeatures(bounds_lonlat=(west, south, east, north))
    feats.polygons["building"] = [poly]
    edges = orm.render_building_edge_control_image(tile, feats, size=256, line_width_px=2)
    assert edges.shape == (256, 256, 3)
    assert int(edges.max()) > 0
    white = edges[:, :, 0]
    assert white.sum() == edges[:, :, 1].sum() == edges[:, :, 2].sum()
    print("ok test_building_edge_control_image")


def test_select_tiles_by_keys():
    tiles = [
        tileio.TileFile(Tile(18, 1, 1), "a"),
        tileio.TileFile(Tile(18, 1, 2), "b"),
        tileio.TileFile(Tile(18, 1, 3), "c"),
    ]
    picked = tileio.select_tiles(tiles, tile_keys=["18/1/3", "18/1/1"])
    assert [tf.tile.key for tf in picked] == ["18/1/3", "18/1/1"]
    print("ok test_select_tiles_by_keys")


def test_tile_rank_orders_by_divergence():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out")
        up = os.path.join(out, "up")
        tile = Tile(18, 9, 9)
        for sub, color in [
            ("A_realesrgan", (40, 40, 40)),
            ("B_controlnet_text", (200, 40, 40)),
            ("C_controlnet_osm", (40, 200, 40)),
        ]:
            root = os.path.join(up, sub)
            tileio.write_tile(root, tile, Image.new("RGB", (64, 64), color))

        quiet = Tile(18, 8, 8)
        for sub in ("A_realesrgan", "B_controlnet_text", "C_controlnet_osm"):
            tileio.write_tile(os.path.join(up, sub), quiet, Image.new("RGB", (64, 64), (100, 100, 100)))

        rows = tile_rank.rank_tiles(out, compare_size=64)
        assert rows[0].tile == "18/9/9"
        assert rows[0].mean_pairwise_mae > rows[-1].mean_pairwise_mae

        dest = tile_rank.write_rankings(out, rows)
        keys = tile_rank.best_tile_keys(dest, 1)
        assert keys == ["18/9/9"]
    print("ok test_tile_rank_orders_by_divergence")


if __name__ == "__main__":
    test_tile_math()
    test_iterative_zoom_ladder()
    test_colorfix_reduces_drift()
    test_retile_grid()
    test_blend_helpers()
    test_consistency_metric()
    test_degradation_roundtrip()
    test_build_prompt_rich_tags()
    test_finalize_prompt_clip_limit()
    test_ingest_tags()
    test_building_edge_control_image()
    test_select_tiles_by_keys()
    test_tile_rank_orders_by_divergence()
    print("\nALL CORE TESTS PASSED")
