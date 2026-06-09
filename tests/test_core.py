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
from tile_upscaler import upscale_controlnet as uc  # noqa: E402
from tile_upscaler.tiles import (  # noqa: E402
    Tile, child_tiles, lonlat_to_tile, neighbours, tile_bounds_3857,
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


if __name__ == "__main__":
    test_tile_math()
    test_colorfix_reduces_drift()
    test_retile_grid()
    test_blend_helpers()
    test_consistency_metric()
    test_degradation_roundtrip()
    print("\nALL CORE TESTS PASSED")
