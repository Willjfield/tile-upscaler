"""Evaluation: degradation test, cross-scale consistency, no-reference metrics.

This module answers two questions:

1. **Does vector guidance help?** Via a self-supervised *degradation test*: take
   genuine high-zoom tiles (HR), bicubic-downsample them to simulate low-res
   (LR), upscale with each method, and score the reconstruction against the real
   HR with PSNR / SSIM / LPIPS. Because we have true HR, this is a fair,
   quantitative comparison of methods.

2. **Is the result seamless when zooming in?** Via *cross-scale consistency*
   checks on any upscaled tile (no HR needed):
     - color/brightness drift: downsample the SR back to source size and compare
       to the LR source (mean abs error + SSIM). Should be ~0 after color-fix.
     - contour adherence: correlation between the low-passed SR and the bicubic
       LR (are large-scale contours preserved?).

Plus optional no-reference perceptual metrics (CLIP-IQA / NIQE via ``pyiqa``)
and side-by-side comparison sheets for eyeballing.

The upscaling itself is done by the upscaler modules; this module operates on
directories of images and is agnostic to how they were produced. Typical flow:

    eval.py make-degraded --hr HR/ --lr LR/ --factor 4      # build the LR set
    # ...run baseline / controlnet on LR/ -> methodA/, methodB/ ...
    eval.py score --hr HR/ --sr methodA/ --csv methodA.csv
    eval.py consistency --lr LR/ --sr methodA/ --csv methodA_consistency.csv
    eval.py sheet --lr LR/ --variants methodA=methodA/ methodB=methodB/ --out sheets/
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import tileio


# ---------------------------------------------------------------------------
# Reference metrics (need ground-truth HR)
# ---------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(
            structural_similarity(a, b, channel_axis=2, data_range=255)
        )
    except ImportError:
        return float("nan")


class _LPIPS:
    """Lazy LPIPS wrapper (optional dependency)."""

    _model = None
    _torch = None

    @classmethod
    def score(cls, a: np.ndarray, b: np.ndarray) -> float:
        try:
            import lpips
            import torch
        except ImportError:
            return float("nan")
        if cls._model is None:
            cls._torch = torch
            cls._model = lpips.LPIPS(net="alex")
            cls._model.eval()

        def to_t(x):
            t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            return t * 2 - 1  # LPIPS expects [-1, 1]

        with cls._torch.no_grad():
            return float(cls._model(to_t(a), to_t(b)).item())


@dataclass
class ScoreRow:
    tile: str
    psnr: float
    ssim: float
    lpips: float


def make_degraded(hr_root: str, lr_root: str, factor: int = 4) -> int:
    """Build an LR tile tree by bicubic-downsampling each HR tile by ``factor``."""
    from PIL import Image

    tiles = tileio.find_tiles(hr_root)
    for tf in tiles:
        img = tileio.load_image(tf.path)
        small = img.resize((img.width // factor, img.height // factor), Image.BICUBIC)
        tileio.write_tile(lr_root, tf.tile, small)
    print(f"Wrote {len(tiles)} degraded tiles -> {lr_root} (factor {factor})")
    return len(tiles)


def score_against_hr(hr_root: str, sr_root: str, csv_path: Optional[str] = None) -> Dict[str, float]:
    """Score every SR tile against the matching HR tile. Returns averages."""
    from PIL import Image

    hr_tiles = {tf.tile.key: tf for tf in tileio.find_tiles(hr_root)}
    rows: List[ScoreRow] = []
    for sr in tileio.find_tiles(sr_root):
        hr = hr_tiles.get(sr.tile.key)
        if hr is None:
            continue
        hr_img = tileio.load_image(hr.path)
        sr_img = tileio.load_image(sr.path)
        if sr_img.size != hr_img.size:
            sr_img = sr_img.resize(hr_img.size, Image.BICUBIC)
        a = np.asarray(hr_img)
        b = np.asarray(sr_img)
        rows.append(
            ScoreRow(sr.tile.key, psnr(a, b), ssim(a, b), _LPIPS.score(a, b))
        )

    summary = _summarise(rows, ["psnr", "ssim", "lpips"])
    if csv_path:
        _write_csv(csv_path, rows)
    _print_summary(f"Reference metrics ({sr_root} vs {hr_root})", summary, len(rows))
    return summary


# ---------------------------------------------------------------------------
# Cross-scale consistency (no HR needed)
# ---------------------------------------------------------------------------
@dataclass
class ConsistencyRow:
    tile: str
    color_mae: float        # mean abs error of SR-downsampled vs LR (lower=better)
    color_ssim: float       # structural similarity of color at LR scale (higher=better)
    contour_corr: float     # correlation of large-scale contours (higher=better)


def _downsample_to(img, size: Tuple[int, int]):
    from PIL import Image

    return img.resize(size, Image.BICUBIC)


def _lowpass(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        gray = arr.mean(axis=2)
        return gaussian_filter(gray, sigma=sigma, mode="reflect")
    except ImportError:
        return arr.mean(axis=2)


def consistency(lr_root: str, sr_root: str, csv_path: Optional[str] = None) -> Dict[str, float]:
    """Measure cross-scale consistency of each SR tile vs its LR source."""
    lr_tiles = {tf.tile.key: tf for tf in tileio.find_tiles(lr_root)}
    rows: List[ConsistencyRow] = []
    for sr in tileio.find_tiles(sr_root):
        lr = lr_tiles.get(sr.tile.key)
        if lr is None:
            continue
        lr_img = tileio.load_image(lr.path)
        sr_img = tileio.load_image(sr.path)

        # Color/brightness drift: SR -> LR scale should match the LR source.
        sr_small = _downsample_to(sr_img, lr_img.size)
        a = np.asarray(lr_img).astype(np.float32)
        b = np.asarray(sr_small).astype(np.float32)
        color_mae = float(np.mean(np.abs(a - b)))
        color_ssim = ssim(a.astype(np.uint8), b.astype(np.uint8))

        # Contour adherence: low-passed SR vs bicubic-upscaled LR.
        lr_up = np.asarray(_downsample_to(lr_img, sr_img.size)).astype(np.float32)
        contour_corr = _corr(_lowpass(np.asarray(sr_img).astype(np.float32)), _lowpass(lr_up))
        rows.append(ConsistencyRow(sr.tile.key, color_mae, color_ssim, contour_corr))

    summary = _summarise(rows, ["color_mae", "color_ssim", "contour_corr"])
    if csv_path:
        _write_csv(csv_path, rows)
    _print_summary(f"Cross-scale consistency ({sr_root} vs {lr_root})", summary, len(rows))
    return summary


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


# ---------------------------------------------------------------------------
# No-reference perceptual metrics (optional, via pyiqa)
# ---------------------------------------------------------------------------
def _load_nr_metrics(metrics: Tuple[str, ...], device: str) -> Dict[str, object]:
    """Load pyiqa metrics individually so one broken backend does not abort eval."""
    import pyiqa

    models: Dict[str, object] = {}
    for name in metrics:
        try:
            models[name] = pyiqa.create_metric(name, device=device)
        except Exception as exc:
            print(f"  [warn] no-reference metric {name!r} unavailable ({exc}); skipping")
    return models


def no_reference(sr_root: str, metrics: Tuple[str, ...] = ("clipiqa", "niqe"), csv_path: Optional[str] = None) -> Dict[str, float]:
    try:
        import torch
    except ImportError:
        print("torch not installed; skipping no-reference metrics")
        return {}

    try:
        import pyiqa  # noqa: F401
    except ImportError:
        print("pyiqa not installed; skipping no-reference metrics "
              "(pip install pyiqa)")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = _load_nr_metrics(metrics, device)
    if not models:
        print("No no-reference metrics could be loaded; skipping")
        return {}
    from PIL import Image
    import torchvision.transforms.functional as TF

    rows: List[dict] = []
    for sr in tileio.find_tiles(sr_root):
        img = tileio.load_image(sr.path)
        t = TF.to_tensor(img)[None].to(device)
        row = {"tile": sr.tile.key}
        for name, model in models.items():
            row[name] = float(model(t).item())
        rows.append(row)

    summary = _summarise_dicts(rows, list(metrics))
    if csv_path:
        _write_csv_dicts(csv_path, rows, ["tile", *metrics])
    _print_summary(f"No-reference metrics ({sr_root})", summary, len(rows))
    return summary


# ---------------------------------------------------------------------------
# Comparison sheets
# ---------------------------------------------------------------------------
def comparison_sheets(
    lr_root: str,
    variants: Dict[str, str],
    out_dir: str,
    hr_root: Optional[str] = None,
    limit: Optional[int] = None,
    tile_keys: Optional[Sequence[str]] = None,
) -> int:
    """Write side-by-side panels: [LR(bicubic)] [variant...] [HR?] per tile."""
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    variant_index = {name: {tf.tile.key: tf for tf in tileio.find_tiles(root)} for name, root in variants.items()}
    hr_index = {tf.tile.key: tf for tf in tileio.find_tiles(hr_root)} if hr_root else {}

    lr_tiles = tileio.select_tiles(tileio.find_tiles(lr_root), limit=limit, tile_keys=tile_keys)

    panel = 512
    made = 0
    for lr in lr_tiles:
        cells: List[Tuple[str, "Image.Image"]] = []
        bic = tileio.load_image(lr.path).resize((panel, panel), Image.BICUBIC)
        cells.append(("bicubic", bic))
        for name in variants:
            tf = variant_index[name].get(lr.tile.key)
            if tf is None:
                continue
            cells.append((name, tileio.load_image(tf.path).resize((panel, panel), Image.BICUBIC)))
        if hr_index:
            tf = hr_index.get(lr.tile.key)
            if tf is not None:
                cells.append(("HR (truth)", tileio.load_image(tf.path).resize((panel, panel), Image.BICUBIC)))

        label_h = 24
        sheet = Image.new("RGB", (panel * len(cells), panel + label_h), "black")
        draw = ImageDraw.Draw(sheet)
        for i, (label, img) in enumerate(cells):
            sheet.paste(img, (i * panel, label_h))
            draw.text((i * panel + 6, 6), f"{label}  [{lr.tile.key}]", fill="white")
        sheet.save(os.path.join(out_dir, f"{lr.tile.z}_{lr.tile.x}_{lr.tile.y}.png"))
        made += 1
    print(f"Wrote {made} comparison sheets -> {out_dir}")
    return made


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _summarise(rows, fields) -> Dict[str, float]:
    out = {}
    for f in fields:
        vals = [getattr(r, f) for r in rows]
        vals = [v for v in vals if v == v and v not in (float("inf"), float("-inf"))]
        out[f] = float(np.mean(vals)) if vals else float("nan")
    return out


def _summarise_dicts(rows, fields) -> Dict[str, float]:
    out = {}
    for f in fields:
        vals = [r[f] for r in rows if f in r and r[f] == r[f]]
        out[f] = float(np.mean(vals)) if vals else float("nan")
    return out


def _write_csv(path, rows) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(asdict(rows[0]).keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def _write_csv_dicts(path, rows, fields) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(title: str, summary: Dict[str, float], n: int) -> None:
    print(f"\n== {title} (n={n}) ==")
    for k, v in summary.items():
        print(f"  {k:14s}: {v:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tile upscaling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-degraded", help="Downsample HR tiles to make an LR set")
    p.add_argument("--hr", required=True)
    p.add_argument("--lr", required=True)
    p.add_argument("--factor", type=int, default=4)

    p = sub.add_parser("score", help="Reference metrics (PSNR/SSIM/LPIPS) vs HR")
    p.add_argument("--hr", required=True)
    p.add_argument("--sr", required=True)
    p.add_argument("--csv")

    p = sub.add_parser("consistency", help="Cross-scale color/contour consistency vs LR")
    p.add_argument("--lr", required=True)
    p.add_argument("--sr", required=True)
    p.add_argument("--csv")

    p = sub.add_parser("no-reference", help="No-reference perceptual metrics (pyiqa)")
    p.add_argument("--sr", required=True)
    p.add_argument("--metrics", nargs="+", default=["clipiqa", "niqe"])
    p.add_argument("--csv")

    p = sub.add_parser("sheet", help="Side-by-side comparison sheets")
    p.add_argument("--lr", required=True)
    p.add_argument("--variants", nargs="+", required=True, help="name=dir ...")
    p.add_argument("--hr")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)

    args = parser.parse_args(argv)

    if args.cmd == "make-degraded":
        make_degraded(args.hr, args.lr, args.factor)
    elif args.cmd == "score":
        score_against_hr(args.hr, args.sr, args.csv)
    elif args.cmd == "consistency":
        consistency(args.lr, args.sr, args.csv)
    elif args.cmd == "no-reference":
        no_reference(args.sr, tuple(args.metrics), args.csv)
    elif args.cmd == "sheet":
        variants = dict(v.split("=", 1) for v in args.variants)
        comparison_sheets(args.lr, variants, args.out, hr_root=args.hr, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
