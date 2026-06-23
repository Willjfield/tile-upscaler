# Vector-Guided Aerial Tile Upscaling

Experimental pipeline for AI-upscaling aerial **XYZ raster tiles** for a slippy
map, optionally **guided by OpenStreetMap vector data**. It compares a no-vector
baseline (Real-ESRGAN / Swin2SR) against OSM-guided diffusion upscaling (SDXL +
ControlNet) and measures, quantitatively, whether the vector guidance helps.

The design goal is **seamless zoom**: hallucinated high-frequency *texture* is
fine, but the upscaled tile must read as a faithful refinement of the lower-zoom
view - same brightness/shade and the same large-scale contours.

## How it works

```
source raster tile ─┬─────────────► baseline upscaler (Real-ESRGAN)        ─┐
   (z/x/y)          │                                                        │
                    └─► OSM vector ──► control image + text prompt ─►        ├─► evaluate ─► retile ─► serve
                                       SDXL + ControlNet (Tile [+ OSM]) ─────┘            (z+2 tiles)
                                       + color-fix (cross-scale consistency)
```

Methods compared:

| Id | Method | Vector guidance |
|----|--------|-----------------|
| A  | Real-ESRGAN (or Swin2SR) | none (baseline) |
| B  | SDXL + Tile ControlNet + OSM **text prompt** | semantic (text) |
| C  | B + OSM-edge **ControlNet** | spatial + semantic |
| (stretch) | [SGDM](https://github.com/wwangcece/SGDM) | purpose-built remote-sensing vector SR |

### Cross-scale consistency (core requirement)

Enforced three ways so zooming in feels seamless:
1. **Low denoise strength** (`strength~0.35`) - add texture, don't repaint.
2. **High Tile-ControlNet weight** - output follows the source's contours.
3. **Mandatory color-fix** ([`colorfix.py`](tile_upscaler/colorfix.py)) - wavelet/AdaIN
   matching to the bicubic source so brightness/shade don't drift.

Seams between tiles are reduced by overlapping-window blending and deterministic
per-tile seeds.

## Compute

These models need a GPU. Target a rented cloud GPU (RunPod / Vast.ai / Lambda);
a single **24 GB** card (RTX 4090 / L4 / A10 / A5000) handles SDXL + ControlNet
at tile sizes comfortably. Rough throughput: Real-ESRGAN ~0.1-0.3 s/tile;
SDXL+ControlNet a few s/tile. A small AOI (hundreds of tiles) is minutes to a
couple of hours of GPU time. A local Intel/AMD iGPU or CPU is not practical for
the diffusion path.

## Install (cloud GPU box, CUDA)

```bash
# 1. Python env
python -m venv .venv && source .venv/bin/activate

# 2. Install torch matching the box's CUDA first (example: CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. The rest
pip install -r requirements.txt
```

> **Real-ESRGAN / basicsr note:** on `torchvision>=0.17`, `basicsr` imports the
> removed `torchvision.transforms.functional_tensor`. One-line fix after install:
> ```bash
> python - <<'PY'
> import importlib.util, pathlib
> spec = importlib.util.find_spec("basicsr")  # locate without importing (import itself crashes)
> p = pathlib.Path(spec.origin).parent / "data" / "degradations.py"
> p.write_text(p.read_text().replace(
>     "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
>     "from torchvision.transforms.functional import rgb_to_grayscale"))
> print("patched", p)
> PY
> ```
> Or just use the `swin2sr` baseline backend, which installs cleanly.

## Inputs you provide

- **Source raster tiles** under `data/raster/<z>/<x>/<y>.png` (you said you handle
  obtaining and structuring these).
- **OSM data**: either a local `.osm.pbf` extract covering your AOI (recommended;
  set `paths.osm_pbf` in the config) or rely on the live Overpass fallback for
  tiny areas.
- *(optional)* genuine **high-zoom tiles** under `data/hr/...` to enable the
  ground-truth degradation test (`eval.hr_root`).

## Run the whole experiment

```bash
cp config.example.yaml config.yaml      # edit paths + AOIs
python run_experiment.py --config config.yaml
# add --limit 8 for a quick smoke test on a few tiles
```

Outputs:
- `out/up/<method>/` - upscaled images (same z/x/y keys, larger images)
- `out/serve/<method>/` - re-cut deeper-zoom tiles ready to serve
- `out/metrics/*.csv` - PSNR/SSIM/LPIPS (if HR), consistency, no-reference
- `out/sheets/` - side-by-side comparison panels
- `out-results.zip` - *(optional)* entire `out/` tree, if `paths.archive_out` is set

### Downloading results from RunPod

RunPod's S3 API is flaky for hundreds of individual tile files. Prefer a single
archive:

```yaml
# config.yaml
paths:
  archive_out: out-results.zip
```

Or zip an existing run without re-processing:

```bash
python run_experiment.py --archive-only out-results.zip
```

Then download one file from your Mac:

```bash
aws s3 cp s3://YOUR_VOLUME/out-results.zip ./out-results.zip \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
unzip out-results.zip
```

See [`scripts/runpod_s3_download.sh`](scripts/runpod_s3_download.sh) if you still
need individual files.

## Run steps individually

Every module is runnable on its own:

```bash
# Tile math
python -m tile_upscaler.tiles list --bbox -122.42,37.78,-122.41,37.79 --zoom 16

# Render OSM palette controls, building-edge controls, and rich prompts
python -m tile_upscaler.osm_render --pbf data/region.osm.pbf \
    --tiles-file tiles.txt --out out/osm --size 1024
# -> out/osm/z/x/y.png (palette)  out/osm/edges/z/x/y.png (building outlines)

# Baseline (no vector)
python -m tile_upscaler.upscale_baseline --src data/raster --out out/up/A_realesrgan \
    --backend realesrgan --scale 4

# Vector-guided diffusion (variant C: + OSM ControlNet on building edges)
python -m tile_upscaler.upscale_controlnet --src data/raster --out out/up/C \
    --osm out/osm --prompts out/osm/prompts.json --use-osm \
    --scale 4 --strength 0.35 --tile-scale 0.9 --osm-scale 0.45

# Cross-scale consistency + (if you have HR) the degradation test
python -m tile_upscaler.eval make-degraded --hr data/hr --lr out/lr --factor 4
python -m tile_upscaler.eval score --hr data/hr --sr out/up/C --csv out/metrics/C.csv
python -m tile_upscaler.eval consistency --lr out/lr --sr out/up/C
python -m tile_upscaler.eval sheet --lr out/lr --out out/sheets \
    --variants A=out/up/A_realesrgan C=out/up/C --hr data/hr

# Re-cut upscaled images into deeper-zoom XYZ tiles
python -m tile_upscaler.retile --src out/up/C --out out/serve/C --factor 4
```

## Evaluating "does vector help?"

The honest test is the **degradation experiment**: provide real high-zoom (HR)
tiles, let the runner bicubic-downsample them to LR, upscale, and score the
reconstruction against the real HR (PSNR/SSIM/LPIPS). Compare A vs B vs C on the
same tiles. Without HR you still get cross-scale consistency + no-reference
metrics and visual sheets, but no ground truth.

## SGDM (stretch)

```bash
python -m tile_upscaler.sgdm_runner setup                 # clone the repo
python -m tile_upscaler.sgdm_runner prepare --lr out/lr --osm out/osm --dst external/sgdm_io
# ...download SGDM checkpoints per its README (start with no-map weights)...
python -m tile_upscaler.sgdm_runner run --input external/sgdm_io/lr --output external/sgdm_out
python -m tile_upscaler.sgdm_runner collect --flat external/sgdm_out --out out/up/SGDM
```
Note: the ready-to-run SGDM weights are the *no-map* variant; the fully
vector-conditioned path is experimental and may need extra training.

## Module map

| File | Role |
|------|------|
| [`tiles.py`](tile_upscaler/tiles.py) | slippy-map z/x/y <-> bbox math, AOI enumeration, child/neighbour tiles |
| [`osm_render.py`](tile_upscaler/osm_render.py) | OSM -> palette + building-edge controls + rich text prompts |
| [`colorfix.py`](tile_upscaler/colorfix.py) | wavelet / AdaIN color matching for cross-scale consistency |
| [`upscale_baseline.py`](tile_upscaler/upscale_baseline.py) | Real-ESRGAN / Swin2SR (no vector) |
| [`upscale_controlnet.py`](tile_upscaler/upscale_controlnet.py) | SDXL + ControlNet (Tile [+ OSM]) diffusion upscaler |
| [`eval.py`](tile_upscaler/eval.py) | degradation test + consistency + no-reference + sheets |
| [`retile.py`](tile_upscaler/retile.py) | cut upscaled images into deeper-zoom XYZ tiles |
| [`sgdm_runner.py`](tile_upscaler/sgdm_runner.py) | stretch: SGDM integration |
| [`run_experiment.py`](run_experiment.py) | end-to-end orchestrator driven by `config.yaml` |

## Status

Experimental research scaffold. Expect to tune `strength`, ControlNet weights and
prompts per land type. Vector guidance tends to help most with structure (sharp
edges, road/field/water boundaries) at large upscale factors.
