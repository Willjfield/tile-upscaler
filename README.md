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

**New to the methods?** See **[Methods A, B, and C — explained for non-experts](docs/methods-explained.md)** for a plain-language walkthrough of what each approach does, how they differ, and what to look for in the results.

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

**RunPod users:** skip the manual steps below — use [`scripts/runpod_setup.sh`](scripts/runpod_setup.sh)
each time you start a **new pod** on your network volume (see [RunPod workflow](#runpod-network-volume)).

Manual install (any cloud GPU box):

```bash
# 1. Python env
python -m venv .venv && source .venv/bin/activate

# 2. Install torch matching the box's CUDA first (example: CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Blackwell GPUs (RTX 50xx, RTX PRO 4500, sm_120) need cu128 instead:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. The rest
pip install -r requirements.txt
bash scripts/install_pyrosm.sh   # local .osm.pbf reader (optional but recommended)
```

> **Real-ESRGAN / basicsr note:** on `torchvision>=0.17`, `basicsr` imports the
> removed `torchvision.transforms.functional_tensor`. `scripts/runpod_setup.sh`
> patches this automatically; for manual installs:
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

## RunPod (network volume)

RunPod pods are ephemeral; a **network volume** persists your repo, tiles, OSM
extract, config, and experiment outputs across pod swaps. The Python virtualenv
does **not** survive cleanly — `.venv/bin/python` is a symlink to the system
interpreter on the pod where you created it. On a new pod image that path often
does not exist, so you get `No such file or directory` even though `.venv/`
is visible on the volume.

### Quick start on every new pod

SSH in, then:

```bash
cd /workspace/tile-upscaler    # or wherever the volume mounts the repo
git pull                       # if you pushed changes from your Mac
bash scripts/runpod_setup.sh --recreate
source .venv/bin/activate
python run_experiment.py --limit 5    # smoke test
python run_experiment.py              # full run
```

`--recreate` is recommended on every new pod. Without it, the script only
rebuilds the venv if the existing one is broken.

Optional flags:

```bash
bash scripts/runpod_setup.sh --recreate --skip-pyrosm   # if you use Overpass only
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/runpod_setup.sh --recreate
```

**Blackwell GPUs** (RTX 50-series, RTX PRO 4500, `sm_120`): the setup script
auto-selects `cu128` PyTorch. If you installed `cu124` manually you will see
`no kernel image is available for execution on the device` — upgrade torch:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The script will:

1. Infer the PyTorch CUDA wheel index from `nvidia-smi` (or use `TORCH_INDEX_URL`)
2. Recreate `.venv` and install `torch`, `requirements.txt`, and `pyrosm`
3. Patch `basicsr` for Real-ESRGAN
4. Set `HF_HOME` to `<repo>/.cache/huggingface` on the volume (so SDXL weights
   are not re-downloaded every pod) and append it to `~/.bashrc`

### What persists vs what you redo

| On the network volume (keeps) | Per new pod (redo via `runpod_setup.sh`) |
|-------------------------------|------------------------------------------|
| Repo, `data/raster/`, `.osm.pbf` | Python `.venv` |
| `config.yaml`, `out/` results | `pip install` / torch |
| `.cache/huggingface/` model weights | `basicsr` patch |
| | SSH key in `authorized_keys` (unless your template injects it) |

### Checklist beyond first-time install

When you spin up a **new** pod on an existing volume:

1. **Attach the correct network volume** — confirm the mount path (`/workspace/...`
   vs `/tile-upscaler/...`) and `cd` into the repo.
2. **Run `bash scripts/runpod_setup.sh --recreate`** — do not assume an old
   `.venv` works on a new container image.
3. **`source .venv/bin/activate`** before running Python (or use `.venv/bin/python` directly).
4. **`git pull`** if you changed code locally and pushed to GitHub.
5. **`config.yaml`** — set `paths.osm_pbf` to your extract (e.g.
   `data/glasgow_extract.osm.pbf`); `null` falls back to Overpass (slow, rate-limited).
6. **Verify GPU** — `nvidia-smi` should show your card; then
   `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.
7. **SSH** — register your public key in RunPod Settings → SSH keys; keys added
   after a pod starts may need to be injected into `~/.ssh/authorized_keys` manually.

### Recreating `.venv` manually

If you prefer not to use the script:

```bash
cd /workspace/tile-upscaler
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash scripts/install_pyrosm.sh
# basicsr patch — see Install section above
```

Diagnose a broken venv:

```bash
ls -la .venv/bin/python .venv/bin/python3   # symlinks — note the target path
which python3 && python3 --version
.venv/bin/python -c "import sys"            # fails if symlink target is missing
```

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
aws s3 cp s3://YOUR_VOLUME/tile-upscaler/out-results.zip ./out-results.zip \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
unzip out-results.zip
```

See [`scripts/runpod_s3_download.sh`](scripts/runpod_s3_download.sh) if you still
need individual files.

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
| [`docs/methods-explained.md`](docs/methods-explained.md) | plain-language guide to methods A / B / C |
| [`scripts/runpod_setup.sh`](scripts/runpod_setup.sh) | bootstrap venv + deps on a new RunPod pod |
| [`scripts/runpod_s3_download.sh`](scripts/runpod_s3_download.sh) | download files from a RunPod volume via S3 |
| [`scripts/install_pyrosm.sh`](scripts/install_pyrosm.sh) | install pyrosm (not in requirements.txt) |

## Status

Experimental research scaffold. Expect to tune `strength`, ControlNet weights and
prompts per land type. Vector guidance tends to help most with structure (sharp
edges, road/field/water boundaries) at large upscale factors.
