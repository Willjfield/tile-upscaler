"""Vector-guided diffusion upscaler: SDXL + ControlNet (Tile + optional OSM).

This is the heart of the experiment. It upscales an aerial tile with a Stable
Diffusion XL img2img pipeline conditioned by one or two ControlNets:

- **Tile ControlNet** (always on): conditioned on a bicubic upscale of the source
  tile. This is what keeps the output *faithful* to the original - same large
  structures, same layout - so we get texture, not a repaint.
- **OSM ControlNet** (optional, ``--use-osm``): a second ControlNet conditioned
  on edges extracted from the rendered OSM control image. This nudges the model
  to align detail with real-world boundaries (roads, building footprints, field
  and water edges) - the "vector guides the AI" mechanism.

Cross-scale consistency (the key quality requirement) is enforced three ways:
  1. low denoise ``--strength`` (texture, not repaint),
  2. high tile-ControlNet weight (follow the source's contours),
  3. a mandatory color-fix post-step (``colorfix.py``) so brightness/shade match
     the lower-zoom view exactly.

Seams between independently-processed regions are reduced by overlapping-window
("tiled diffusion") processing with feathered blending, and runs are seeded
deterministically per tile.

Method variants (see plan):
  B = tile ControlNet + OSM text prompt           (text-guided vector)
  C = B + OSM-edge ControlNet                      (spatial vector guidance)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import colorfix, tileio
from .tiles import Tile

DEFAULT_NEGATIVE = (
    "blurry, low quality, jpeg artifacts, noise, oversaturated, cartoon, "
    "painting, text, watermark, distorted"
)

# SDXL model ids (overridable via UpscaleConfig).
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_TILE = "xinsir/controlnet-tile-sdxl-1.0"
CONTROLNET_CANNY = "xinsir/controlnet-canny-sdxl-1.0"


@dataclass
class UpscaleConfig:
    outscale: int = 4
    steps: int = 20
    strength: float = 0.35           # low: add texture, don't repaint
    guidance_scale: float = 5.0
    tile_cond_scale: float = 0.9     # high: follow source contours
    osm_cond_scale: float = 0.45
    use_osm: bool = False
    seed: int = 1234
    window: int = 1024               # diffusion window (SDXL native ~1024)
    overlap: int = 128               # feathered overlap between windows
    color_fix: colorfix.ColorFixMethod = "wavelet"
    negative_prompt: str = DEFAULT_NEGATIVE
    device: Optional[str] = None
    base_model: str = SDXL_BASE
    vae_model: str = SDXL_VAE
    tile_model: str = CONTROLNET_TILE
    osm_model: str = CONTROLNET_CANNY


class ControlNetUpscaler:
    def __init__(self, config: UpscaleConfig):
        try:
            import torch
            from diffusers import (
                AutoencoderKL,
                ControlNetModel,
                StableDiffusionXLControlNetImg2ImgPipeline,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ControlNet upscaler needs torch + diffusers "
                "(pip install torch diffusers transformers accelerate)"
            ) from exc

        self.cfg = config
        self.torch = torch
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = dtype

        controlnets = [ControlNetModel.from_pretrained(config.tile_model, torch_dtype=dtype)]
        if config.use_osm:
            controlnets.append(ControlNetModel.from_pretrained(config.osm_model, torch_dtype=dtype))
        controlnet = controlnets if len(controlnets) > 1 else controlnets[0]

        vae = AutoencoderKL.from_pretrained(config.vae_model, torch_dtype=dtype)
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            config.base_model,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
        )
        self.pipe = self.pipe.to(device)
        if device == "cuda":
            self.pipe.enable_vae_tiling()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            self.pipe.enable_attention_slicing()

    # -- control signal helpers ------------------------------------------------
    @staticmethod
    def _canny(image, low: int = 80, high: int = 160):
        """Canny edge map from a (OSM) image, returned as an RGB PIL image."""
        from PIL import Image

        try:
            import cv2

            arr = np.asarray(image.convert("RGB"))
            edges = cv2.Canny(arr, low, high)
        except ImportError:
            # Pure-numpy Sobel fallback if OpenCV is unavailable.
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            gx = np.abs(np.gradient(gray, axis=1))
            gy = np.abs(np.gradient(gray, axis=0))
            mag = np.hypot(gx, gy)
            edges = (mag > mag.mean() + mag.std()).astype(np.uint8) * 255
        return Image.fromarray(np.stack([edges] * 3, axis=-1))

    def _seed_for(self, tile: Tile) -> int:
        h = hashlib.sha256(f"{self.cfg.seed}:{tile.key}".encode()).hexdigest()
        return int(h[:8], 16)

    # -- core ------------------------------------------------------------------
    def upscale_tile(
        self,
        source_image,
        tile: Tile,
        prompt: str,
        osm_control_image=None,
    ):
        """Upscale a single source tile (PIL) -> color-fixed PIL image."""
        from PIL import Image

        target = source_image.width * self.cfg.outscale
        # The bicubic upscale is both the img2img init and the Tile control image.
        base = source_image.convert("RGB").resize((target, target), Image.BICUBIC)

        osm_edges = None
        if self.cfg.use_osm and osm_control_image is not None:
            osm_edges = self._canny(osm_control_image).resize((target, target), Image.NEAREST)

        generator = self.torch.Generator(device=self.device).manual_seed(self._seed_for(tile))
        result = self._tiled_diffuse(base, prompt, osm_edges, generator)

        # Mandatory cross-scale color fix against the bicubic reference.
        return colorfix.apply_color_fix(result, base, self.cfg.color_fix)

    def _run_window(self, init_img, control_imgs, scales, prompt, generator):
        kwargs = dict(
            prompt=prompt,
            negative_prompt=self.cfg.negative_prompt,
            image=init_img,
            num_inference_steps=self.cfg.steps,
            strength=self.cfg.strength,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
        )
        if len(control_imgs) == 1:
            kwargs["control_image"] = control_imgs[0]
            kwargs["controlnet_conditioning_scale"] = scales[0]
        else:
            kwargs["control_image"] = control_imgs
            kwargs["controlnet_conditioning_scale"] = scales
        return self.pipe(**kwargs).images[0]

    def _tiled_diffuse(self, base, prompt, osm_edges, generator):
        """Process ``base`` in overlapping windows with feathered blending.

        For images <= the window size this is a single pass. Larger images are
        split so each window fits SDXL's native resolution; overlaps are blended
        with a 2D Hann window to avoid visible seams.
        """
        from PIL import Image

        W, H = base.size
        T = self.cfg.window
        if W <= T and H <= T:
            controls = [base] + ([osm_edges] if osm_edges is not None else [])
            scales = [self.cfg.tile_cond_scale] + (
                [self.cfg.osm_cond_scale] if osm_edges is not None else []
            )
            return self._run_window(base, controls, scales, prompt, generator)

        stride = T - self.cfg.overlap
        acc = np.zeros((H, W, 3), dtype=np.float32)
        weight = np.zeros((H, W, 1), dtype=np.float32)
        feather = _hann2d(T)[..., None]

        xs = _window_starts(W, T, stride)
        ys = _window_starts(H, T, stride)
        for y0 in ys:
            for x0 in xs:
                box = (x0, y0, x0 + T, y0 + T)
                init_w = base.crop(box)
                controls = [init_w]
                scales = [self.cfg.tile_cond_scale]
                if osm_edges is not None:
                    controls.append(osm_edges.crop(box))
                    scales.append(self.cfg.osm_cond_scale)
                out_w = self._run_window(init_w, controls, scales, prompt, generator)
                acc[y0:y0 + T, x0:x0 + T] += np.asarray(out_w, dtype=np.float32) * feather
                weight[y0:y0 + T, x0:x0 + T] += feather
        blended = acc / np.clip(weight, 1e-6, None)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _hann2d(size: int) -> np.ndarray:
    w = np.hanning(size)
    w = np.clip(w, 1e-3, None)
    return np.outer(w, w).astype(np.float32)


def _window_starts(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


# ---------------------------------------------------------------------------
# Tree runner
# ---------------------------------------------------------------------------
def run_tree(
    src_root: str,
    out_root: str,
    config: UpscaleConfig,
    osm_root: Optional[str] = None,
    prompts_path: Optional[str] = None,
    default_prompt: str = "high-resolution aerial orthophoto, top-down satellite view, sharp realistic detail",
    limit: Optional[int] = None,
) -> int:
    """Upscale every raster tile under ``src_root`` into ``out_root``.

    ``osm_root`` (optional) is an XYZ tree of OSM control images (from
    ``osm_render.py``); ``prompts_path`` is the ``prompts.json`` it emits.
    """
    upscaler = ControlNetUpscaler(config)

    prompts = {}
    if prompts_path and os.path.exists(prompts_path):
        with open(prompts_path) as fh:
            prompts = json.load(fh)

    tiles = tileio.find_tiles(src_root)
    if limit:
        tiles = tiles[:limit]

    done = 0
    t0 = time.time()
    for tf in tiles:
        source = tileio.load_image(tf.path)
        prompt = prompts.get(tf.tile.key, default_prompt)
        osm_img = None
        if config.use_osm:
            osm_path = tileio.find_companion(osm_root, tf.tile)
            if osm_path:
                osm_img = tileio.load_image(osm_path)
        out = upscaler.upscale_tile(source, tf.tile, prompt, osm_control_image=osm_img)
        tileio.write_tile(out_root, tf.tile, out)
        done += 1
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(tiles)} tiles ({rate:.2f}/s) :: {prompt[:60]}", flush=True)
    print(f"Upscaled {done} tiles -> {out_root} "
          f"(osm={'on' if config.use_osm else 'off'}, x{config.outscale})")
    return done


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Vector-guided SDXL+ControlNet tile upscaler")
    parser.add_argument("--src", required=True, help="Source raster tile tree (z/x/y)")
    parser.add_argument("--out", required=True, help="Output tile tree")
    parser.add_argument("--osm", help="OSM control-image tile tree (for --use-osm)")
    parser.add_argument("--prompts", help="prompts.json from osm_render")
    parser.add_argument("--use-osm", action="store_true", help="Enable the OSM ControlNet (variant C)")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--tile-scale", type=float, default=0.9, help="Tile ControlNet weight")
    parser.add_argument("--osm-scale", type=float, default=0.45, help="OSM ControlNet weight")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--color-fix", choices=["wavelet", "adain", "none"], default="wavelet")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    config = UpscaleConfig(
        outscale=args.scale,
        steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance,
        tile_cond_scale=args.tile_scale,
        osm_cond_scale=args.osm_scale,
        use_osm=args.use_osm,
        seed=args.seed,
        color_fix=args.color_fix,
        device=args.device,
    )
    run_tree(
        src_root=args.src,
        out_root=args.out,
        config=config,
        osm_root=args.osm,
        prompts_path=args.prompts,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
