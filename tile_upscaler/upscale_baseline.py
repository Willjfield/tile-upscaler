"""No-vector baseline upscalers: Real-ESRGAN and Swin2SR.

These establish the quality floor and the pipeline plumbing *without* any vector
guidance. Everything the vector-guided diffusion path adds is measured against
these (see ``eval.py``).

Backends
--------
- ``realesrgan``: the GAN workhorse (RealESRGAN_x4plus). Fast, faithful, the best
  default for ordinary imagery. Needs ``realesrgan`` + ``basicsr``.
- ``swin2sr``: a transformer SR model via HuggingFace ``transformers``
  (clean to install, good global structure).

Both read an XYZ raster tile tree and write an upscaled tree of the same z/x/y
keys (the images are larger; use ``retile.py`` to cut them into deeper-zoom
tiles for serving).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import tileio
from .tiles import Tile


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class RealESRGANBackend:
    def __init__(self, scale: int = 4, model_name: str = "RealESRGAN_x4plus", device: Optional[str] = None, half: bool = True):
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Real-ESRGAN backend needs torch, basicsr and realesrgan "
                "(pip install torch basicsr realesrgan)"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scale = scale
        self.device = device

        # Model architecture + pretrained weights URL for the common x4 model.
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        )
        self._upsampler = RealESRGANer(
            scale=4,
            model_path=url,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half and device == "cuda",
            device=device,
        )

    def upscale(self, image, outscale: int):
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # RGB->BGR for realesrgan
        out, _ = self._upsampler.enhance(bgr, outscale=outscale)
        from PIL import Image

        return Image.fromarray(out[:, :, ::-1])  # BGR->RGB


class Swin2SRBackend:
    def __init__(self, scale: int = 4, device: Optional[str] = None):
        try:
            import torch
            from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Swin2SR backend needs torch + transformers "
                "(pip install torch transformers)"
            ) from exc

        if scale not in (2, 4):
            raise ValueError("Swin2SR backend supports scale 2 or 4")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scale = scale
        self.device = device

        repo = {
            2: "caidas/swin2SR-classical-sr-x2-64",
            4: "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
        }[scale]
        self._processor = Swin2SRImageProcessor.from_pretrained(repo)
        self._model = Swin2SRForImageSuperResolution.from_pretrained(repo).to(device).eval()

    def upscale(self, image, outscale: int):
        import torch
        from PIL import Image

        inputs = self._processor(image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs).reconstruction.clamp(0, 1)
        arr = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
        result = Image.fromarray(arr)
        if outscale != self.scale:
            target = int(image.width * outscale)
            result = result.resize((target, target), Image.BICUBIC)
        return result


def make_backend(name: str, scale: int = 4, device: Optional[str] = None):
    if name == "realesrgan":
        return RealESRGANBackend(scale=scale, device=device)
    if name == "swin2sr":
        return Swin2SRBackend(scale=scale, device=device)
    raise ValueError(f"Unknown baseline backend: {name}")


# ---------------------------------------------------------------------------
# Tree runner
# ---------------------------------------------------------------------------
def run_tree(
    src_root: str,
    out_root: str,
    backend_name: str = "realesrgan",
    outscale: int = 4,
    device: Optional[str] = None,
    limit: Optional[int] = None,
) -> int:
    """Upscale every tile under ``src_root`` into ``out_root``. Returns count."""
    backend = make_backend(backend_name, scale=outscale, device=device)
    tiles = tileio.find_tiles(src_root)
    if limit:
        tiles = tiles[:limit]

    done = 0
    t0 = time.time()
    for tf in tiles:
        image = tileio.load_image(tf.path)
        out = backend.upscale(image, outscale=outscale)
        tileio.write_tile(out_root, tf.tile, out)
        done += 1
        if done % 20 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(tiles)} tiles ({rate:.1f}/s)", flush=True)
    print(f"Upscaled {done} tiles -> {out_root} (backend={backend_name}, x{outscale})")
    return done


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="No-vector baseline tile upscaler")
    parser.add_argument("--src", required=True, help="Source raster tile tree (z/x/y)")
    parser.add_argument("--out", required=True, help="Output tile tree")
    parser.add_argument("--backend", choices=["realesrgan", "swin2sr"], default="realesrgan")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N tiles")
    args = parser.parse_args(argv)

    run_tree(
        src_root=args.src,
        out_root=args.out,
        backend_name=args.backend,
        outscale=args.scale,
        device=args.device,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
