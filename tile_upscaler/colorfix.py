"""Color / brightness matching for cross-scale consistency.

The diffusion upscalers may add detail *and* subtly shift tone (brightness,
white balance, saturation). For a slippy map we want the opposite: the
super-resolved tile must read as a faithful refinement of the lower-zoom view -
same brightness and shade - so zooming in feels seamless. Hallucinated
high-frequency texture is fine; hallucinated *color* is not.

This module re-aligns a generated (target) image's color/luminance to a
reference derived from the original low-res tile (a bicubic upscale of it). Two
classic methods from StableSR are provided:

- ``adain_color_fix``:   match per-channel mean/std (AdaIN in color space).
- ``wavelet_color_fix``: keep the target's high frequencies, swap in the
  reference's low frequencies (so only fine texture comes from the model).

Both operate on PIL images and use only numpy/scipy, so they have no GPU or
torch dependency and can be applied as a cheap post-process to any upscaler.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required for colorfix (pip install pillow)") from exc


ColorFixMethod = Literal["adain", "wavelet", "none"]


def _to_array(img: "Image.Image") -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> "Image.Image":
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _match_size(reference: "Image.Image", target: "Image.Image") -> "Image.Image":
    if reference.size != target.size:
        reference = reference.resize(target.size, Image.BICUBIC)
    return reference


def adain_color_fix(target: "Image.Image", reference: "Image.Image") -> "Image.Image":
    """Match the target's per-channel mean and std to the reference (AdaIN).

    Fast and effective for global tone/white-balance correction.
    """
    reference = _match_size(reference, target)
    t = _to_array(target)
    r = _to_array(reference)

    t_mean = t.mean(axis=(0, 1), keepdims=True)
    t_std = t.std(axis=(0, 1), keepdims=True) + 1e-6
    r_mean = r.mean(axis=(0, 1), keepdims=True)
    r_std = r.std(axis=(0, 1), keepdims=True)

    out = (t - t_mean) / t_std * r_std + r_mean
    return _to_image(out)


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Per-channel Gaussian blur. Uses scipy if available, else a separable
    numpy fallback."""
    try:
        from scipy.ndimage import gaussian_filter

        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            out[..., c] = gaussian_filter(arr[..., c], sigma=sigma, mode="reflect")
        return out
    except ImportError:
        return _gaussian_blur_numpy(arr, sigma)


def _gaussian_blur_numpy(arr: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    def conv1d(a: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0)] * a.ndim
        pad[axis] = (radius, radius)
        ap = np.pad(a, pad, mode="reflect")
        out = np.zeros_like(a)
        for i, w in enumerate(kernel):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += w * ap[tuple(sl)]
        return out

    out = arr
    out = conv1d(out, 0)
    out = conv1d(out, 1)
    return out


def wavelet_color_fix(
    target: "Image.Image",
    reference: "Image.Image",
    levels: int = 5,
) -> "Image.Image":
    """Replace the target's low-frequency color with the reference's.

    Implements the StableSR "wavelet color fix" using a Gaussian pyramid: the
    image is split into high-frequency residuals (kept from the target) and a
    low-frequency base (taken from the reference). The result therefore inherits
    the model's fine texture but the original image's color/brightness contours.
    """
    reference = _match_size(reference, target)
    t = _to_array(target)
    r = _to_array(reference)

    # Build each image's low-frequency component via progressively wider blurs.
    t_low = t.copy()
    r_low = r.copy()
    for i in range(levels):
        sigma = 2.0 ** i
        t_low = _gaussian_blur(t_low, sigma)
        r_low = _gaussian_blur(r_low, sigma)

    high = t - t_low  # target's high-frequency detail
    out = high + r_low  # reference's low-frequency color
    return _to_image(out)


def apply_color_fix(
    target: "Image.Image",
    reference: "Image.Image",
    method: ColorFixMethod = "wavelet",
) -> "Image.Image":
    """Dispatch helper used by the upscalers."""
    if method == "none":
        return target
    if method == "adain":
        return adain_color_fix(target, reference)
    if method == "wavelet":
        return wavelet_color_fix(target, reference)
    raise ValueError(f"Unknown color-fix method: {method}")


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Color-match an image to a reference")
    parser.add_argument("target", help="Generated/upscaled image")
    parser.add_argument("reference", help="Reference (e.g. bicubic of the source)")
    parser.add_argument("output", help="Where to write the corrected image")
    parser.add_argument("--method", choices=["adain", "wavelet", "none"], default="wavelet")
    args = parser.parse_args(argv)

    tgt = Image.open(args.target)
    ref = Image.open(args.reference)
    out = apply_color_fix(tgt, ref, args.method)
    out.save(args.output)
    print(f"Wrote {args.output} (method={args.method})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
