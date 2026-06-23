"""Vector-guided aerial tile upscaling.

A small experimental pipeline for upscaling aerial XYZ raster tiles, optionally
guided by OpenStreetMap vector data, comparing a no-vector baseline against
OSM-guided diffusion upscaling.

Modules
-------
- tiles:             slippy-map tile math and AOI enumeration
- osm_render:        rasterize OSM into palette + building-edge controls + rich prompts
- colorfix:          wavelet / AdaIN color matching for cross-scale consistency
- upscale_baseline:  Real-ESRGAN / Swin2SR (no vector) upscaling
- upscale_controlnet: SDXL + ControlNet (Tile + optional OSM) diffusion upscaling
- eval:              degradation test + cross-scale consistency + no-reference metrics
- retile:            cut upscaled images into z/x/y XYZ folders
"""

__version__ = "0.1.0"
