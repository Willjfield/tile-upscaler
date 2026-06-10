# Tile Fetcher

A tiny local tool for grabbing source aerial tiles for the upscaling pipeline.

```bash
# from the repo root
python tile_fetcher/server.py            # serves http://localhost:8080
python tile_fetcher/server.py --out data/raster --port 8080
```

Open the page, enter an XYZ tile URL template (must contain `{z}/{x}/{y}`;
optional `{s}` subdomains; include any API key directly in the URL, e.g.
`...{z}/{x}/{y}.png?key=YOUR_KEY`), pick a zoom, click
**Draw box** and drag on the map. Confirm the prompt and every tile touching
the box is downloaded into `data/raster/z/x/y.png` - exactly the layout
`run_experiment.py` expects. Already-downloaded tiles are skipped, so re-runs
are cheap.

Note: respect your tile provider's terms of service and rate limits. Bulk
downloading is against the ToS of some free providers (e.g. OSM's public tile
servers); use a provider/plan that permits it.
