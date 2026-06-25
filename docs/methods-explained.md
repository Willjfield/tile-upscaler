# Methods A, B, and C — explained for non-experts

This page explains what the three upscaling methods in this project actually do,
in plain language. You do not need to understand neural networks, diffusion
models, or ControlNet to follow along — though a passing familiarity with “AI
models take an input image and produce an output image” is enough.

For setup and running the experiment, see the [main README](../README.md).

---

## The problem we are solving

You have **aerial map tiles**: small square images that show a patch of ground
from directly above (like a satellite photo). At a given zoom level they look
fine, but when a user zooms in further the map would ideally show **more detail**
— sharper rooftops, clearer road edges, finer texture on fields and water.

We do not have higher-resolution photos for every area. So we use **machine
learning upscaling**: software that takes a lower-resolution tile and produces a
larger, sharper-looking version.

The hard part is not “make it bigger.” A simple resize (bicubic interpolation)
makes pixels bigger but does not add real detail. AI can **invent plausible
detail** — but it can also get things wrong: wrong building shapes, colours that
do not match the wider map, or a “repainted” look that breaks when you zoom.

This project compares three approaches and asks: **does OpenStreetMap (OSM) data
help the AI upscale more faithfully?**

OSM is a collaborative map of the world: building footprints, roads, land use,
names, and so on. It is **vector data** (points, lines, polygons), not a photo.
We use it as extra context for the upscaler.

---

## What every method starts with

All three methods receive the **same source tile**: a PNG from `data/raster/z/x/y.png`
for your area of interest.

Methods B and C also use **OSM for that geographic patch**, which we turn into:

1. **A text description** (prompt) — e.g. “dense buildings and rooftops, asphalt
   roads, parkland” — summarising what OSM says is in the tile.
2. **(Method C only)** **A building-outline image** — white lines on black,
   tracing building footprints — aligned pixel-for-pixel with the tile.

Method A ignores OSM entirely. That makes it our **baseline**: “how good is
upscaling with only the photo, no map data?”

---

## Method A — Real-ESRGAN (photo only)

**Config name:** `baseline_realesrgan`  
**Output folder:** `out/up/A_realesrgan`

### What it is

Real-ESRGAN is a **super-resolution** model: a type of neural network trained on
millions of image pairs to turn blurry or small images into sharper, larger ones.
It is fast, widely used, and does not “imagine” scenes from text — it only looks
at the pixels you give it.

Think of it as a very smart sharpen-and-enlarge filter. It learns patterns like
“roof edges often look like this” or “grass texture often looks like that” from
training data, not from a map.

### What it does *not* do

- It does not read street names, building tags, or OSM geometry.
- It cannot know that a fuzzy blob is specifically a church, a warehouse, or a
  terrace house — only that it looks like some kind of rooftop.

### When it tends to work well

Open areas, uniform texture (fields, water), general sharpening.

### When it tends to struggle

Dense urban areas where fine structure matters — individual roof lines, narrow
gaps between buildings, exact road widths. It may blur or merge details that OSM
could have hinted at.

### Speed and hardware

Fastest of the three (roughly a fraction of a second to a few seconds per tile on
a GPU). Smallest memory footprint.

---

## Method B — SDXL + Tile ControlNet + OSM text prompt

**Config name:** `controlnet_text`  
**Output folder:** `out/up/B_controlnet_text`

### What it is

Method B uses a **diffusion model** (Stable Diffusion XL, SDXL). Diffusion
models are the same family of tools used for text-to-image AI art, but here we
use them differently: we start from your **actual aerial tile** and ask the
model to **refine** it rather than paint something new from scratch.

Two ideas matter:

**1. Image-to-image, not blank canvas**  
We upscale the tile with bicubic interpolation first, then run diffusion with
**low strength** — the model is allowed to add texture and crispness, but not to
redraw the whole scene. A **colour-fix** step at the end forces brightness and
shade to stay aligned with the source so zooming in still feels continuous.

**2. Tile ControlNet — “stay faithful to this layout”**  
ControlNet is an add-on that steers diffusion using an extra **control image**.
Here, the control image is the bicubic-upscaled tile itself. It tells the model:
*keep the same large shapes and layout; only improve detail.*

**3. OSM text prompt — “this is what is supposed to be here”**  
Separately, we build a **text prompt** from OSM tags in that tile: building
density, road types, land use, roof materials, and similar. That text is fed to
the model the same way a prompt is fed to an image generator.

So Method B knows things like “there are dense buildings and asphalt roads here”
even when the photo is ambiguous — but it only receives that knowledge as
**words**, not as a drawing.

### Plain analogy

Imagine handing an artist a blurry aerial print and saying:

> “Enhance this photo. Here is a rough upscaled copy — **do not change where
> things are**, only add detail. Also, OpenStreetMap says this block has **dense
> Victorian terraces and a main road along the north edge**.”

The artist uses the map **notes**, not a traced blueprint.

### Trade-offs

- Often richer, more plausible texture than Method A.
- Slower and heavier (large models, several seconds per tile on a GPU).
- Text is vague: “dense buildings” does not tell the model the exact footprint of
  building #42.

---

## Method C — Method B + OSM building-outline ControlNet

**Config name:** `controlnet_osm`  
**Output folder:** `out/up/C_controlnet_osm`

### What it is

Method C is **everything in Method B**, plus a **second ControlNet** driven by
a **building-edge image** derived from OSM: white outlines of building footprints
on a black background, registered to the tile.

So the model now gets:

| Guidance | Type | Source |
|----------|------|--------|
| Layout / contours | Control image | Bicubic-upscaled tile (Tile ControlNet) |
| Semantic context | Text | OSM tags → prompt |
| Building geometry | Control image | OSM footprints → edge map (OSM ControlNet) |

### Plain analogy

Same artist as Method B, but now you also hand them a **tracing paper overlay**
showing exact building outlines from the map:

> “Enhance the photo, keep the layout, remember it is a dense urban block — and
> **these white lines are where building walls meet the ground; respect them**.”

### What we hope Method C improves

- Sharper, better-placed building edges.
- Less merging of adjacent rooftops.
- Structure that matches the map even when the source photo is soft or shadowed.

### Trade-offs

- Slowest and most memory-hungry (two ControlNets plus SDXL).
- Quality depends on OSM coverage and accuracy. Missing or misaligned buildings
  in OSM can mislead the model.
- Roads, trees, and water still rely mainly on the photo and text prompt — the
  extra spatial control is **building outlines** specifically.

---

## Side-by-side summary

| | **A — Real-ESRGAN** | **B — Diffusion + text** | **C — Diffusion + text + outlines** |
|---|---------------------|--------------------------|-------------------------------------|
| **Uses source photo** | Yes | Yes | Yes |
| **Uses OSM** | No | Yes (text only) | Yes (text + building shapes) |
| **How it upscales** | Learned super-resolution | Diffusion refinement | Diffusion refinement |
| **“Do not move the scenery”** | Implicit in training | Tile ControlNet + colour-fix | Tile ControlNet + colour-fix |
| **Map knowledge** | None | Descriptive words | Words + drawn footprints |
| **Typical speed** | Fastest | Slower | Slowest |
| **Best question it answers** | How good is a standard AI upscaler? | Does *knowing what is there* (in words) help? | Does *showing where buildings are* help even more? |

---

## What “better” means in this project

We are not only asking “which looks prettiest?” We care about **seamless zoom**:
when you zoom from a wider view into these tiles, the upscaled detail should feel
like a natural continuation — same lighting, same large shapes — not a different
photo pasted on top.

The pipeline checks this in several ways:

- **Cross-scale consistency** — if we shrink the upscaled tile back down, how
  closely does it match the original? (Colour and large-scale structure should
  align.)
- **Comparison sheets** — side-by-side panels in `out/sheets/` for visual review.
- **No-reference metrics** — automated scores for sharpness / naturalness without
  a ground-truth photo.
- **Degradation test** *(optional)* — if you provide real high-zoom tiles in
  `data/hr/`, we downsample them, upscale, and measure error against the truth.
  That is the fairest A-vs-B-vs-C comparison.

---

## What to look for when you open the results

When comparing `out/sheets/` or `out/up/`:

1. **Building edges** — Are rooftops distinct? Do outlines match what you know is
   on the ground? Method C is aimed specifically at this.
2. **Roads and open land** — Often similar across B and C; Method A may look
   softer or smoother.
3. **Colour drift** — Does the upscaled tile look like the same sunny/overcast
   scene as its parent zoom level? Large shifts suggest the diffusion step
   “repainted” too aggressively.
4. **Hallucinated junk** — Extra chimneys, roads that bend wrong, water texture
   that ripples unnaturally. More creative models (B/C) can do this if settings
   are too strong.

None of the methods invents **new geography** reliably — they embellish what is
already in the tile and (for B/C) what OSM claims is there.

---

## How the experiment is organised (one sentence each)

- **Method A** establishes a strong, map-agnostic baseline.
- **Method B** tests whether **semantic** OSM knowledge (text) improves diffusion
  upscaling.
- **Method C** tests whether adding **spatial** OSM knowledge (building outlines)
  improves further, on top of B.

All three run on the same tiles so differences in the outputs are attributable to
the method, not the input data.

---

## Glossary (minimal)

| Term | Meaning here |
|------|----------------|
| **Tile** | One square map image at a specific zoom and grid position (`z/x/y`). |
| **Upscaling / super-resolution** | Producing a larger, sharper image from a smaller one. |
| **Diffusion model** | A generative model that refines noise into an image; here constrained to stay near the source. |
| **ControlNet** | Extra inputs (images or edges) that steer a diffusion model so outputs follow structure. |
| **Prompt** | Short text describing the scene, derived from OSM tags. |
| **OSM** | OpenStreetMap — collaborative vector map data (roads, buildings, etc.). |
| **Bicubic** | A simple, traditional resize — used as a safe starting point before AI refinement. |

---

## Further reading in this repo

- [README — How it works](../README.md#how-it-works) — technical pipeline diagram
- [README — Evaluating "does vector help?"](../README.md#evaluating-does-vector-help) — how we score the methods
- `config.yaml` — turn methods on/off under `methods:`
