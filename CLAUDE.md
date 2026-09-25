# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A generator for a Farming Simulator 25 "x4" map ("Granja Bonita", 4096 m playable). It
produces two artefacts a human then imports into Giants Editor: a 16-bit heightmap PNG
(`dem_generator/dem_8k.png`, 8192 px: 2048 m of border either side of the playable
square) and an OSM vector file (`osm_generator/map.osm`). There is no application and no
test suite. Verification is two acceptance scripts that exit non-zero, plus
`map_layout.validate()`, which both generators run first and refuse to proceed on.

The map is the 16x map (`~/git/FS25_Granja_Bonita_x16`) converted once to a quarter of
the area; see "Where the inputs come from". This repo is independent of that one.

The user works in Spanish (commit messages, `pf_generator/README.md`, `render_pda.py`);
code and most docstrings are in English. Either language is fine in replies.

## Commands

    python3 map_layout.py                             # layout self-check, ~3 s, no output files
    python3 dem_generator/generate_dem.py             # ~20 s -> dem_8k.png, terrain_stats.json, 2 preview PNGs
    python3 dem_generator/measure_elevation.py        # acceptance report on the DEM, exit 1 on failure
    python3 osm_generator/generate_osm.py             # -> map.osm
    python3 osm_generator/check_osm.py                # inventory + invariants, exit 1 on failure
    python3 osm_generator/visualize_osm.py            # -> map_osm_visual.png (2D render)
    python3 visualizer/create_3d_viewer.py            # -> dem_viewer_3d.html (Three.js, DEM + OSM together)

Run the DEM before the OSM: `generate_osm.py` reads `dem_generator/terrain_stats.json`,
which the DEM publishes. Scripts work from the repo root or their own directory; each
puts what it needs on `sys.path`. System `python3` (3.14) has numpy, scipy, Pillow,
matplotlib and contourpy. There is no `.venv`.

Not part of the pipeline:

    python3 tools/convert_from_x16.py [--from DIR]    # ~25 s: rewrites input/ from the 16x repo's inputs
    python3 pf_generator/generate_soil.py -s <seed>   # Precision Farming soilMap.png, pure seeded noise
    python3 render_pda.py                             # Windows-only: paints overview.dds for the mod folder via texconv.exe, hard-coded paths

## Architecture

**`map_layout.py` is the single source of geometry.** The DEM sculpts around it and the
OSM writes it out; neither half may define geometry of its own. A feature in one output
and not the other is invisible in either on its own. Standard library only: nothing in
`osm_generator/` may import numpy, and terrain facts the OSM side needs come through
`terrain_stats.json` via `map_layout.load_roughness()`, never re-derived.

**The geometry comes from `input/custom_osm.osm`, not from code.** The loader block at
the bottom of `map_layout.py` (look for `_INPUT_OSM`) parses that JOSM file at import
time and fills the registries. It raises at import if the file's `<bounds>` do not span
`PLAYABLE_M`: nodes are placed from the bounds' north-west corner, so a survey of another
size would load without complaint and put everything in the wrong place.

| Registry | Filled from | What it is |
|---|---|---|
| `CORRIDORS` | `highway=*` ways | roads: axis, class, platform half-width, feather, grade |
| `WATER` | `natural=water` | the one lake, as a shore ring |
| `PADS` | `landuse=farmyard`, `place=town` | levelled platforms (terrain only; `m4fs:level=yes` marks which yards get graded) |
| `AREAS` | woods, fields, farmyards | every tagged ring the OSM draws; `FIELDS` is the farmland subset |
| `RAILWAY` | **derived** by `build_railway()` | one `railway=rail` corridor offset `RAIL_OFFSET_M` from `RAIL_ALONG_ROAD`, run to the boundary at both ends; every ring it passes through is clipped to its reserve (`_cut_ring_by_axis`), split in two where it crosses |
| `SHELTERBELTS` | **derived** by `build_shelterbelts(FIELDS)` | laid after the railway, so they keep off it; fields are cut back to make room |

Ways can be dropped by id in two sets (`_TOWN_RESERVOIR_WAYS`, `_DROPPED_WAYS`) and
retagged as fields in a third (`_FARMLAND_WAYS`); all three are empty since the
conversion, which left the dropped ways out of the survey it wrote. The mechanism stays
so the survey can be edited in JOSM and a way taken out of the build without touching
it. The wood on way `EAST_RIDGE_WAY` is stretched out to the clean strip because the DEM
runs that ridge into the border, then split into `WOOD_PARCELS` equal-area parcels
(`split_ring_across`) so it can be bought piecemeal; the whole stretched ring survives as
`EAST_RIDGE_RING`, which is what `extend_east_ridge` reads. `_LEVELLED_NAME` marks the
"Granja N" yards for levelling.

**Much of `map_layout.py` is dead code from the previous procedural map.** The river,
PLSS road grid, towns, roadside yards, gallery timber and aliquot parcelling (roughly
lines 160 to 2600: `river_axis`, `_ns_road`, `town_*`, `roadside_pad`, `gallery_areas`,
`build_fields`, and their `RIVER_*`/`PLSS_*`/`TOWN_*`/`FIELD_*`/`GALLERY_*` constants) still
compile and some constants are still referenced (`TOWN_PAD_FEATHER_M`, `YARD_FEATHER_M`,
`FIELD_MIN_HA`, `FIELD_MIN_SIDE_M`, `FIELD_CORNER_R_M`, the `SHELTER_*` block, the `RIM_*`
block), but none of that geometry reaches the output. The module docstring and many
comments still describe the old map. Trust `python3 map_layout.py` and the loader block
over any prose.

**The DEM generator sculpts the source, it does not synthesise.** `generate_dem.py`
reads `input/valle_bonito.png`, which is already the 8192 px canvas (border and all), and
works the playable square in playable metres at 1 m/px: `roughen_till` (the swell and
swale of the till, from the `TILL_*` block in the layout; off on the lake, the boundary,
the eastern ridge and the deliberately flat strip along the north edge, which it detects
rather than draws), `level_platforms` (the one place `PADS` reach the ground),
`sculpt_western_lake` (the basin under the lake ring), then on the whole canvas
`extend_east_ridge` (the ridge under the eastern wood run into the border range) and
`blend_apron` (the border brought down to the playable edge over `RIM_APRON_M`). The
border is otherwise the source's, verbatim. The `RIM_*` constants in the layout describe
nothing that is built any more; `validate()` still checks one of them.

**The acceptance harness.** `measure_elevation.py` checks canvas size, encoding,
elevation bands, the steepest slope, the till's slope percentiles, that the source's
flat ground came through flat, and that `terrain_stats.json` matches the PNG.
`check_osm.py` checks node and ring integrity, the closed tag vocabulary, the clean
strip, and that the file and the layout agree. Both call `validate()`. When adding a
placement rule, add it to `validate()` rather than only fixing coordinates.

Support modules: `osm_generator/map_extent.py` and `map_source.py` are re-export shims so
the OSM scripts and the 3D viewer read the projection from `map_layout`. `dem_generator/terrain_ops.py`
holds the terrain primitives (`smootherstep`, `value_noise`, `rect_sdf`, `slope_deg`, and
older ones nothing calls now).

`FS25_Granja_bonita/` and `FS25_Granja_Bonita_V3/` are Maps4FS-generated mod folders
**for the 16x map** (8192 m). Nothing in this pipeline writes into them; the x4 mod folder
has to be generated with Maps4FS at 4096 m, and getting the DEM and OSM in there is a
manual Giants Editor step.

## Where the inputs come from

`input/custom_osm.osm` (4096 m) and `input/valle_bonito.png` (8192 px) were written once
by `tools/convert_from_x16.py` from the 16x repo's inputs, and the survey is meant to be
edited in JOSM from there. Re-running the conversion overwrites both. What it does, so
its output makes sense:

- **A separable warp** `fx`, `fy` takes 8192 survey metres to 4096. The scale is 1.0 over
  the x and y bands spanned by `KEEP_REAL_SIZE_WAYS` (the two towns and Granja 1 to 3,
  which keep their real dimensions) and over the outer 60 m of each axis (so distances to
  the clean strip are kept), and about 0.32 in x / 0.38 in y elsewhere, with 80 m ramps.
  Separable, so grid roads stay straight and nothing crosses that did not cross before.
  Heights are not scaled: slopes in the playable square are up to three times the
  survey's. The DEM's playable square goes through the inverse of the same warp; its
  2048 m border keeps its width at 1:1 and is compressed only along the edge.
- **Every way that is not a field is copied** with its id and tags. The ids the layout
  names (`EAST_RIDGE_WAY = 517`, the "Granja N" yards) point at the same things.
- **Fields are reparcelled, not warped.** Warped field rings are rasterised at 1 m, the
  hedgerow gaps closed, roads, yards, woods, lake (40 m off) and the clean strip cut out,
  and every block of field ground is bisected recursively across its long side until the
  pieces are under 1.5 times the survey's median field (23.5 ha), with an 18 m gap
  between parcels. Blocks smaller than one field merge across the tertiary lane that
  bounds them, so **lanes that serve nothing but fields are dropped** (never primaries,
  secondaries, or a lane within 15 m of a yard or town); dead-end fragments under 150 m
  go with them. The result is 62 fields where the survey had 162.
- **Two repairs to the source DEM are baked in**: the old river's outlet trenches through
  the border are closed (four of them), and the corner where the old town and reservoir
  were is flattened. The generator no longer does either.

The knobs are the constants at the top of the script; `EAST_RIDGE_SEARCH_M` in
`generate_dem.py` (770 m) is 1500 survey metres through the warp and is printed by the
conversion so it can be updated if the bands change.

## Coordinates and encoding

- Playable metres: **x east, y south from the north edge**, centre `(2048, 2048)`. Canvas
  is 8192 m with the 4096 m playable square centred, so canvas coordinates run
  `-2048 .. 6144` in the same frame.
- Projection is equirectangular about `LAT_CENTER, LON_CENTER` at 111111.0 m per degree.
  The loader derives its own centre from the input file's `<bounds>`; the module constants
  match it. Moving either moves every node in `map.osm` relative to the heightmap and
  nothing downstream catches it.
- Heights are 16-bit centimetres: raw 4640 = 46.40 m. `BASE_ELEV_M` (46.4) is a datum, not
  a height anything sits at. The playable plain sits in the thirties.
- Everything is 1 m/px on the canvas, output pixel `i` at `i + 0.5`. `roughen_till` builds
  its relief on a 4 m grid (`TILL_PX`) and resamples; `level_platforms` works directly at
  1 m/px in playable metres.
- A 50 m clean strip (`EDGE_CLEAR_M`) inside the playable boundary: every ring the OSM
  draws is clipped back to it (`strip_ring`), pads are held inside it by extent, and
  `check_osm.py` fails on anything planted in it. Roads and water are exempt.

## OSM tag vocabulary is closed

Emit only what `map_layout.RENDERED_TAGS` lists: `natural=water` (+ `water=*`),
`natural=wood`, `landuse=forest|farmyard|farmland`, `highway=*`, `railway=*`. Both
renderers (`visualize_osm.py`, `create_3d_viewer.py`) silently drop anything else, and
`check_osm.py` fails the build over a way whose only tags are outside the list.
Attribute tags on a drawable ring (`leaf_type`, `building`) are fine. Every wood ring
carries both `natural=wood` and `landuse=farmyard` (`wood_tags()`); all three renderers
test wood first, so that is safe. Rings must close on the same node id or the 3D viewer
draws them as lines.

## Pitfalls that still apply

The full list of bugs found on the previous map is in the old CLAUDE.md at commit
`4926122` of the 16x repo ("Things that have already gone wrong here"). The ones the
current code can hit:

- **Clip a ring, do not clamp it.** Clamping folds overhanging vertices onto the boundary.
  `clip_ring_to_rect` / `strip_ring` only ever put a vertex on the ring's own edge.
- **Offsetting a polyline** by more than its radius of curvature folds the ring through
  itself (`offset_polyline`). `_primary_belt_runs` drops a folded ring rather than lay
  it; the compressed bend of Mountain Pass Road is where it happens.
- **A vertex-in-polygon test misses two rectangles crossing in a plus.** Use
  `rings_overlap`, which tests edges too.
- **Platform feathers must widen with the cut**: `max(nominal, 1.5*|dz|/tan(4 deg))`,
  capped at `FEATHER_CAP_M`. `level_platforms` does this; keep it when touching pads.
- **Arrive at flat ground with `smootherstep`, not `smoothstep`**, where the surface
  meets a datum, or the 4 m to 1 m resample rings on the curvature jump.
- **Measure the thing, not its average.** Slope is read over a 5 m baseline because the
  centimetre quantisation gives a ~0.3 degree noise floor per pixel.
- **Determinism**: no floating-point randomness in alignments; the DEM uses named RNG
  streams with fixed indices (`STREAMS`) and seed `map_layout.SEED`.
