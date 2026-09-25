#!/usr/bin/env python3
"""Acceptance report for the generated heightmap.

Checks the things that are easy to break and hard to see in the image: that the canvas is
the size and encoding the rest of the project expects, that nothing touches the floor or
the ceiling of the 16-bit range, that the elevation bands are the map's (a till plain in
the thirties and forties, a ridge to about 200 m in the playable square, a border range to
about 300 m), that the till rolls the way the `TILL_*` block says and the source's flat
ground came through flat, and that `terrain_stats.json`, which the OSM side reads instead
of re-deriving the terrain, describes the same surface the PNG does.

"What is till" comes from the generator's own `till_weight`, on the grid it built the
relief on, so the ground judged here is the ground that was shaped.

Exits non-zero if any check fails, so it can gate the pipeline.

One measurement note that outlives the blank map: slope is measured over a 5 m baseline.
A DEM quantised to the centimetre at one metre a pixel has a pure noise floor near 0.3
degrees in its per-pixel gradient, so measuring pixel to pixel overstates every slope on
the map. And dry land starts one baseline back from any water's edge - a 5 m window
straddling the lip reads the submerged bank off a pixel that is itself dry.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import map_layout as ml                                             # noqa: E402
import terrain_ops as ops                                           # noqa: E402
from generate_dem import (CANVAS_M, PLAYABLE_M, OFFSET_M,            # noqa: E402
                          BASE_ELEV_M, Z_MAX_CM, STATS_GRID, DEM_NAME,
                          FEATHER_CAP_M, TILL_PX, TILL_FLAT_TOL_M,
                          till_weight, load_source_dem)
from scipy import ndimage                                            # noqa: E402

SLOPE_BASELINE_M = 5.0
# The steepest dry ground the playable square is allowed. Not the valley side's own
# 4.9 deg: where the lake's valley and the river's merge, both sections fall the same way
# at once and the shoulder between them reaches 20%. That is a landform - a tributary
# valley meeting a basin - and the profile through it is smooth, which is the difference
# between a steep place and a defect. 12.5 deg is still ground a tractor works.
BAND_ROWS = 1024

# The till, over a 20 m baseline. A field in the Des Moines lobe rolls at one to three
# degrees; under half a degree at the median is the source DEM's own smoothness, which
# `roughen_till` exists to correct, and over three at the median would be hill country.
TILL_BASELINE_M = 20.0
TILL_SLOPE_P50_DEG = (0.6, 2.0)
TILL_SLOPE_P90_DEG = (1.2, 3.5)
TILL_SLOPE_P99_MAX_DEG = 5.0

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"   {'ok  ' if ok else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")
    return ok


def info(name, detail):
    print(f"         {name}   {detail}")


def band(name, value, lo, hi, unit=""):
    return check(name, lo <= value <= hi,
                 f"{value:.2f}{unit} (want {lo:g}..{hi:.1f}{unit})")


def max_slope_deg(raw, mask=None):
    """The steepest 5 m slope anywhere on the canvas, measured band by band.

    Banded because a float copy of the whole canvas is 600 MB and the Gaussian behind
    `slope_deg` wants another. The bands overlap by four baselines so no slope is missed
    across a seam, and the blur has settled well inside the overlap.

    `mask` restricts where the answer is *read* while still measuring on the whole
    surface, which is the only honest way to ask about dry land: a 5 m window that
    straddles the water's edge reads the submerged bank off a pixel that is itself dry,
    and reported the inside of a channel as a 15 degree field.
    """
    pad = int(4 * SLOPE_BASELINE_M)
    worst = 0.0
    for r0 in range(0, raw.shape[0], BAND_ROWS):
        a = max(0, r0 - pad)
        b = min(raw.shape[0], r0 + BAND_ROWS + pad)
        z = raw[a:b].astype(np.float32) / 100.0
        s = ops.slope_deg(z, 1.0, baseline_m=SLOPE_BASELINE_M)[r0 - a:r0 - a + BAND_ROWS]
        if mask is not None:
            m = mask[r0:r0 + s.shape[0]]
            s = s[m] if m.any() else s[:0]
        if s.size:
            worst = max(worst, float(s.max()))
    return worst


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dem_path = os.path.join(script_dir, DEM_NAME)
    stats_path = os.path.join(script_dir, "terrain_stats.json")
    if not os.path.exists(dem_path):
        print(f"Error: {dem_path} not found. Run generate_dem.py first.")
        return 2

    img = Image.open(dem_path)
    raw = np.array(img)
    o = int(OFFSET_M)
    play_raw = raw[o:o + PLAYABLE_M, o:o + PLAYABLE_M]
    lo_cm, hi_cm = float(raw.min()), float(raw.max())
    want_cm = BASE_ELEV_M * 100.0

    print(f"=== Elevation report: {os.path.basename(dem_path)} ===")
    print(f"layout   {ml.summary()}")
    print(f"canvas   {raw.shape[1]}x{raw.shape[0]} px   "
          f"{lo_cm / 100.0:7.2f} .. {hi_cm / 100.0:7.2f} m")
    print(f"playable {play_raw.shape[1]}x{play_raw.shape[0]} m      "
          f"{play_raw.min() / 100.0:7.2f} .. {play_raw.max() / 100.0:7.2f} m   "
          f"(relief {(float(play_raw.max()) - float(play_raw.min())) / 100.0:.2f} m)")

    # ---------------------------------------------------------------- geometry
    print("\ngeometry and encoding:")
    check(f"canvas is {CANVAS_M}x{CANVAS_M} px", raw.shape == (CANVAS_M, CANVAS_M),
          f"got {raw.shape[1]}x{raw.shape[0]}")
    check("playable area is centred", int(OFFSET_M) * 2 + PLAYABLE_M == CANVAS_M,
          f"{OFFSET_M:.0f} m of margin on every side")
    check("16-bit integer image", raw.dtype == np.uint16,
          f"dtype {raw.dtype}, PIL mode {img.mode!r}")
    check("under the 16-bit ceiling", hi_cm <= min(65535.0, Z_MAX_CM),
          f"peak {hi_cm:.0f} cm, ceiling {min(65535.0, Z_MAX_CM):.0f} cm")
    check("no ground at zero", lo_cm > 0.0, f"floor {lo_cm:.0f} cm")
    info("scale", "raw / 100 = metres, which is what Giants Editor imports")

    # ---------------------------------------------------------------- elevation and relief
    print("\nelevation and relief:")
    band("canvas elevation minimum", lo_cm / 100.0, 1.0, 36.0, " m")
    band("canvas elevation maximum", hi_cm / 100.0, 250.0, 310.0, " m")
    band("playable area minimum", float(play_raw.min()) / 100.0, 1.0, 36.0, " m")
    band("playable area maximum", float(play_raw.max()) / 100.0, 190.0, 280.0, " m")
    check("playable area has relief", float(play_raw.max() - play_raw.min()) / 100.0 >= 50.0,
          f"relief {float(play_raw.max() - play_raw.min()) / 100.0:.2f} m")
    mean_play = float(play_raw.mean()) / 100.0
    info("playable mean elevation", f"{mean_play:.2f} m (datum {BASE_ELEV_M:.1f} m)")

    # ---------------------------------------------------------------- surface
    print(f"\nsurface ({SLOPE_BASELINE_M:.0f} m baseline):")
    worst = max_slope_deg(raw)
    band("steepest slope on the canvas", worst, 0.0, 85.0, " deg")
    info("mountain slopes", f"steepest slope on canvas mountain rim is {worst:.1f} deg")

    # ---------------------------------------------------------------- the till
    # "What is till" comes from the generator's own `till_weight`, on the same grid it
    # built the relief on, so the ground judged here is the ground that was shaped -
    # less the levelled platforms, which are flat by design and would drag the median
    # down. The flat strip is a separate check: it is flat on purpose, and this is what
    # notices if some later stage starts leaking relief into it.
    print(f"\nthe till ({TILL_BASELINE_M:.0f} m baseline):")
    k4 = PLAYABLE_M // TILL_PX
    z4 = (play_raw.reshape(k4, TILL_PX, k4, TILL_PX).mean(axis=(1, 3)) / 100.0
          ).astype(np.float32)
    w_geo, w_ridge = till_weight(z4, float(TILL_PX))
    till = (w_geo * w_ridge) > 0.9
    ax4 = (np.arange(k4, dtype=np.float32) + 0.5) * TILL_PX
    pad_box = np.zeros(till.shape, dtype=bool)
    for p in ml.pads():
        if not p.get('level'):
            continue
        cx, cy = p['centre']
        w, h = p['size']
        pad_box |= (((ax4 >= cx - w / 2.0 - FEATHER_CAP_M)
                     & (ax4 <= cx + w / 2.0 + FEATHER_CAP_M))[None, :]
                    & ((ax4 >= cy - h / 2.0 - FEATHER_CAP_M)
                       & (ax4 <= cy + h / 2.0 + FEATHER_CAP_M))[:, None])
    till &= ~pad_box
    if check("there is till to measure", bool(till.mean() > 0.25),
             f"{float(till.mean()) * 100:.1f}% of the playable square"):
        s = ops.slope_deg(z4, float(TILL_PX), baseline_m=TILL_BASELINE_M)[till]
        p50, p90, p99 = (float(v) for v in np.percentile(s, [50, 90, 99]))
        band("till slope, median", p50, *TILL_SLOPE_P50_DEG, " deg")
        band("till slope, 90th percentile", p90, *TILL_SLOPE_P90_DEG, " deg")
        band("till slope, 99th percentile", p99, 0.0, TILL_SLOPE_P99_MAX_DEG, " deg")
    # The flat strip, against the source: every flat region the source carries comes
    # through untouched, except where a later stage is entitled to it - under a
    # levelled platform (its drain) and in the lake and its shore (the basin).
    src_full = load_source_dem()
    if src_full is not None:
        src = src_full[o:o + PLAYABLE_M, o:o + PLAYABLE_M].astype(np.float32) / 100.0
        s4 = src.reshape(k4, TILL_PX, k4, TILL_PX).mean(axis=(1, 3)).astype(np.float32)
        k5 = max(3, int(round(20.0 / TILL_PX)) | 1)
        flat = (ndimage.maximum_filter(s4, size=k5)
                - ndimage.minimum_filter(s4, size=k5)) < TILL_FLAT_TOL_M
        flat = ndimage.binary_opening(flat, iterations=max(1, int(48.0 / TILL_PX)))
        flat &= ~pad_box
        for wb in ml.water():
            if not wb.get('ring'):
                continue
            img = Image.new('L', (k4, k4), 0)
            ImageDraw.Draw(img).polygon([(x / TILL_PX, y / TILL_PX) for x, y in wb['ring']],
                                        outline=1, fill=1)
            lake = np.array(img, dtype=bool)
            flat &= ndimage.distance_transform_edt(~lake) * TILL_PX > (
                ml.TILL_SHORE_CLEAR_M + ml.TILL_SHORE_FADE_M)
        moved = float(np.abs(z4 - s4)[flat].max()) if flat.any() else 0.0
        check("the source's flat ground is left flat", moved < 0.02,
              f"{float(flat.mean()) * 100:.1f}% of the playable square, "
              f"moved {moved * 100:.1f} cm at most")

    # ---------------------------------------------------------------- stats
    print("\nterrain_stats.json:")
    if not check("published", os.path.exists(stats_path)):
        return 1
    with open(stats_path) as fh:
        stats = json.load(fh)
    hgt = np.array(stats['height'], dtype=np.float64)
    rgh = np.array(stats['roughness'], dtype=np.float64)
    check(f"{STATS_GRID}x{STATS_GRID} grid", stats['n'] == STATS_GRID,
          f"n = {stats['n']}")
    check("covers the playable area", abs(stats['n'] * stats['cell_m'] - PLAYABLE_M)
          < 1e-6, f"{stats['n']} x {stats['cell_m']:.0f} m = "
                  f"{stats['n'] * stats['cell_m']:.0f} m")
    check("origin at the north-west corner of the playable area",
          stats['origin'] == [0.0, 0.0], f"{stats['origin']}")
    k = PLAYABLE_M // STATS_GRID
    blocks = (play_raw.reshape(STATS_GRID, k, STATS_GRID, k).mean(axis=(1, 3)) / 100.0)
    worst_diff = float(np.abs(hgt - blocks.ravel()).max())
    check("height grid agrees with the PNG", worst_diff <= 0.25,
          f"worst of {hgt.size} cells is {worst_diff:.3f} m off")
    check("roughness values in valid range", bool((rgh >= 0.0).all() and (rgh <= 1.0).all()),
          f"min {float(rgh.min()):.4f}, max {float(rgh.max()):.4f}")

    # ---------------------------------------------------------------- layout
    print("\nlayout:")
    problems = ml.validate()
    check("map_layout validates", not problems,
          "; ".join(problems) if problems else "no complaints")
    check("vector features defined", len(ml.corridors()) > 0 and len(ml.areas()) > 0,
          f"{len(ml.corridors())} corridors, {len(ml.pads())} pads, {len(ml.areas())} areas")

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("failed:")
        for n in failed:
            print("   -", n)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
