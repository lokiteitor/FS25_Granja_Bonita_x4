#!/usr/bin/env python3
"""FS25 heightmap generator for the x4 map.

Builds the 8192x8192 px canvas (1 px = 1 m) with the 4096x4096 m playable square
centred in it, from `input/valle_bonito.png`: the source DEM, already at canvas size,
with the valley wall standing in its 2048 m border and the till plain, the lake basin
and the eastern ridge in its playable square. The source is what the map's ground *is*;
this script is the sculpting that turns it into the map the layout describes, and every
stage takes its geometry from `map_layout.py`, which the OSM generator reads too, so a
feature the vectors draw is a feature the ground was shaped for.

The playable square is worked at 1 m in playable metres, in this order:

1. `roughen_till` - the swell and swale of the till, from the `TILL_*` block in the
   layout, off on the lake, the boundary, the eastern ridge and the deliberately flat
   strip the source carries (detected, not drawn). First, so the platforms and the lake
   come out exactly as they would have on ground that now rolls up to them.
2. `level_platforms` - every pad the layout marks `level` (the towns and the "Granja N"
   yards), levelled to the median of the ground under it with a drain grade and a
   feather that widens with the cut.
3. `sculpt_western_lake` - the basin under the lake ring, bank, shelf and deep bed, and
   the plain restored round it.
4. `extend_east_ridge` - on the whole canvas: the ridge under the eastern wood is run
   across the boundary and into the border range instead of dying in a cliff.
5. The apron: over the `RIM_APRON_M` outside the playable square the border is blended
   down to the playable edge, so the boundary is not a step.

Then `terrain_stats.json` (a 128x128 grid of height and roughness the OSM side reads
through `map_layout.load_roughness`) and two preview PNGs.

Heights are stored as 16-bit centimetres (raw / 100 = metres), matching the rest of the
project and Giants Editor's import convention. The canvas metre of output pixel `i` is
`i + 0.5`; getting that wrong shifts the terrain against the vectors and is invisible
in the image. The primitives - `smootherstep`, `rect_sdf`, `slope_deg`, `value_noise` -
are in `terrain_ops.py`.
"""
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import map_layout as ml                                             # noqa: E402
import terrain_ops as ops                                           # noqa: E402

# --- canvas geometry -------------------------------------------------------------------
CANVAS_M = int(ml.CANVAS_M)
PLAYABLE_M = int(ml.PLAYABLE_M)
OFFSET_M = int(ml.OFFSET_M)
BAND_ROWS = 1024                      # the canvas is worked in eight of these

# --- outputs, named for the canvas -------------------------------------------------------
DEM_NAME = f"dem_{CANVAS_M // 1024}k.png"
VIS_NAME = f"dem_visual_{CANVAS_M // 1024}k.png"
DETAIL_NAME = f"dem_visual_detail_{CANVAS_M // 1024}k.png"
SOURCE_DEM = os.path.join(_ROOT, 'input', 'valle_bonito.png')

# --- datum -----------------------------------------------------------------------------
BASE_ELEV_M = ml.BASE_ELEV_M          # the datum the layout quotes heights against
Z_MAX_CM = 62000.0                    # Giants' working ceiling, in centimetres

# The widest a platform's feather is allowed to grow. It is sized at 1.5*|dz|/tan(4 deg),
# which is right and which runs away where a platform sits deep in ground it disagrees
# with by a lot; this is the belt to that pair of braces.
FEATHER_CAP_M = 120.0
BANK_DEG = 4.0                        # the slope every feather is sized to hold

MASTER_SEED = ml.SEED
# Named streams with fixed, spaced indices: adding one later must not shift the streams
# that already exist, or the whole terrain changes underneath you.
STREAMS = {'till_swell': 30, 'till_swale': 31, 'till_knob': 32,
           'till_warp_x': 33, 'till_warp_y': 34}

STATS_GRID = 128                      # terrain_stats.json resolution
# What counts as fully broken ground, as a gradient. The parcelling reads this to size
# fields, so it has to discriminate across the ground the map actually has: six degrees
# puts the flats near zero, the moraine flanks in the middle and the ridge at the top.
ROUGH_FULL_SCALE = 0.105              # tan(6 deg)


def rng_for(name):
    return np.random.default_rng([MASTER_SEED, STREAMS[name]])


# ==================================================================================
# figures
# ==================================================================================
def style(ax, title):
    ax.set_xlabel("X (East-West) [metres]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y (North-South) [metres]", fontsize=11, fontweight='bold')
    ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5, alpha=0.35)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=14, color='white')


def shade(sub, vmin, vmax):
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.shade(sub, cmap=plt.get_cmap('terrain'), blend_mode='overlay',
                    vert_exag=2.0, vmin=vmin, vmax=vmax)


def draw_layout(ax):
    """The layout on top of the terrain: if the two disagree, it shows here."""
    for c in ml.corridors():
        if c['kind'] in ('track', 'street'):
            continue
        ax.plot([p[0] for p in c['axis']], [p[1] for p in c['axis']],
                color=('#F59E0B' if c['kind'] == 'rail' else '#E5E7EB'),
                lw=(1.6 if c['kind'] == 'rail' else 0.9),
                ls=('--' if c['kind'] == 'rail' else '-'), alpha=0.85)
    for w in ml.water():
        if w.get('ring'):
            ax.fill([p[0] for p in w['ring']], [p[1] for p in w['ring']],
                    color='#0284C7', alpha=0.75)
        elif w.get('axis'):
            ax.plot([p[0] for p in w['axis']], [p[1] for p in w['axis']],
                    color='#38BDF8', lw=1.8)
    for p in ml.pads():
        ax.plot([q[0] for q in p['ring']], [q[1] for q in p['ring']],
                color={'industry': '#6366F1', 'farm': '#22C55E'}.get(
                    p.get('kind'), '#DB2777'), lw=1.2)


def draw_figures(raw, out_vis, out_detail):
    n = CANVAS_M
    k = n // 1024
    vis = raw.reshape(1024, k, 1024, k).mean(axis=(1, 3)) / 100.0
    vmin, vmax = np.percentile(vis, 0.5), np.percentile(vis, 99.5)
    # A flat canvas has no range to stretch a colour map over, and the hillshade would
    # divide by zero. Half a metre either side gives it something to work with and reads
    # as the single flat tone it is.
    if vmax - vmin < 1e-6:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.imshow(shade(vis, vmin, vmax), extent=[0, n, n, 0])
    im = ax.imshow(vis, extent=[0, n, n, 0], cmap='terrain', vmin=vmin, vmax=vmax,
                   alpha=0.0)
    ax.set_xticks(np.arange(0, n + 1, 1024))
    ax.set_yticks(np.arange(0, n + 1, 1024))
    style(ax, f"Full DEM canvas ({n}x{n} px, 1 px = 1 m)")
    ax.add_patch(plt.Rectangle((OFFSET_M, OFFSET_M), PLAYABLE_M, PLAYABLE_M, fill=False,
                               edgecolor='white', linewidth=2, linestyle='--',
                               label=f'Playable border ({PLAYABLE_M / 1000:.1f} km)'))
    ax.legend(loc='upper right', facecolor='black', labelcolor='white', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("height [m]", color='white')
    cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')
    plt.savefig(out_vis, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    p0 = (OFFSET_M * 1024) // n
    p1 = ((OFFSET_M + PLAYABLE_M) * 1024) // n
    sub = vis[p0:p1, p0:p1]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.imshow(shade(sub, vmin, vmax), extent=[0, PLAYABLE_M, PLAYABLE_M, 0])
    im = ax.imshow(sub, extent=[0, PLAYABLE_M, PLAYABLE_M, 0], cmap='terrain',
                   vmin=vmin, vmax=vmax, alpha=0.0)
    if sub.max() - sub.min() > 2.0:
        xs = np.linspace(0, PLAYABLE_M, sub.shape[1])
        ax.contour(xs, xs, sub, levels=np.arange(np.floor(sub.min()), sub.max(), 2.0),
                   colors='white', linewidths=0.4, alpha=0.25)
    draw_layout(ax)
    ax.set_xticks(np.arange(0, PLAYABLE_M + 1, 1024))
    ax.set_yticks(np.arange(0, PLAYABLE_M + 1, 1024))
    style(ax, f"Playable area ({PLAYABLE_M / 1000:.1f} x {PLAYABLE_M / 1000:.1f} km)")
    ax.set_xlim(0, PLAYABLE_M)
    ax.set_ylim(PLAYABLE_M, 0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("height [m]", color='white')
    cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')
    plt.savefig(out_detail, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ==================================================================================
# The ridge in the east of the map dies before it gets out: the crest holds its height
# across the uplands and then collapses over the last two hundred metres of playable
# ground, so the range ends in a cliff at the boundary with the border's own mountains
# standing clear of it. Running it on is one operation and not two - the ground inside the
# map and the ground outside it are the same ridge - so it is done on the whole canvas at
# once, and the apron is blended afterwards against the ground this leaves rather than
# against the ground that was there before.
#
# The construction is a swept cross-section: the last station where the ridge still stands
# is the donor, and its section is carried east, losing EAST_RIDGE_SAG_M to a col on the
# way, until the border range rises over it and a maximum hands the ground back. Nothing
# says where the range ends, which is the point - it ends where it meets the other one.
EAST_RIDGE_KEEP = 0.90       # the donor is the last station standing this high
EAST_RIDGE_SEARCH_M = 770.0  # ... looked for over this much of the ridge's own length:
                             # 1500 survey metres through the x4 conversion's warp
EAST_RIDGE_SAG_M = 25.0      # the col between the two ranges
EAST_RIDGE_SAG_RUN_M = 500.0 # ... reached over this much of the run
EAST_RIDGE_FEATHER_M = 200.0 # the section is carried to nothing over this, either side
EAST_RIDGE_BLUR_M = 25.0     # rounds the break the swept section makes at the donor


def extend_east_ridge(raw):
    """Run the eastern ridge across the boundary and into the border range. In place.

    The band it works in is the wood's own: the timber is what is drawn on this ridge, so
    the ring is the one record of where the ridge is, and taking the band from anywhere
    else would be a second opinion about a thing the layout already states. The section is
    added rather than written - `maximum` against the ground that is there - which is what
    makes the far end need no constant: over the flanks of the border range the range is
    already higher and the operation does nothing at all.
    """
    ring = getattr(ml, 'EAST_RIDGE_RING', None) or next(
        (a['ring'] for a in ml.AREAS if a['id'] == f'wood_{ml.EAST_RIDGE_WAY}'), None)
    if ring is None:
        return raw, None

    x_east = max(p[0] for p in ring)
    tip = [p for p in ring if p[0] > x_east - ml.EAST_WOOD_STRETCH_M]
    y0 = min(p[1] for p in tip) - EAST_RIDGE_FEATHER_M
    y1 = max(p[1] for p in tip) + EAST_RIDGE_FEATHER_M

    r0 = max(0, int(round(y0)) + OFFSET_M)
    r1 = min(raw.shape[0], int(round(y1)) + OFFSET_M)
    c_end = int(round(x_east)) + OFFSET_M
    c_lo = max(0, c_end - int(EAST_RIDGE_SEARCH_M))
    z = raw[r0:r1, c_lo:].astype(np.float32) / 100.0

    # The donor: the easternmost station whose crest still stands at EAST_RIDGE_KEEP of
    # the best the ridge makes over the search. Read off the ground rather than set as a
    # setback from the boundary, because how far in the collapse reaches is a property of
    # the source DEM and not of the map.
    crest = z[:, :c_end - c_lo].max(axis=0)
    j0 = int(np.nonzero(crest >= EAST_RIDGE_KEEP * crest.max())[0].max())

    xs = np.arange(z.shape[1] - j0, dtype=np.float32)
    drop = EAST_RIDGE_SAG_M * ops.smoothstep(xs / EAST_RIDGE_SAG_RUN_M)
    ys = np.arange(r0, r1, dtype=np.float32) - OFFSET_M
    w = ops.smoothstep(np.minimum(ys - y0, y1 - ys) / EAST_RIDGE_FEATHER_M)

    blk = z[:, j0:]
    lift = np.maximum(0.0, z[:, j0][:, None] - drop[None, :] - blk) * w[:, None]
    # The swept section meets the collapsing ridge in a break line at the donor. Blurring
    # the lift rounds it off without touching the ridge itself: the lift is zero over
    # everything the extension does not reach, so what the blur spreads there is zero.
    lift = ndimage.gaussian_filter(lift, EAST_RIDGE_BLUR_M)
    z[:, j0:] = blk + lift
    raw[r0:r1, c_lo:] = np.rint(np.clip(z * 100.0, 0.0, 65535.0)).astype(np.uint16)
    return raw, (c_lo + j0 - OFFSET_M, float(lift.max()))


TILL_PX = 4                  # the till relief is built at this pitch and resampled:
                             # nothing in it is under 110 m of wavelength
TILL_FLAT_TOL_M = 0.005      # "exactly the datum of the flat strip", half a centimetre


def till_weight(z4, dx=float(TILL_PX)):
    """Where the till relief goes, on a `dx` m grid of the playable square.

    Returns `(w_geo, w_ridge)`, both 0..1. `w_geo` is the hard geography - zero on the
    lake and its shore, along the playable boundary and over the deliberately flat strip
    - and `w_ridge` is 1 on the till and 0 on the eastern ridge, read off the ground's own
    height and slope. The measurer takes its "what is till" from this same call, so the
    surface that is judged is the one that was built.

    The flat strip is *detected*, not drawn: it is every region of `TILL_FLAT_MIN_HA` or
    more where the source DEM sits at one height to the half-centimetre. The source is
    the only record of where that strip is, and a rectangle written here would stop
    being right the day it changes. Isolated pixels that happen to land on the same
    value are opened away first, or each would punch a fade-sized hole in the relief.
    """
    sss = ops.smootherstep
    n = z4.shape[0]
    ax = (np.arange(n, dtype=np.float32) + 0.5) * dx
    X, Y = ax[None, :], ax[:, None]

    # The boundary.
    d_edge = np.minimum(np.minimum(X, ml.PLAYABLE_M - X), np.minimum(Y, ml.PLAYABLE_M - Y))
    w = sss(d_edge / ml.TILL_EDGE_FADE_M)

    # The lake and its shore, from the ring the lake is carved from.
    for wb in ml.water():
        if not wb.get('ring'):
            continue
        img = Image.new('L', (n, n), 0)
        ImageDraw.Draw(img).polygon([(x / dx, y / dx) for x, y in wb['ring']],
                                    outline=1, fill=1)
        lake = np.array(img, dtype=bool)
        d_out = ndimage.distance_transform_edt(~lake) * dx
        w = w * sss((d_out - ml.TILL_SHORE_CLEAR_M) / ml.TILL_SHORE_FADE_M)

    # The flat strip: one height, held over a region big enough to be meant.
    k5 = max(3, int(round(20.0 / dx)) | 1)
    flat = (ndimage.maximum_filter(z4, size=k5)
            - ndimage.minimum_filter(z4, size=k5)) < TILL_FLAT_TOL_M
    flat = ndimage.binary_opening(flat, iterations=max(1, int(48.0 / dx)))
    lbl, k = ndimage.label(flat)
    if k:
        sizes = ndimage.sum(flat, lbl, index=np.arange(1, k + 1)) * dx * dx / 1.0e4
        keep = np.isin(lbl, 1 + np.nonzero(sizes >= ml.TILL_FLAT_MIN_HA)[0])
        if keep.any():
            d_flat = ndimage.distance_transform_edt(~keep) * dx
            w = w * sss(d_flat / ml.TILL_FLAT_FADE_M)

    # The ridge, off the ground itself. Blurred so the fade has no contour of its own.
    z0, z1 = ml.TILL_RIDGE_Z_M
    s0, s1 = ml.TILL_RIDGE_SLOPE_DEG
    slope = ops.slope_deg(z4, dx, baseline_m=20.0)
    w_ridge = (1.0 - sss((z4 - z0) / (z1 - z0))) * (1.0 - sss((slope - s0) / (s1 - s0)))
    w_ridge = ndimage.gaussian_filter(w_ridge.astype(np.float32), 40.0 / dx)
    return w.astype(np.float32), w_ridge.astype(np.float32)


def roughen_till(valle_play):
    """Add the till's swell and swale to the playable square. Returns a new array.

    The source DEM has the moraines and nothing much under 400 m; this is the relief
    between 110 and 450 m that a field in the Des Moines lobe actually has - see the
    `TILL_*` block in the layout for the octaves. Built on a `TILL_PX` m grid, weighted
    by `till_weight`, and resampled to 1 m with a cubic kernel.

    It runs before the platforms and before the lake: a platform is levelled over
    whatever ground it stands on, and the lake carves whatever it meets, so both come
    out exactly as they would have without this, on ground that now rolls up to them.
    """
    n1 = valle_play.shape[0]
    n = n1 // TILL_PX
    dx = float(TILL_PX)
    z4 = valle_play.reshape(n, TILL_PX, n, TILL_PX).mean(axis=(1, 3)).astype(np.float32)
    ax = (np.arange(n, dtype=np.float32) + 0.5) * dx
    X, Y = np.meshgrid(ax, ax)
    span = float(ml.PLAYABLE_M)

    wx = ml.TILL_WARP_M * ops.value_noise(X, Y, ml.TILL_WARP_LAM_M,
                                          rng_for('till_warp_x'), span)
    wy = ml.TILL_WARP_M * ops.value_noise(X, Y, ml.TILL_WARP_LAM_M,
                                          rng_for('till_warp_y'), span)
    Xw, Yw = X + wx, Y + wy
    a = math.radians(ml.LAND_MORAINE_GRAIN_DEG)
    c, sn = math.cos(a), math.sin(a)
    u = (Xw * c + Yw * sn) / ml.TILL_SWELL_STRETCH
    v = -Xw * sn + Yw * c
    swell = ml.TILL_SWELL_M * ops.value_noise(u, v, ml.TILL_SWELL_LAM_M,
                                              rng_for('till_swell'), span)
    swale = ml.TILL_SWALE_M * ops.value_noise(Xw, Yw, ml.TILL_SWALE_LAM_M,
                                              rng_for('till_swale'), span)
    knob = ml.TILL_KNOB_M * ops.value_noise(Xw, Yw, ml.TILL_KNOB_LAM_M,
                                            rng_for('till_knob'), span)

    w_geo, w_ridge = till_weight(z4, dx)
    lift = w_geo * (w_ridge * (swell + swale + knob)
                    + (1.0 - w_ridge) * ml.TILL_RIDGE_KEEP * swale)

    # To 1 m, in bands. Grid pixel j is centred at dx*j + dx/2, so output
    # pixel i (centre i + 0.5) sits at grid coordinate (i + 0.5 - dx/2) / dx.
    out = valle_play.copy()
    cols = (np.arange(n1, dtype=np.float32) + 0.5 - dx * 0.5) / dx
    for r0 in range(0, n1, BAND_ROWS):
        r1 = min(n1, r0 + BAND_ROWS)
        rows = (np.arange(r0, r1, dtype=np.float32) + 0.5 - dx * 0.5) / dx
        coords = np.stack(np.broadcast_arrays(rows[:, None], cols[None, :]))
        out[r0:r1] += ndimage.map_coordinates(lift, coords, order=3, mode='nearest',
                                              output=np.float32)
    on = (w_geo * w_ridge) > 0.9
    print(f"   till relief on {float(on.mean()) * 100:.1f}% of the playable square, "
          f"rms {float(np.sqrt((lift[on] ** 2).mean())):.2f} m, "
          f"{float(np.abs(lift).max()):.2f} m at most")
    return out


def level_platforms(valle_play):
    """Level the ground under every platform the layout marks in the replicated DEM.

    The playable area of the output is copied from the input PNG, so the platforms
    `grade_pads` levels in `sculpt()` never reach it - this is the one place a pad
    can be applied and survive. The geometry is still the layout's: the rectangle,
    the feather and the drain grade all come off the `town` pad record, and nothing
    here decides where a town is.

    Three things it shares with `grade_pads`, because they are the same operation:

    * **The target is the median of the ground the platform stands on**, not a
      constant, so the town sits on its own hillside instead of being quoted against
      a datum that is a mean and not a height.
    * **It is not dead flat.** `drain_grade` leaves a third of a percent of fall to
      the south, clamped to the platform's own extent - left to run on, the target
      plane keeps climbing past the edge while the ground under it does whatever it
      does, and the feather sized off `dz` grows with distance instead of settling.
    * **The feather widens with the cut**, `max(nominal, 1.5*|dz|/tan(4 deg))`, because
      in a smoothstep the steepest gradient is `1.5*rise/run` and a constant feather
      cuts a step wherever the platform sits deep.

    The work is done in a window round the platform rather than over the whole playable
    square: `rect_sdf` of a 400 m pad over 67 megapixels is 268 MB of float per
    temporary, and the answer is zero everywhere past the feather cap anyway.
    """
    pads = [p for p in ml.pads() if p.get('level')]
    if not pads:
        return valle_play
    out = valle_play.copy()
    tan_bank = math.tan(math.radians(BANK_DEG))
    n = out.shape[0]
    for p in pads:
        cx, cy = p['centre']
        w, h = p['size']
        pad = FEATHER_CAP_M + 2.0
        x0 = max(0, int(math.floor(cx - w / 2.0 - pad)))
        x1 = min(n, int(math.ceil(cx + w / 2.0 + pad)))
        y0 = max(0, int(math.floor(cy - h / 2.0 - pad)))
        y1 = min(n, int(math.ceil(cy + h / 2.0 + pad)))
        if x1 <= x0 or y1 <= y0:
            continue
        # Playable metres of pixel centres, the frame map_layout works in.
        X = (np.arange(x0, x1, dtype=np.float32) + 0.5)[None, :]
        Y = (np.arange(y0, y1, dtype=np.float32) + 0.5)[:, None]
        z = out[y0:y1, x0:x1]
        d = ops.rect_sdf(X, Y, cx - w / 2.0, cy - h / 2.0,
                         cx + w / 2.0, cy + h / 2.0)
        on = d <= 0.0
        if not bool(on.any()):
            continue
        sy = np.clip(cy - Y, -h / 2.0, h / 2.0)
        target = (float(np.median(z[on])) + p['drain_grade'] * sy).astype(np.float32)
        dz = target - z
        feather = np.clip(1.5 * np.abs(dz) / tan_bank, p['feather_m'], FEATHER_CAP_M)
        out[y0:y1, x0:x1] = z + (1.0 - ops.smoothstep(d / feather)) * dz
        print(f"   {p['name']}: {w:.0f} x {h:.0f} m platform levelled to "
              f"{float(np.median(z[on])):.2f} m, cut/fill "
              f"{float(np.abs(dz[on]).max()):.2f} m at worst")
    return out


def sculpt_western_lake(valle_play):
    """Carves the western mountain lake basin into valle_play.

    The lake sits strictly BELOW the surrounding terrain with no elevated border.
    The outer margin where the old mountain foot was raised above the plain is
    restored to the natural plain level (35.0 - 36.0 m).
    Inside the lake, the bed descends over a bank and littoral shelf to a deep
    natural basin with a maximum depth of up to 40 meters (bed down to 1.5 m).
    """
    water_bodies = [w for w in ml.water() if w.get('ring')]
    if not water_bodies:
        return valle_play

    out = valle_play.copy()
    playable_m = int(ml.PLAYABLE_M)

    for w in water_bodies:
        pts = w['ring']
        mask_img = Image.new('L', (playable_m, playable_m), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(pts, outline=1, fill=1)
        mask = np.array(mask_img, dtype=bool)

        d_in = ndimage.distance_transform_edt(mask)
        d_out = ndimage.distance_transform_edt(~mask)

        # 1. Restore the outside plain: find the clean reference plain height outside the old mountain ramp
        ref_mask = (d_out >= 75) & (d_out <= 85)
        _, (ry, rx) = ndimage.distance_transform_edt(~ref_mask, return_indices=True)
        plain_ref_z = out[ry, rx]

        ramp_width = 80.0
        out_u = np.clip(d_out / ramp_width, 0.0, 1.0)
        out_w = out_u * out_u * (3.0 - 2.0 * out_u)

        outside_restored = np.where(~mask & (d_out < ramp_width),
                                    (1.0 - out_w) * np.minimum(out, plain_ref_z) + out_w * out,
                                    out)

        # 2. Shore bank height from outside_restored
        _, (sy, sx) = ndimage.distance_transform_edt(mask, return_indices=True)
        shore_bank_z = outside_restored[sy, sx]

        water_ws = 35.0

        # Bank slope over 0..20m: from shore_bank_z down to water_ws (35.0m)
        bank_w_m = 20.0
        b_u = np.clip(d_in / bank_w_m, 0.0, 1.0)
        b_w = b_u * b_u * (3.0 - 2.0 * b_u)
        near_shore_z = (1.0 - b_w) * np.maximum(shore_bank_z, water_ws) + b_w * water_ws

        # Bed slope over 20..140m: from water_ws down to deep lake bed
        shelf_m = 120.0
        s_u = np.clip((d_in - bank_w_m) / shelf_m, 0.0, 1.0)
        s_w = s_u * s_u * (3.0 - 2.0 * s_u)

        deep_u = np.clip((d_in - bank_w_m - shelf_m) / (d_in.max() - bank_w_m - shelf_m), 0.0, 1.0)
        deep_w = deep_u * deep_u * (3.0 - 2.0 * deep_u)
        # Deepest bed at 1.5 m (depth up to 40 m from surrounding 41.5m terrain, 33.5 m below water surface)
        target_bed = (water_ws - 20.0) - 13.5 * deep_w

        lake_bed_z = np.where(d_in <= bank_w_m,
                              near_shore_z,
                              (1.0 - s_w) * water_ws + s_w * target_bed)

        out = np.where(mask, lake_bed_z, outside_restored)

    return out


def load_source_dem(path=SOURCE_DEM):
    """The source canvas, or None if it is not there. It has to be the canvas already:
    nothing here resizes it, and a source of another size would put the border in the
    playable square."""
    if not os.path.exists(path):
        return None
    raw = np.array(Image.open(path))
    if raw.shape != (CANVAS_M, CANVAS_M) or raw.dtype != np.uint16:
        raise RuntimeError(f"{path}: {raw.shape[1]}x{raw.shape[0]} {raw.dtype}, not the "
                           f"{CANVAS_M}x{CANVAS_M} uint16 canvas this map is")
    return raw


def blend_apron(raw, valle_play):
    """Over RIM_APRON_M outside the playable square, bring the border down to the
    playable edge on a smoothstep. In place, in bands."""
    ys = np.arange(CANVAS_M)
    xs = np.arange(CANVAS_M)
    dx = np.maximum(0, np.maximum(OFFSET_M - xs, xs - (OFFSET_M + PLAYABLE_M - 1)))
    dy = np.maximum(0, np.maximum(OFFSET_M - ys, ys - (OFFSET_M + PLAYABLE_M - 1)))
    for r0 in range(0, CANVAS_M, BAND_ROWS):
        r1 = r0 + BAND_ROWS
        d_out = np.sqrt(dx[None, :] ** 2 + dy[r0:r1, None] ** 2)
        apron_mask = (d_out > 0) & (d_out < ml.RIM_APRON_M)
        if not apron_mask.any():
            continue
        yc = np.clip(ys[r0:r1, None] - OFFSET_M, 0, PLAYABLE_M - 1)
        xc = np.clip(xs[None, :] - OFFSET_M, 0, PLAYABLE_M - 1)
        yc_grid = np.broadcast_to(yc, (BAND_ROWS, CANVAS_M))
        xc_grid = np.broadcast_to(xc, (BAND_ROWS, CANVAS_M))
        z_edge = valle_play[yc_grid[apron_mask], xc_grid[apron_mask]] * 100.0
        z_rim = raw[r0:r1][apron_mask].astype(np.float32)
        u = d_out[apron_mask] / ml.RIM_APRON_M
        w = u * u * (3.0 - 2.0 * u)
        raw[r0:r1][apron_mask] = np.rint((1.0 - w) * z_edge + w * z_rim).astype(np.uint16)
    return raw


def write_stats(raw, path):
    """`terrain_stats.json`: STATS_GRID x STATS_GRID of mean height and roughness over
    the playable square. Roughness is the slope over a 40 m baseline on a 4 m grid,
    against ROUGH_FULL_SCALE."""
    play = raw[OFFSET_M:OFFSET_M + PLAYABLE_M, OFFSET_M:OFFSET_M + PLAYABLE_M].astype(np.float32) / 100.0
    n4 = PLAYABLE_M // 4
    k_down = PLAYABLE_M // n4
    play_4m = play.reshape(n4, k_down, n4, k_down).mean(axis=(1, 3))
    slope_4m = np.tan(np.radians(ops.slope_deg(play_4m, 4.0, baseline_m=40.0)))
    k_stat = PLAYABLE_M // STATS_GRID
    hgt = play.reshape(STATS_GRID, k_stat, STATS_GRID, k_stat).mean(axis=(1, 3))
    k_stat_4m = n4 // STATS_GRID
    slp = slope_4m.reshape(STATS_GRID, k_stat_4m, STATS_GRID, k_stat_4m).mean(axis=(1, 3))
    rough = np.clip(slp / ROUGH_FULL_SCALE, 0.0, 1.0)
    with open(path, 'w') as fh:
        json.dump({'n': STATS_GRID, 'cell_m': PLAYABLE_M / STATS_GRID,
                   'origin': [0.0, 0.0],
                   'height': [round(float(v), 2) for v in hgt.ravel()],
                   'roughness': [round(float(v), 4) for v in rough.ravel()]}, fh)


def main():
    t_start = time.time()
    print(f"=== FS25 DEM generator ({CANVAS_M}x{CANVAS_M} m canvas, "
          f"{PLAYABLE_M} m playable) ===")
    print("   ", ml.summary())
    problems = ml.validate()
    if problems:
        print("!! layout problems:")
        for p in problems:
            print("   -", p)
        return 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dem = os.path.join(script_dir, DEM_NAME)
    out_stats = os.path.join(script_dir, "terrain_stats.json")
    out_vis = os.path.join(script_dir, VIS_NAME)
    out_detail = os.path.join(script_dir, DETAIL_NAME)

    print(f"1. Reading the source canvas '{SOURCE_DEM}'...")
    raw = load_source_dem()
    if raw is None:
        print(f"!! {SOURCE_DEM} not found: run tools/convert_from_x16.py first")
        return 1
    play = raw[OFFSET_M:OFFSET_M + PLAYABLE_M, OFFSET_M:OFFSET_M + PLAYABLE_M]
    print(f"   canvas {raw.min() / 100.0:.2f} .. {raw.max() / 100.0:.2f} m, "
          f"playable {play.min() / 100.0:.2f} .. {play.max() / 100.0:.2f} m")
    valle_play = play.astype(np.float32) / 100.0

    # The till's own relief, before anything is levelled onto it or carved out of it,
    # so the platforms and the lake end up exactly as they would have on ground that
    # now rolls up to them.
    print("2. Adding the till's swell and swale...")
    valle_play = roughen_till(valle_play)

    # The platforms, before the lake: a water body carves whatever it meets, so where
    # the two ever overlap the basin wins rather than a flat pan over it.
    print(f"3. Levelling {len([p for p in ml.pads() if p.get('level')])} platform(s)...")
    valle_play = level_platforms(valle_play)

    print("4. Sculpting the western lake...")
    valle_play = sculpt_western_lake(valle_play)
    raw[OFFSET_M:OFFSET_M + PLAYABLE_M, OFFSET_M:OFFSET_M + PLAYABLE_M] = \
        np.rint(valle_play * 100.0).astype(np.uint16)

    # The ridge spans the boundary, so it is run after the playable square is in place
    # and before the apron is blended - and the apron's inner value is read back out of
    # `raw` afterwards, because what it has to come down to is the ground that is there
    # and not the copy `valle_play` was before this.
    print("5. Running the eastern ridge into the border...")
    raw, ridge = extend_east_ridge(raw)
    if ridge is not None:
        print(f"   from x = {ridge[0]:.0f} m, {ridge[1]:.0f} m at the deepest")
        valle_play = (raw[OFFSET_M:OFFSET_M + PLAYABLE_M,
                          OFFSET_M:OFFSET_M + PLAYABLE_M].astype(np.float32) / 100.0)

    print(f"6. Blending the border down to the playable edge over {ml.RIM_APRON_M:.0f} m...")
    raw = blend_apron(raw, valle_play)
    Image.fromarray(raw).save(out_dem)

    print("7. Publishing terrain_stats.json...")
    write_stats(raw, out_stats)

    print("8. Figures...")
    draw_figures(raw, out_vis, out_detail)

    lo, hi = raw.min() / 100.0, raw.max() / 100.0
    play = raw[OFFSET_M:OFFSET_M + PLAYABLE_M, OFFSET_M:OFFSET_M + PLAYABLE_M]
    print(f"\n   canvas {lo:.2f} .. {hi:.2f} m, playable "
          f"{play.min() / 100.0:.2f} .. {play.max() / 100.0:.2f} m, "
          f"relief {hi - lo:.2f} m")
    for p in (out_dem, out_stats, out_vis, out_detail):
        print(f"   [+] {p}")
    print(f"   done in {time.time() - t_start:.1f} s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
