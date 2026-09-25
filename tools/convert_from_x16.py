#!/usr/bin/env python3
"""One-off conversion of the 16x survey and source DEM into this repo's x4 inputs.

Reads `custom_osm.osm` (an 8192 m JOSM survey) and `valle_bonito.png` (its 12288 px
source DEM, 16-bit centimetres) from the 16x repository and writes `input/custom_osm.osm`
(4096 m) and `input/valle_bonito.png` (8192 px: 2048 m of border either side of the
4096 m playable square) here. It is the record of how this repo's inputs were obtained;
the pipeline itself never runs it and knows nothing of the 16x map.

What it does, in order:

1. **A separable warp of the survey.** One monotone function per axis, `fx` and `fy`,
   taking 8192 survey metres to 4096 map metres. The local scale is 1.0 over the bands
   spanned by the ways in KEEP_REAL_SIZE_WAYS (the two towns and the three working farms,
   which keep their real dimensions) and over the outer EDGE_KEEP_M of each axis (so the
   distance from a feature to the clean strip is preserved), and a constant `s_min` -
   about a third - everywhere else, with a smootherstep ramp of RAMP_M between the two so
   a diagonal road bends instead of kinking where it enters a band. Separable, so every
   grid road stays straight and nothing crosses that did not cross before.
2. **Every way that is not a field is copied through the warp** with its id and tags,
   so the ids the layout names (`EAST_RIDGE_WAY`, the levelled "Granja N" yards) still
   point at the same things. The maps4fs helper ways (id <= 0) and the ways the 16x
   layout dropped by id are left out.
3. **The fields are reparcelled, not warped.** A warped field is a quarter to a tenth of
   its area; the map wants fewer fields of the same size. The warped field rings are
   rasterised at 1 m, the hedgerow gaps between neighbours are closed, the roads, yards,
   woods, lake and clean strip are cut back out, and every connected block of field
   ground that is left is split into parcels of the median area the survey's own fields
   had there, with a HEDGE_GAP_M gap between them. The parcels are traced back to
   polygons and written as new `landuse=farmland` ways.
4. **The DEM goes through the same warp**, so the ground stays under the lake ring and
   the ridge under its wood: the border keeps its 2048 m width at 1:1 (compressed only
   along the edge, with the playable square it runs beside), the playable square is
   resampled through the inverse of `fx`/`fy`. Then the two repairs the 16x generator
   applied to its source every run - closing the old river's outlet trenches through the
   border, and flattening the corner where the old town and reservoir were - are baked
   in, because both describe this file and not the map.

Heights are not scaled: every slope on the map is steeper than the survey's by the
local compression. numpy, scipy, Pillow and contourpy are fine here; they are not in
`map_layout.py` or `osm_generator/`, which is why the conversion lives in `tools/`.
"""
import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from scipy import ndimage
import contourpy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- the map this repo builds. Asserted against map_layout at the end, once the new
# survey is in place and the module can be imported over it.
PLAYABLE_M = 4096
CANVAS_M = 8192
OFFSET_M = 2048
EDGE_CLEAR_M = 15.0
RIM_APRON_M = 100.0
LAT_CENTER = 43.14569235
LON_CENTER = -95.14507865
M_PER_DEG = 111111.0
M_PER_DEG_LON = M_PER_DEG * math.cos(math.radians(LAT_CENTER))
FIELD_MIN_HA = 3.0
FIELD_MIN_SIDE_M = 100.0

# --- the survey this is converted from.
SRC_PLAYABLE_M = 8192
SRC_CANVAS_M = 12288
SRC_OFFSET_M = 2048

# --- the warp
KEEP_REAL_SIZE_WAYS = {1, 629, 635, 636, 637}   # Villa del Sur, Pueblo del Oeste, Granja 1-3
KEEP_MARGIN_M = 10.0        # a band reaches this far past the way's own bbox
EDGE_KEEP_M = 60.0          # the outer metres of each axis stay 1:1
RAMP_M = 80.0               # smootherstep from s_min up to 1.0 outside every band
WARP_STEP_M = 0.25          # the integral is taken at this pitch, tabulated at 1 m

# --- the fields
HALF_WIDTH_M = {'primary': 4.0, 'secondary': 3.0}   # anything else 2.5, as the layout has it
DEFAULT_HALF_WIDTH_M = 2.5
CLOSE_R_M = 20.0            # bridges gaps up to twice this: hedgerows, and lanes that go
LAKE_MARGIN_M = 40.0        # fields stay this far off the lake ring (its bank ramp)
LANE_KEEP_NEAR_M = 15.0     # a lane this close to a yard, town, wood or the lake serves it
JOIN_M = 5.0                # a road end this close to another road is joined to it
STUB_M = 150.0              # a free-ended lane fragment under this is a stub, and goes
HEDGE_GAP_M = 18.0          # between neighbouring parcels, as the survey drew them
MIN_STRIP_M = 100.0         # a parcel is never cut thinner than this
PARCEL_MAX_X = 1.5          # a piece over this many targets is cut in two
MIN_HOLE_PX = 100           # a smaller hole in a block is filled; a bigger one splits it
SIMPLIFY_M = 1.5            # Douglas-Peucker tolerance on the traced rings
FIELD_ID_BASE = 10000       # new field ways are numbered from here

# The 16x layout dropped these by id; they are not built and are not copied.
SRC_TOWN_RESERVOIR_WAYS = {2, 4, 9, 14, 15, 16, 17, 18, 282, 337, 338}
SRC_DROPPED_WAYS = {292, 296, 300, 303, 306, 309, 312, 339}
SRC_FARMLAND_WAYS = {326}   # a yard the 16x layout worked as a field: a field here too

# --- the source DEM repairs (survey metres and 16x pixel widths, warped where noted)
CLEAN_X0_SRC = 6930         # the old town and reservoir: x >= this ...
CLEAN_Y_FLAT_SRC = 2260     # ... rows below this are the plain continued eastwards ...
CLEAN_Y_END_SRC = 2400      # ... and blend back into the mountain foot by this row
BORDER_TRENCH_M = 3.0
BORDER_SCAN_PX = 4
EAST_RIDGE_SEARCH_SRC_M = 1500.0   # the generator's search, in survey metres: reported


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ==================================================================================
# the survey
# ==================================================================================
def load_survey(path):
    root = ET.parse(path).getroot()
    b = root.find('bounds')
    minlat, minlon = float(b.get('minlat')), float(b.get('minlon'))
    maxlat, maxlon = float(b.get('maxlat')), float(b.get('maxlon'))
    lat_c = (minlat + maxlat) / 2.0
    m_lat = M_PER_DEG
    m_lon = M_PER_DEG * math.cos(math.radians(lat_c))
    nodes = {}
    for n in root.findall('node'):
        lat, lon = float(n.get('lat')), float(n.get('lon'))
        nodes[int(n.get('id'))] = ((lon - minlon) * m_lon, (maxlat - lat) * m_lat)
    ways = []
    for w in root.findall('way'):
        wid = int(w.get('id'))
        tags = {t.get('k'): t.get('v') for t in w.findall('tag')}
        refs = [int(nd.get('ref')) for nd in w.findall('nd')]
        ways.append((wid, tags, refs))
    extent = ((maxlon - minlon) * m_lon, (maxlat - minlat) * m_lat)
    return nodes, ways, extent


def ring_area_ha(pts):
    if len(pts) > 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0 / 1.0e4


# ==================================================================================
# the warp
# ==================================================================================
def build_axis_warp(bands, src_m, dst_m):
    """`(xs, f)` tabulated at 1 m: f[i] is where survey metre i lands on the map."""
    xs = np.arange(0.0, src_m + WARP_STEP_M / 2.0, WARP_STEP_M)
    w = np.zeros_like(xs)
    for a, b in bands:
        inside = (xs >= a) & (xs <= b)
        below = smootherstep((xs - (a - RAMP_M)) / RAMP_M)
        above = smootherstep(((b + RAMP_M) - xs) / RAMP_M)
        w = np.maximum(w, np.where(inside, 1.0, np.minimum(below, above)))
    W = float(np.trapezoid(w, xs))
    s_min = (dst_m - W) / (src_m - W)
    if not 0.0 < s_min < 1.0:
        raise RuntimeError(f"the 1:1 bands cover {W:.0f} m of {src_m} m: nothing left "
                           f"to compress into {dst_m} m")
    s = s_min + (1.0 - s_min) * w
    f = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(xs))])
    f *= dst_m / f[-1]
    step = int(round(1.0 / WARP_STEP_M))
    return xs[::step], f[::step], s_min


def bands_for(nodes, ways, axis):
    """The 1:1 intervals of one axis: the kept ways' extents and the two edges."""
    out = [(0.0, EDGE_KEEP_M), (SRC_PLAYABLE_M - EDGE_KEEP_M, float(SRC_PLAYABLE_M))]
    for wid, _tags, refs in ways:
        if wid not in KEEP_REAL_SIZE_WAYS:
            continue
        vs = [nodes[r][axis] for r in refs]
        out.append((min(vs) - KEEP_MARGIN_M, max(vs) + KEEP_MARGIN_M))
    return out


# ==================================================================================
# the fields
# ==================================================================================
def _simplify(pts, eps):
    """Douglas-Peucker on an open polyline."""
    if len(pts) < 3:
        return list(pts)
    a, b = np.asarray(pts[0]), np.asarray(pts[-1])
    P = np.asarray(pts[1:-1])
    ab = b - a
    L = float(np.hypot(*ab))
    if L < 1e-9:
        d = np.hypot(*(P - a).T)
    else:
        d = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L
    k = int(np.argmax(d))
    if d[k] <= eps:
        return [tuple(pts[0]), tuple(pts[-1])]
    left = _simplify(pts[:k + 2], eps)
    right = _simplify(pts[k + 1:], eps)
    return left[:-1] + right


def simplify_ring(ring, eps):
    """Douglas-Peucker on a closed ring: split at the vertex farthest from the first,
    simplify both halves, and join them back."""
    pts = [tuple(p) for p in ring]
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 4:
        return pts + [pts[0]]
    P = np.asarray(pts)
    k = int(np.argmax(np.hypot(*(P - P[0]).T)))
    a = _simplify(pts[:k + 1], eps)
    b = _simplify(pts[k:] + [pts[0]], eps)
    out = a[:-1] + b
    return out


def _segments_cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def ring_is_simple(ring):
    n = len(ring) - 1
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(ring[i], ring[i + 1], ring[j], ring[j + 1]):
                return False
    return True


def trace_mask(mask, x0, y0):
    """The outline of a boolean sub-array as a closed ring in map metres, simplified.
    Pixel (i, j) of `mask` covers metres [x0+j, x0+j+1) x [y0+i, y0+i+1)."""
    z = np.pad(mask.astype(np.float32), 1)
    lines = contourpy.contour_generator(z=z).lines(0.5)
    closed = [l for l in lines if len(l) > 3 and np.allclose(l[0], l[-1])]
    if not closed:
        return None
    best = max(closed, key=lambda l: abs(ring_area_ha([tuple(p) for p in l])))
    ring = [(float(x) - 1.0 + 0.5 + x0, float(y) - 1.0 + 0.5 + y0) for x, y in best]
    return simplify_ring(ring, SIMPLIFY_M)


def field_ok(ring):
    if ring is None or len(ring) < 5 or not ring_is_simple(ring):
        return False
    if ring_area_ha(ring) < FIELD_MIN_HA:
        return False
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(max(xs) - min(xs), max(ys) - min(ys)) >= FIELD_MIN_SIDE_M


def _erode(mask, r):
    return ndimage.distance_transform_edt(mask) > r


def _split_holes(blocks):
    """Cut every block that has a hole in it, so no ring needs an inner boundary.
    A hole below MIN_HOLE_PX is filled instead. Returns the new field mask."""
    out = blocks.copy()
    for _ in range(20):
        lbl, k = ndimage.label(out)
        cut = 0
        for i, sl in enumerate(ndimage.find_objects(lbl), 1):
            if sl is None:
                continue
            sub = lbl[sl] == i
            holes = ndimage.binary_fill_holes(sub) & ~sub
            if not holes.any():
                continue
            hl, hk = ndimage.label(holes)
            for h, hs in enumerate(ndimage.find_objects(hl), 1):
                hole = hl[hs] == h
                if hole.sum() < MIN_HOLE_PX:
                    out[sl][hs][hole] = True
                    continue
                cy, cx = ndimage.center_of_mass(hole)
                col = sl[1].start + hs[1].start + int(round(cx))
                out[sl[0].start:sl[0].stop, col] = False
                cut += 1
        if cut == 0:
            return out
    raise RuntimeError("blocks with holes did not converge")


def _parcel_block(blk, target_ha):
    """Split one block (bool sub-array) into parcels of about `target_ha`.

    Recursive bisection: a piece over PARCEL_MAX_X times the target is cut in two
    halves of equal area across the longer side of its bbox, and each connected
    component of each half is taken up again on its own - so an L-shaped block round a
    wood is cut where it is wide and not into strips across its arm. A cut that would
    leave a piece under MIN_STRIP_M across is not made. Returns bool sub-arrays."""
    out = []
    stack = [blk]
    while stack:
        piece = stack.pop()
        area_ha = float(piece.sum()) / 1.0e4
        if area_ha < FIELD_MIN_HA:
            continue
        rows = np.flatnonzero(piece.any(axis=1))
        cols = np.flatnonzero(piece.any(axis=0))
        h, w = rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1
        cut = area_ha > PARCEL_MAX_X * target_ha and max(h, w) >= 2 * MIN_STRIP_M
        if cut:
            along_x = w >= h
            prof = piece.sum(axis=0) if along_x else piece.sum(axis=1)
            cum = np.cumsum(prof)
            at = int(np.searchsorted(cum, cum[-1] / 2.0))
            lo = cols[0] if along_x else rows[0]
            hi = cols[-1] if along_x else rows[-1]
            cut = (at - lo) >= MIN_STRIP_M and (hi - at) >= MIN_STRIP_M
        if not cut:
            out.append(piece)
            continue
        a, b = np.zeros_like(piece), np.zeros_like(piece)
        if along_x:
            a[:, :at], b[:, at:] = piece[:, :at], piece[:, at:]
        else:
            a[:at, :], b[at:, :] = piece[:at, :], piece[at:, :]
        for half in (a, b):
            lbl, k = ndimage.label(half)
            for i in range(1, k + 1):
                stack.append(lbl == i)
    return out


def _adjacent_blocks(lbl, roadmap):
    """{road index: set of block labels touching it}, read off the pixels: a road pixel
    with a block pixel beside it."""
    road = roadmap > 0
    out = {}
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(lbl, (dy, dx), axis=(0, 1))
        m = road & (nb > 0)
        pairs = np.unique(roadmap[m].astype(np.int64) * (1 << 32) + nb[m].astype(np.int64))
        for v in pairs.tolist():
            out.setdefault(v >> 32, set()).add(v & 0xFFFFFFFF)
    return out


def _block_targets(lbl, k, idmap, field_ha):
    """The parcel size for every block: the survey's own median field, everywhere. The
    16x map parcelled some of its ground small - ten-acre plots against the towns, thin
    strips along the north edge - and a block that inherited that as its target would
    come out as a grid of small fields again; the brief is fewer fields of the size
    the survey's typical field has."""
    t = max(float(np.median(list(field_ha.values()))), FIELD_MIN_HA)
    return {i: t for i in range(1, k + 1)}


def reparcel(field_rings, field_ha, obstacle_rings, served_rings, lake_rings, roads, note):
    """Field rings (warped, map metres) -> new parcel rings, and the lanes dropped.

    `field_ha` maps the survey's field ids to their 16x area and `field_rings` is
    {id: ring}. `roads` is [(way id, class, half_width_m, axis)]; `obstacle_rings` are
    the woods and every other ring fields keep off, `served_rings` the yards and towns
    among them (what a lane can be there for), `lake_rings` the water (held
    LAKE_MARGIN_M off).

    The compressed lane grid is the thing that decides field size, so it is thinned:
    a tertiary lane that serves nothing but fields is dropped while a block it bounds
    cannot hold one field of the target size, merging that block into the largest
    neighbour across the lane. Primary and secondary roads, and any lane within
    LANE_KEEP_NEAR_M of a yard or town, are never dropped. What is left is tidied: a
    lane fragment under STUB_M with a free end, serving nothing, is a stub of a lane
    that went, and goes too."""
    n = PLAYABLE_M
    # Which survey field is under each pixel, for the size target.
    idimg = Image.new('I', (n, n), 0)
    d = ImageDraw.Draw(idimg)
    for fid, ring in field_rings.items():
        d.polygon(ring, fill=int(fid))
    idmap = np.array(idimg, dtype=np.int32)
    F = idmap > 0

    obs = Image.new('L', (n, n), 0)
    d = ImageDraw.Draw(obs)
    for ring in obstacle_rings:
        d.polygon(ring, fill=1, outline=1)
    obstacle = np.array(obs, dtype=bool)
    lk = Image.new('L', (n, n), 0)
    d = ImageDraw.Draw(lk)
    for ring in lake_rings:
        d.polygon(ring, fill=1, outline=1)
    lake = np.array(lk, dtype=bool)
    m = int(EDGE_CLEAR_M)
    strip = np.zeros((n, n), dtype=bool)
    strip[:m, :] = strip[-m:, :] = True
    strip[:, :m] = strip[:, -m:] = True

    # Every road, by index, and which lanes may go.
    rimg = Image.new('I', (n, n), 0)
    d = ImageDraw.Draw(rimg)
    for i, (_wid, _kind, half_w, axis) in enumerate(roads, 1):
        d.line(axis, fill=i, width=max(1, int(round(2.0 * half_w))), joint='curve')
    roadmap = np.array(rimg, dtype=np.int32)
    sv = Image.new('L', (n, n), 0)
    d = ImageDraw.Draw(sv)
    for ring in served_rings:
        d.polygon(ring, fill=1, outline=1)
    near = ndimage.distance_transform_edt(~np.array(sv, dtype=bool)) <= LANE_KEEP_NEAR_M
    serving = set(np.unique(roadmap[near]).tolist()) - {0}
    candidates = {i for i, (_w, kind, _h, _a) in enumerate(roads, 1)
                  if kind == 'tertiary' and i not in serving}

    t0 = time.time()
    dil = ndimage.distance_transform_edt(~F) <= CLOSE_R_M
    closed = ndimage.distance_transform_edt(dil) > CLOSE_R_M
    base = closed & ~obstacle & ~strip
    base &= ndimage.distance_transform_edt(~lake) > LAKE_MARGIN_M
    note(f"   field ground {F.sum() / 1e4:.0f} ha drawn, {base.sum() / 1e4:.0f} ha "
         f"after closing the gaps and cutting the obstacles out ({time.time() - t0:.1f} s)")

    active = set(range(1, len(roads) + 1))
    dropped = []
    for it in range(40):
        ground = base & ~np.isin(roadmap, list(active))
        lbl, k = ndimage.label(ground)
        area = ndimage.sum(ground, lbl, index=np.arange(1, k + 1)) / 1.0e4
        target = _block_targets(lbl, k, idmap, field_ha)
        adj = _adjacent_blocks(lbl, roadmap)
        by_block = {}
        for r in candidates & active:
            for b in adj.get(r, ()):
                by_block.setdefault(b, set()).add(r)
        pick = set()
        for b, lanes in by_block.items():
            if area[b - 1] >= target[b]:
                continue
            # Into the largest neighbour: the lane whose other side is biggest.
            def other(r):
                return max((area[o - 1] for o in adj[r] if o != b), default=0.0)
            pick.add(max(lanes, key=other))
        if not pick:
            break
        active -= pick
        dropped.extend(sorted(pick))
        stubs = _stubs(roads, active, candidates)
        active -= stubs
        dropped.extend(sorted(stubs))
    ground = base & ~np.isin(roadmap, list(active))
    lbl, k = ndimage.label(ground)
    note(f"   {len(dropped)} lanes dropped in {it + 1} passes, {k} blocks of field ground")

    sizes = ndimage.sum(ground, lbl, index=np.arange(1, k + 1))
    keep = np.isin(lbl, 1 + np.nonzero(sizes >= FIELD_MIN_HA * 1.0e4)[0])
    ground = _split_holes(keep)
    lbl, k = ndimage.label(ground)
    target = _block_targets(lbl, k, idmap, field_ha)

    rings = []
    for i, sl in enumerate(ndimage.find_objects(lbl), 1):
        blk = lbl[sl] == i
        if blk.sum() < FIELD_MIN_HA * 1.0e4:
            continue
        pad = int(HEDGE_GAP_M) + 2
        for strip_mask in _parcel_block(blk, target[i]):
            sub = np.pad(strip_mask, pad)
            core = _erode(sub, HEDGE_GAP_M / 2.0)
            pl, pk = ndimage.label(core)
            for p, ps in enumerate(ndimage.find_objects(pl), 1):
                piece = pl[ps] == p
                if piece.sum() < FIELD_MIN_HA * 1.0e4:
                    continue
                x0 = sl[1].start - pad + ps[1].start
                y0 = sl[0].start - pad + ps[0].start
                ring = trace_mask(piece, x0, y0)
                if field_ok(ring):
                    rings.append(ring)
    # Numbered by rows of centroids, so the names are stable and readable on the map.
    def centroid(r):
        return (sum(p[0] for p in r[:-1]) / (len(r) - 1), sum(p[1] for p in r[:-1]) / (len(r) - 1))
    rings.sort(key=lambda r: (round(centroid(r)[1] / 400.0), centroid(r)[0]))
    return rings, sorted(set(target.values())), [roads[i - 1][0] for i in dropped]


def _stubs(roads, active, candidates):
    """Lane fragments left dangling by a dropped lane: short, one end touching no other
    active road, and serving nothing. Iterated, because removing one can free the next."""
    out = set()
    while True:
        alive = [i for i in active if i not in out]
        axes = {i: roads[i - 1][3] for i in alive}
        found = set()
        for i in alive:
            if i not in candidates:
                continue
            axis = axes[i]
            if _length(axis) >= STUB_M:
                continue
            for end in (axis[0], axis[-1]):
                touched = any(j != i and any(_seg_dist(end, a, b) <= JOIN_M
                                             for a, b in zip(axes[j], axes[j][1:]))
                              for j in alive)
                if not touched:
                    found.add(i)
                    break
        if not found:
            return out
        out |= found


def _length(axis):
    return sum(math.dist(a, b) for a, b in zip(axis, axis[1:]))


def _road_components(roads):
    """Connected groups of roads, by geometry: an end of one within JOIN_M of another."""
    def near(p, axis):
        return any(_seg_dist(p, a, b) <= JOIN_M for a, b in zip(axis, axis[1:]))
    parent = list(range(len(roads)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (_w, _k, _h, ai) in enumerate(roads):
        for j, (_w2, _k2, _h2, aj) in enumerate(roads):
            if i < j and (near(ai[0], aj) or near(ai[-1], aj) or near(aj[0], ai) or near(aj[-1], ai)):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(roads)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0.0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


# ==================================================================================
# the source DEM
# ==================================================================================
def warp_dem(src, fx, fy):
    """The 16x canvas through the warp: borders 1:1 across, playable through the
    inverse of the survey warp, bilinear, one axis at a time."""
    n_src = src.shape[0]
    if src.shape != (SRC_CANVAS_M, SRC_CANVAS_M) or src.dtype != np.uint16:
        raise RuntimeError(f"source DEM is {src.shape} {src.dtype}, not "
                           f"{SRC_CANVAS_M}x{SRC_CANVAS_M} uint16")
    axis1 = np.arange(len(fx), dtype=np.float64)

    def src_coord(f):
        u = np.arange(CANVAS_M, dtype=np.float64) + 0.5
        inv = SRC_OFFSET_M + np.interp(u - OFFSET_M, f, axis1)
        s = np.where(u < OFFSET_M, u,
                     np.where(u < OFFSET_M + PLAYABLE_M, inv, u + (SRC_CANVAS_M - CANVAS_M)))
        s = np.clip(s - 0.5, 0.0, n_src - 1.0)
        i0 = np.floor(s).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_src - 1)
        return i0, i1, (s - i0).astype(np.float32)

    r0, r1, wr = src_coord(fy)
    tmp = np.empty((CANVAS_M, n_src), dtype=np.float32)
    for a in range(0, CANVAS_M, 512):
        b = min(CANVAS_M, a + 512)
        tmp[a:b] = (src[r0[a:b]].astype(np.float32) * (1.0 - wr[a:b, None])
                    + src[r1[a:b]].astype(np.float32) * wr[a:b, None])
    c0, c1, wc = src_coord(fx)
    out = np.empty((CANVAS_M, CANVAS_M), dtype=np.float32)
    for a in range(0, CANVAS_M, 512):
        b = min(CANVAS_M, a + 512)
        out[a:b] = tmp[a:b][:, c0] * (1.0 - wc[None, :]) + tmp[a:b][:, c1] * wc[None, :]
    del tmp
    return np.rint(np.clip(out, 0.0, 65535.0)).astype(np.uint16)


def _inpaint(sub, m):
    _, idx = ndimage.distance_transform_edt(m, return_indices=True)
    sub[m] = sub[idx[0][m], idx[1][m]]
    for sigma in (16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0):
        for _ in range(10):
            sub[m] = ndimage.gaussian_filter(sub, sigma)[m]
    return sub


def fill_border_trenches(raw, bridge_m, grow_m):
    """Close every channel cut clean through the non-playable border (the old river's
    outlets). Copied from the 16x generator; a trench is what a grey closing `bridge_m`
    wide fills by more than BORDER_TRENCH_M, and a breach is a trench that reaches both
    the canvas edge and the apron. In place; returns the number of breaches closed."""
    k = BORDER_SCAN_PX
    n = raw.shape[0]
    m = n // k
    z = (raw.reshape(m, k, m, k).mean(axis=(1, 3)) / 100.0).astype(np.float32)
    c = np.arange(m, dtype=np.float32) * k + 0.5 * k
    dc = np.maximum(0.0, np.maximum(OFFSET_M - c, c - (OFFSET_M + PLAYABLE_M)))
    d_out = np.hypot(dc[None, :], dc[:, None])
    w = max(3, int(round(bridge_m / k)) | 1)
    deficit = ndimage.grey_closing(z, size=(w, w), mode='nearest') - z
    trench = (deficit > BORDER_TRENCH_M) & (d_out >= RIM_APRON_M)
    lbl, _ = ndimage.label(trench)
    at_edge = np.zeros(trench.shape, dtype=bool)
    at_edge[0] = at_edge[-1] = True
    at_edge[:, 0] = at_edge[:, -1] = True
    at_apron = d_out < RIM_APRON_M + k
    keep = sorted((set(np.unique(lbl[at_edge & trench])) - {0})
                  & (set(np.unique(lbl[at_apron & trench])) - {0}))
    if not keep:
        return 0
    mask = ndimage.distance_transform_edt(~np.isin(lbl, keep)) <= grow_m / k
    mask &= d_out > 0.0
    delta = np.zeros_like(z)
    parts, _ = ndimage.label(mask)
    for sy, sx in ndimage.find_objects(parts):
        pad = 2 * w
        y0, y1 = max(0, sy.start - pad), min(m, sy.stop + pad)
        x0, x1 = max(0, sx.start - pad), min(m, sx.stop + pad)
        sub, mm = z[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
        delta[y0:y1, x0:x1] = _inpaint(sub, mm) - z[y0:y1, x0:x1]
    for sy, sx in ndimage.find_objects(parts):
        y0, y1 = max(0, sy.start - 1) * k, min(m, sy.stop + 1) * k
        x0, x1 = max(0, sx.start - 1) * k, min(m, sx.stop + 1) * k
        rr = (np.arange(y0, y1, dtype=np.float32) + 0.5) / k - 0.5
        cc = (np.arange(x0, x1, dtype=np.float32) + 0.5) / k - 0.5
        d = ndimage.map_coordinates(
            delta, np.meshgrid(rr, cc, indexing='ij'), order=1, mode='nearest')
        box = raw[y0:y1, x0:x1].astype(np.float32) + d * 100.0
        raw[y0:y1, x0:x1] = np.rint(np.clip(box, 0.0, 65535.0)).astype(np.uint16)
    return len(keep)


def clean_town_and_reservoir_area(play, x0, y_flat, y_end):
    """Flatten the corner the old town and reservoir were cut into: the plain at
    column `x0` is continued east over rows < y_flat, then blended back into the
    mountain foot by row y_end. Copied from the 16x generator, on a float32 playable
    crop in metres."""
    out = play.copy()
    ref_col = play[0:y_end, x0].copy()
    out[0:y_flat, x0:] = ref_col[0:y_flat, None]
    for y in range(y_flat, y_end):
        t = (y - y_flat) / (y_end - y_flat)
        w = t * t * (3.0 - 2.0 * t)
        orig = np.maximum(ref_col[y], play[y, x0:])
        out[y, x0:] = (1.0 - w) * ref_col[y] + w * orig
    return out


# ==================================================================================
# writing the survey
# ==================================================================================
def local_to_global(x, y):
    half = PLAYABLE_M / 2.0
    return (LAT_CENTER - (y - half) / M_PER_DEG, LON_CENTER + (x - half) / M_PER_DEG_LON)


def write_survey(path, ways):
    """`ways` is [(id, tags, [(x, y), ...], shared_key_list)] in map metres. Nodes are
    shared between ways through `shared_key_list`: a key that is not None names a node
    reused wherever the same key appears (the survey's road junctions); None makes a
    fresh node. A ring closes on its own first node when its last point repeats it."""
    node_ids = {}
    node_xy = []

    def node_for(key, x, y):
        if key is not None and key in node_ids:
            return node_ids[key]
        nid = len(node_xy) + 1
        node_xy.append((x, y))
        if key is not None:
            node_ids[key] = nid
        return nid

    way_refs = []
    for wid, tags, pts, keys in ways:
        refs = []
        for i, ((x, y), key) in enumerate(zip(pts, keys)):
            if i == len(pts) - 1 and len(pts) > 2 and pts[0] == pts[-1]:
                refs.append(refs[0])
            else:
                refs.append(node_for(key, x, y))
        way_refs.append((wid, tags, refs))

    minlat, minlon = local_to_global(0.0, PLAYABLE_M)
    maxlat, maxlon = local_to_global(PLAYABLE_M, 0.0)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("<?xml version='1.0' encoding='utf-8'?>\n")
        fh.write('<osm version="0.6" generator="convert_from_x16">\n')
        fh.write(f'  <bounds minlat="{minlat:.10f}" minlon="{minlon:.10f}" '
                 f'maxlat="{maxlat:.10f}" maxlon="{maxlon:.10f}" origin="convert_from_x16" />\n')
        for i, (x, y) in enumerate(node_xy, 1):
            lat, lon = local_to_global(x, y)
            fh.write(f'  <node id="{i}" visible="true" version="1" '
                     f'lat="{lat:.10f}" lon="{lon:.10f}" />\n')
        for wid, tags, refs in way_refs:
            fh.write(f'  <way id="{wid}" visible="true" version="1">\n')
            for r in refs:
                fh.write(f'    <nd ref="{r}" />\n')
            for k, v in tags.items():
                fh.write(f'    <tag k="{_esc(k)}" v="{_esc(v)}" />\n')
            fh.write('  </way>\n')
        fh.write('</osm>\n')
    return len(node_xy), len(way_refs)


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('"', '&quot;')
            .replace('<', '&lt;').replace('>', '&gt;'))


# ==================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--from', dest='src', default=os.path.join(_ROOT, '..', 'FS25_Granja_Bonita_x16', 'input'),
                    help="directory holding the 16x custom_osm.osm and valle_bonito.png")
    ap.add_argument('--out', default=os.path.join(_ROOT, 'input'))
    ap.add_argument('--bridge-m', type=float, default=70.0,
                    help="widest border trench the repair bridges (141 m at 16x)")
    ap.add_argument('--grow-m', type=float, default=40.0,
                    help="how far the trench fill reaches onto its flanks (80 m at 16x)")
    ap.add_argument('--skip-dem', action='store_true')
    args = ap.parse_args()
    t_start = time.time()

    src_osm = os.path.join(args.src, 'custom_osm.osm')
    src_dem = os.path.join(args.src, 'valle_bonito.png')
    out_osm = os.path.join(args.out, 'custom_osm.osm')
    out_dem = os.path.join(args.out, 'valle_bonito.png')
    for p in (src_osm, src_dem):
        if not os.path.exists(p):
            print(f"!! {p} not found")
            return 1

    print(f"=== 16x -> x4: {SRC_PLAYABLE_M} m survey to {PLAYABLE_M} m, "
          f"{SRC_CANVAS_M} px DEM to {CANVAS_M} px ===")
    nodes, ways, extent = load_survey(src_osm)
    if max(abs(extent[0] - SRC_PLAYABLE_M), abs(extent[1] - SRC_PLAYABLE_M)) > 0.5:
        print(f"!! survey bounds span {extent[0]:.1f} x {extent[1]:.1f} m, not {SRC_PLAYABLE_M}")
        return 1
    ways = [w for w in ways if w[0] > 0 and w[0] not in SRC_TOWN_RESERVOIR_WAYS
            and w[0] not in SRC_DROPPED_WAYS]

    # 1. The warp.
    xs, fx, sx = build_axis_warp(bands_for(nodes, ways, 0), SRC_PLAYABLE_M, PLAYABLE_M)
    _, fy, sy = build_axis_warp(bands_for(nodes, ways, 1), SRC_PLAYABLE_M, PLAYABLE_M)
    print(f"1. Warp: {len(KEEP_REAL_SIZE_WAYS)} ways kept at real size, "
          f"{sx:.3f} x and {sy:.3f} y elsewhere")

    def warp(p):
        return (float(np.interp(p[0], xs, fx)), float(np.interp(p[1], xs, fy)))

    # 2. Copy everything that is not a field; collect what is.
    copied = []
    obstacle_rings = []
    lake_rings = []
    served_rings = []
    roads = []
    field_rings = {}
    field_ha = {}
    for wid, tags, refs in ways:
        pts = [warp(nodes[r]) for r in refs]
        if tags.get('landuse') == 'farmland' or wid in SRC_FARMLAND_WAYS:
            field_rings[wid] = pts
            field_ha[wid] = ring_area_ha([nodes[r] for r in refs])
            continue
        if 'highway' in tags:
            roads.append((wid, tags['highway'], HALF_WIDTH_M.get(tags['highway'], DEFAULT_HALF_WIDTH_M), pts))
        elif tags.get('natural') == 'water':
            lake_rings.append(pts)
        else:
            obstacle_rings.append(pts)
            if tags.get('natural') != 'wood':
                served_rings.append(pts)
        copied.append((wid, tags, pts, [('n', r) for r in refs]))
    print(f"2. {len(copied)} ways copied through the warp ({len(roads)} roads), "
          f"{len(field_rings)} survey fields ({sum(field_ha.values()):.0f} ha) to reparcel")

    # 3. The fields.
    print("3. Reparcelling...")
    rings, targets, gone = reparcel(field_rings, field_ha, obstacle_rings, served_rings,
                                    lake_rings, roads, print)
    # A lane that is gone from the ground is gone from the map, and so is any road that
    # only ever reached the network through one.
    roads = [r for r in roads if r[0] not in gone]
    for group in _road_components(roads):
        if not any(roads[i][1] in ('primary', 'secondary') for i in group):
            gone.extend(roads[i][0] for i in group)
    roads = [r for r in roads if r[0] not in gone]
    copied = [w for w in copied if w[0] not in gone]
    print(f"   {len(gone)} roads dropped: {' '.join(str(g) for g in sorted(gone))}")
    print(f"   {len(roads)} roads kept")
    areas = sorted(ring_area_ha(r) for r in rings)
    for i, ring in enumerate(rings, 1):
        ha = ring_area_ha(ring)
        copied.append((FIELD_ID_BASE + i, {'landuse': 'farmland', 'name': f'Campo {i} ({ha:.1f} ha)'},
                       ring, [None] * len(ring)))
    print(f"   {len(rings)} fields, {sum(areas):.0f} ha: min {areas[0]:.1f}, median "
          f"{areas[len(areas) // 2]:.1f}, mean {sum(areas) / len(areas):.1f}, max {areas[-1]:.1f} ha"
          f" (target {targets[0]:.1f} ha, the survey's median field)")
    n_nodes, n_ways = write_survey(out_osm, copied)
    print(f"   [+] {out_osm}: {n_nodes} nodes, {n_ways} ways")

    # 4. The DEM.
    if not args.skip_dem:
        print("4. DEM through the same warp...")
        raw = warp_dem(np.array(Image.open(src_dem)), fx, fy)
        n_breach = fill_border_trenches(raw, args.bridge_m, args.grow_m)
        print(f"   {n_breach} breach(es) closed through the border "
              f"(bridge {args.bridge_m:.0f} m, grow {args.grow_m:.0f} m)")
        o = OFFSET_M
        cx0 = int(round(np.interp(CLEAN_X0_SRC, xs, fx)))
        cy0 = int(round(np.interp(CLEAN_Y_FLAT_SRC, xs, fy)))
        cy1 = int(round(np.interp(CLEAN_Y_END_SRC, xs, fy)))
        play = raw[o:o + PLAYABLE_M, o:o + PLAYABLE_M].astype(np.float32) / 100.0
        play = clean_town_and_reservoir_area(play, cx0, cy0, cy1)
        raw[o:o + PLAYABLE_M, o:o + PLAYABLE_M] = np.rint(play * 100.0).astype(np.uint16)
        print(f"   old town and reservoir cleaned: x >= {cx0} m, rows < {cy0} flat, blended by {cy1}")
        Image.fromarray(raw).save(out_dem)
        print(f"   [+] {out_dem}: {raw.shape[1]}x{raw.shape[0]} px, "
              f"{raw.min() / 100:.2f} .. {raw.max() / 100:.2f} m")

    search = float(np.interp(SRC_PLAYABLE_M, xs, fx) - np.interp(SRC_PLAYABLE_M - EAST_RIDGE_SEARCH_SRC_M, xs, fx))
    print(f"   EAST_RIDGE_SEARCH_M for the generator: {search:.0f} m "
          f"(was {EAST_RIDGE_SEARCH_SRC_M:.0f} survey metres)")

    # 5. Read the survey back through the layout module and let it judge.
    print("5. map_layout over the new survey...")
    sys.path.insert(0, _ROOT)
    import map_layout as ml
    for name, mine in (('PLAYABLE_M', PLAYABLE_M), ('CANVAS_M', CANVAS_M), ('OFFSET_M', OFFSET_M),
                       ('EDGE_CLEAR_M', EDGE_CLEAR_M), ('RIM_APRON_M', RIM_APRON_M),
                       ('LAT_CENTER', LAT_CENTER), ('LON_CENTER', LON_CENTER),
                       ('M_PER_DEG', M_PER_DEG), ('FIELD_MIN_HA', FIELD_MIN_HA),
                       ('FIELD_MIN_SIDE_M', FIELD_MIN_SIDE_M)):
        if abs(getattr(ml, name) - mine) > 1e-9:
            print(f"!! {name}: this script has {mine}, map_layout has {getattr(ml, name)}")
            return 1
    print("   ", ml.summary())
    problems = ml.validate()
    for p in problems:
        print("   -", p)
    print(f"   {'layout is sound' if not problems else str(len(problems)) + ' problem(s)'}"
          f", done in {time.time() - t_start:.1f} s")
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
