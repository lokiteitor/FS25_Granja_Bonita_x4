#!/usr/bin/env python3
"""The layout of the map: where everything is, in playable metres.

This is the one source of truth shared by the two halves of the pipeline. The DEM
generator sculpts terrain around the geometry defined here; the OSM generator writes the
same geometry out as vectors. If the two disagree - a river carved where no river is
drawn, a farm pad flattened where no farmyard exists - the map is broken in a way that is
invisible in either output on its own, so both read their geometry from this module and
neither is allowed to invent its own.

**What is on the map so far.** The uplands are a till plain after the country round
Royal in Clay County, northwest Iowa. A river runs the length of it from north to south
and widens into a lake in the northern quarter, with a wooded island in it; both sit in a
valley of their own cut into the floodplain. The roads are the Public Land Survey System
- two trunk roads a mile in from the east and west edges, five section lines on the mile
grid between them, of which only the outer two are bridged. And on the four corners where
a trunk road meets a bridged section line there is a town: a grid of blocks with the
trunk road running up the middle of it, standing on a platform levelled out of the till
plain. Hung off the roads are twelve industrial aprons and six farms, all square; five
woods and eleven shelterbelts stand along them, and hardwood covers the valley side down
both banks of the river and all round the lake, coming to fifteen metres of the water on
one side and stopping fifteen metres short of the fields on the other. The rest is parcelled
into 145 fields on
the survey's own aliquot grid - quarter sections out in the country, forties nearer the
towns, ten-acre parcels against them, merged where they share a whole edge and held 5 m
off each other after that - stopping at the rim of the water's valley, which nothing is
grown in. That is 3252 ha, 49% of the playable square, and the 145 is a ceiling somebody
set rather than a number that fell out: the bands are as wide as 150 fields will pay
for.

**What the ground already is** is the frame all of that goes in: the border around the
playable square is the wall of a valley. Two mountain ranges stand in the east and west
border and rise to `RIM_CREST_M`; the valley runs north to south between them and leaves
the map over a low sill at either end. That is a parametric shape rather than a registry
entry - the constants are in the rim section below, the one implementation of the shape
is `terrain_ops.rim_field`, and no vectors are drawn for it because none of it is
playable ground.

To add to the map, fill the registries (`CORRIDORS`, `WATER`, `PADS`, `AREAS`) and add
the rules that have to hold about them to `validate()`. Both generators refuse to run
while `validate()` complains, which is what keeps a placement mistake from reaching the
editor. The record each registry holds is documented above it.

Coordinates are playable metres: x east, y south from the north edge, so the centre of
the playable area is (2048, 2048). The DEM canvas is larger than the playable area, so
canvas coordinates run from -2048 to 6144 in the same frame. The geometry is read from
`input/custom_osm.osm`, a 4096 m JOSM survey converted once from the 16x map by
`tools/convert_from_x16.py` (see its docstring for how); the projection that ties the
survey to lat/lon is the one the survey's `<bounds>` declare, and the constants below
match it.

Standard library only. The scripts in `osm_generator/` run without numpy, and the DEM
generator needs the same numbers, so nothing here may depend on anything else. Keep the
alignments free of floating-point randomness too: a river meander that comes out of a
closed formula gives both halves of the pipeline the identical polyline, and one that
comes out of an RNG does not.
"""
import bisect
import json
import math
import os
import random
import re

# --- where the map is -------------------------------------------------------------
# The projection anchor matching the input bounds.
LAT_CENTER = 43.14569235
LON_CENTER = -95.14507865

# --- how big it is ----------------------------------------------------------------
PLAYABLE_M = 4096.0
HALF_M = PLAYABLE_M / 2.0
CANVAS_M = 8192.0
OFFSET_M = (CANVAS_M - PLAYABLE_M) / 2.0      # 2048 m of margin on every side

# How far past the canvas the road, rail and river alignments run. Terrain features that
# stop at the canvas edge leave a valley dying in mid-air or a road ending at a cliff.
EXTEND_M = 300.0
EDGE_MIN = -OFFSET_M - EXTEND_M               # -2348
EDGE_MAX = PLAYABLE_M + OFFSET_M + EXTEND_M   # 6444

# A clean strip inside the playable boundary: nothing planted or parcelled stands in it.
# Everything the OSM draws as a ring - field, wood, yard, town block - is clipped back to
# it by `generate_osm.strip_ring`, and `check_osm` fails the build if any of them reaches
# in. Clipped and not clamped: clamping folds whatever hangs over onto the boundary
# itself, which is how a strip of timber once ended up with a run of nodes lying across
# the river channel.
#
# The roads and the water are the two exemptions, and neither is a loophole: a road that
# stopped 15 m short of the edge would end in mid-air, and a river has to leave the map.
EDGE_CLEAR_M = 15.0

# --- the datum --------------------------------------------------------------------
BASE_ELEV_M = 46.4

# --- the non-playable rim ----------------------------------------------------------
# What the player sees past the boundary: the map is the floor of a valley running north
# to south between two mountain ranges. The ranges stand in the east and west border,
# and the valley leaves the map over a low sill at the north and the south, so the
# horizon is closed on two sides and open on the other two. The first RIM_APRON_M past
# the boundary is left flat, so the ground does not change character at the edge of play
# and the playable square keeps its own relief right up to the boundary.
#
# These are **absolute** elevations, not lifts. The valley floor is BASE_ELEV_M and the
# summits are RIM_CREST_M, so every pixel of the canvas lives between 20 m and 250 m.
#
# Turning the valley through ninety degrees is one change: swap dx for dy in the range
# weight `w` in `terrain_ops.rim_field`. Nothing else here knows which way it runs.
RIM_APRON_M = 100.0           # flat apron between the boundary and the first slope
                              # - one EDGE_CLEAR_M, so the clean strip inside the
                              # boundary and the apron outside it are the same
                              # width and the mountains start as soon as the
                              # ground stops being ground the player works
RIM_BACK_M = 150.0            # crest shoulder held at full height inside the canvas edge
RIM_CREST_M = 250.0           # the highest summits
RIM_SADDLE_M = 200.0          # the lowest saddle along a range crest
RIM_MOUTH_M = 115.0           # the sill the valley leaves the map over, north and south
RIM_MOUTH_VAR_M = 8.0         # how far the sill rolls either side of that
RIM_RIDGE_LOBES = 13          # summits around the whole rim - see below, must be whole
RIM_RIDGE_BEAT = 23           # a second, coprime, so no two summits come out alike
RIM_SPUR_LAM_M = 1100.0       # spurs and re-entrants down the flank
RIM_SPUR_AMP = 0.30           # ... as a fraction of the saddle-to-crest band
RIM_ROUGH_LAM_M = 520.0       # the coarsest thing on the flank below a spur
RIM_ROUGH_AMP = 0.10

# How much of that relief the *west* range keeps. The two ranges are not asked to read
# alike: the west one is the one the morning light rakes across, and at full amplitude
# its crest wanders the whole 50 m between saddle and summit and the flank comes down as
# a row of scallops rather than a wall. Damping pulls it toward the middle of the band
# without narrowing the band itself, so the east range still carries the summits. 1.0 is
# the two sides identical.
RIM_WEST_SMOOTH = 0.40

# Derived. The flank climbs the whole crest over RIM_RAMP_M: the apron eats the first
# stretch of the border and the shoulder the last, and what is left is the slope itself.
RIM_HEIGHT_M = RIM_CREST_M - BASE_ELEV_M      # highest summit, above the boundary ground
RIM_RAMP_M = OFFSET_M - RIM_APRON_M - RIM_BACK_M

# The ridge line is a profile around the perimeter of the playable square, so it has to
# come back to where it started: a whole number of lobes around the ring, or the west
# edge arrives at the north-west corner half a summit away from where the north edge
# leaves it and there is a step in the crest. That is why the lobe counts are integers
# and the wavelengths are what falls out of them, rather than the other way round.
RIM_PERIM_M = 4.0 * PLAYABLE_M
RIM_RIDGE_LAM_M = RIM_PERIM_M / RIM_RIDGE_LOBES

# The ceiling the flank is built to. In a smoothstep the steepest gradient is
# 1.5 * rise / run, the same identity that sets every platform feather, so the ramp's
# run is what keeps the mountains walkable rather than a wall. `validate()` enforces it.
RIM_MAX_FLANK_DEG = 18.0

# --- how the two relate -----------------------------------------------------------
# Equirectangular about the centre. 111111.0 m per degree is the constant the rest of the
# pipeline was built with (1e7 m from the equator to the pole, over 90 degrees).
M_PER_DEG = 111111.0
M_PER_DEG_LON = M_PER_DEG * math.cos(math.radians(LAT_CENTER))

SEED = 20250902

# --- the till plain ---------------------------------------------------------------
# The uplands, after the country round Royal in Clay County, northwest Iowa. That is
# Des Moines Lobe ground: the last ice sheet pulled off it about fourteen thousand years
# ago and left a young, barely drained till plain behind. What that looks like is three
# things at once, and all three are here:
#
#   * swell and swale - a gentle, aimless undulation a few metres deep, the surface of
#     the till itself;
#   * low recessional moraines - broader rises the ice left where its edge stalled,
#     running in arcs, which is why this octave is stretched along a grain rather than
#     isotropic like the other two;
#
# The third thing the real place has - prairie potholes, the closed depressions the
# retreating ice left where blocks of it were buried in the till - is deliberately not
# here. They are the signature of the landscape and they read as craters at any useful
# vertical exaggeration, which is not what this map is for.
#
# Total relief is about 18 m across the 8 km of the map, which is the order of what the
# real place does. It has to stay well over the bank top or the river would come out of
# its valley and flood the low ground; `validate()` bounds that analytically.
LAND_MORAINE_M, LAND_MORAINE_LAM_M = 3.6, 3000.0
LAND_MORAINE_GRAIN_DEG = -35.0    # the lobe's ridges trend roughly northwest-southeast
LAND_MORAINE_STRETCH = 2.6        # ... and are that many times longer than they are wide
LAND_SWELL_M, LAND_SWELL_LAM_M = 1.9, 900.0
LAND_SWALE_M, LAND_SWALE_LAM_M = 0.8, 380.0
LAND_WARP_M, LAND_WARP_LAM_M = 130.0, 1700.0     # domain warp, so nothing reads as a sine

# --- the till's own swell and swale -----------------------------------------------
# The source DEM (`input/valle_bonito.png`) carries the long wave of the till plain - the
# moraines at a kilometre and more - but almost nothing between 100 and 400 m, which is
# the swell-and-swale a field in Clay County actually rises and falls over: a metre or
# two every few hundred metres, slopes of one to three degrees. The DEM generator adds
# these three octaves and a warp to the playable square after the copy (`roughen_till`),
# and nowhere else - the flat strip along the north edge is flat on purpose and is left
# so, the lake and its shore are left for the lake to carve, the eastern ridge keeps its
# own shape, and the ground under a levelled platform is levelled again afterwards.
#
# Amplitudes are rms in metres (`value_noise` is unit-variance). `*_LAM_M` is the pitch
# of the noise lattice, and the relief it makes is mostly two to four times longer than
# that - the 55 m knob lattice puts nothing under ~110 m, which is the generator's own
# floor for what the 4 m to 1 m resample can carry without ringing. Measured on the
# till this gives a median slope near 0.9 deg and a 90th percentile near 1.8 deg over a
# 20 m baseline, against 0.3 and 0.6 in the source.
TILL_SWELL_M, TILL_SWELL_LAM_M = 1.2, 220.0
TILL_SWELL_STRETCH = 1.5          # along LAND_MORAINE_GRAIN_DEG, so it reads as till
TILL_SWALE_M, TILL_SWALE_LAM_M = 0.5, 100.0
TILL_KNOB_M, TILL_KNOB_LAM_M = 0.2, 55.0
TILL_WARP_M, TILL_WARP_LAM_M = 60.0, 450.0
TILL_SHORE_CLEAR_M = 100.0        # no relief this close to the lake's shore ring ...
TILL_SHORE_FADE_M = 150.0         # ... and it comes in over this much beyond that
TILL_EDGE_FADE_M = 100.0          # off over the last metres inside the playable boundary
TILL_FLAT_FADE_M = 150.0          # off over this much next to the deliberately flat strip
TILL_FLAT_MIN_HA = 10.0           # a flat region smaller than this is not the strip
TILL_RIDGE_Z_M = (48.0, 60.0)     # the relief fades out between these heights ...
TILL_RIDGE_SLOPE_DEG = (3.0, 6.0) # ... and between these slopes, so the ridge keeps its shape
TILL_RIDGE_KEEP = 0.3             # what is left on the ridge: this much of the swale only

# --- the water and its valley -----------------------------------------------------
# A river runs the whole length of the map from north to south and widens into a lake in
# the northern quarter, with an island in the middle of it. Both sit at the bottom of a
# valley of their own, cut into the floodplain: the ground falls away from the water over
# VALLEY_HALF_W_M either side and is back at the datum beyond that.
#
# Everything below the floodplain is quoted **relative to the water surface at that
# station**, and the water surface falls from north to south. That is the only way the
# numbers stay honest along a river that runs downhill through flat ground: quote the
# bank as an absolute height and it drowns at one end of the map and stands 5 m proud at
# the other. The stack, top to bottom, at any point on the water:
#
#     BASE_ELEV_M                     the mean upland                       82.0 m
#       - VALLEY_DEPTH_M              the valley, over VALLEY_HALF_W_M      60.0 m
#       - WATER_BANK_M                the bank top, over BANK_RUN_M         58.0 m  <- ws
#       - RIVER_DEPTH_M               the river bed                         55.0 m
#       - LAKE_DEPTH_M                the lake bed, at its deepest          20.0 m
#
# The first line is a mean and not a height: the till plain swings about eleven metres
# either side of it, so the valley is anything from 11 m to 33 m deep along its length,
# which is what a river cutting across a moraine field actually does. Every line under it
# is measured from the **local** ground or the **local** water surface, never from the
# datum - quote any of them as a constant and the section lands in the wrong place
# everywhere the ground is not exactly average.
#
# So the river is cut five metres into the ground it runs through and the lake forty,
# which is what was asked for; read as water columns rather than as excavations they are
# three metres and thirty-eight.
#
# The meanders are closed form - two sines beating against each other - and not a random
# walk, so both halves of the pipeline get the identical polyline whatever else is added
# to the map first. The shortest radius of curvature they produce is about 250 m, which
# is what keeps `buffer_ring` at the drawn half-width from folding the channel polygon
# through itself.
RIVER_X0 = 3900.0             # the line the river meanders about
RIVER_A1, RIVER_L1 = 850.0, 5400.0            # the long meander
RIVER_A2, RIVER_L2, RIVER_P2 = 340.0, 2200.0, 1.0     # and the short one riding on it
RIVER_STEP_M = 40.0           # how finely the closed form is sampled into a polyline
# 45 m of half-width, not the 28 it started at. The synthesis grid is 4 m, and the
# cross-section of a 56 m river puts its submerged bank inside 17 m of that grid - six
# cells for a shape with two corners in it. What came out was a trough whose waterline
# sat 25 cm below where the vectors drew it and three metres further out, which is the
# exact disagreement `water_half_w` exists to prevent. Ninety metres across resolves.
RIVER_HALF_W_M = 45.0         # half the drawn water surface
RIVER_DEPTH_M = 3.0           # waterline down to the bed
RIVER_FALL_M = 2.5            # total fall of the water surface across the whole axis

WATER_BANK_M = 2.0            # bank top above the waterline
BANK_RUN_M = 60.0             # how far out from the waterline the bank climbs
VALLEY_HALF_W_M = 500.0       # waterline out to where the ground is floodplain again
# 22 m, not the 15 it was while the uplands were a flat sheet. The datum is the *mean*
# upland now, and the till plain swings about eleven metres either side of it: a valley
# only 15 m deep would have the river standing at the level of the low ground a kilometre
# away, and it would come out of its valley into it. `validate()` bounds that at four
# sigma of the relief, and this is what that bound asks for.
VALLEY_DEPTH_M = 22.0         # mean upland down to the bank top

# The lake. Its shore is a lobed ellipse rather than a drawn one, and the DEM shapes its
# basin with `terrain_ops.ellipse_r` from these same numbers and the same harmonics - if
# the two ever drift the water is painted somewhere the basin is not, which shows up in
# neither output alone. `measure_elevation.py` walks the drawn ring and checks the ground
# under it really is at the waterline.
LAKE_C = (4630.0, 1900.0)
LAKE_A, LAKE_B, LAKE_ROT = 880.0, 620.0, 15.0
LAKE_HARMONICS = ((0.060, 3, 0.7), (0.035, 5, 2.1))
LAKE_DEPTH_M = 38.0           # waterline down to the deepest bed
LAKE_SHELF_M = 260.0          # how far in from either shore the bed takes to fall

# The island, concentric with the lake. It is a wood in the vectors and a rise in the
# terrain, and it breaks the surface by ISLAND_H_M - low enough to read as a piece of
# floodplain the lake left standing rather than as a hill.
ISLAND_A, ISLAND_B = 240.0, 175.0
ISLAND_H_M = 4.0
ISLAND_RISE_M = 120.0

# Where the rim has to let the river through. The river crosses the northern and the
# southern sill on its way off the map, and the rim is added by addition on top of
# whatever is already there - so without a notch the sill simply rides up on the channel
# and the water runs forty metres uphill to leave the map. The notch is held to the
# corridor's own width and feathered wider than the sill is tall.
RIVER_NOTCH_HALF_M = 560.0
RIVER_NOTCH_FEATHER_M = 450.0


def river_axis():
    """The centreline, from EDGE_MIN to EDGE_MAX so it does not die at the canvas edge."""
    def at(y):
        return (RIVER_X0
                + RIVER_A1 * math.sin(2.0 * math.pi * y / RIVER_L1)
                + RIVER_A2 * math.sin(2.0 * math.pi * y / RIVER_L2 + RIVER_P2), y)

    n = int(math.ceil((EDGE_MAX - EDGE_MIN) / RIVER_STEP_M))
    # The last step is short rather than the axis stopping short: an alignment that ends
    # inside the canvas ends at a cliff, and the check in `validate()` is there because
    # eight metres of it is not something the hillshade would ever show.
    return [at(EDGE_MIN + min(i * RIVER_STEP_M, EDGE_MAX - EDGE_MIN))
            for i in range(n + 1)]


def lobed_ellipse_ring(cx, cy, a, b, rot_deg=0.0, harmonics=(), n=96):
    """The shore `terrain_ops.ellipse_r` draws with the same harmonics.

    `ellipse_r` returns q/m, where q is the normalised elliptical radius and m is the
    harmonic modulation, so its shore - the level set at 1 - is q = m. This walks that
    level set. Anything else, including `ellipse_ring`, gives a shore the basin does not
    sit under.
    """
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    out = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        m = 1.0
        for amp, k, ph in harmonics:
            m += amp * math.sin(k * th + ph)
        u, v = a * m * math.cos(th), b * m * math.sin(th)
        out.append((cx + u * c - v * s, cy + u * s + v * c))
    out.append(out[0])
    return out


def lake_ring():
    return lobed_ellipse_ring(*LAKE_C, LAKE_A, LAKE_B, LAKE_ROT, LAKE_HARMONICS)


def island_ring():
    return lobed_ellipse_ring(*LAKE_C, ISLAND_A, ISLAND_B, LAKE_ROT, (), n=64)


def river_profile():
    """The water surface along the river, as `(s_in, s_out, grade, length)`.

    The lake is a flat sheet, so the profile is flat across it and falls at a constant
    grade above and below - which is the whole of the hydrology this map needs, and it is
    exact rather than iterated. `s_in` and `s_out` are the arc lengths at which the axis
    enters and leaves the drawn lake shore, found against that same ring, so the sheet
    ends exactly where the water does.
    """
    axis = river_axis()
    ring = lake_ring()
    arc = [0.0]
    for i in range(len(axis) - 1):
        arc.append(arc[-1] + math.dist(axis[i], axis[i + 1]))
    inside = [i for i, p in enumerate(axis) if point_in_ring(p, ring)]
    if not inside:
        return 0.0, 0.0, RIVER_FALL_M / arc[-1], arc[-1]
    s_in, s_out, length = arc[inside[0]], arc[inside[-1]], arc[-1]
    return s_in, s_out, RIVER_FALL_M / (s_in + length - s_out), length


LAKE_WS_M = BASE_ELEV_M - VALLEY_DEPTH_M - WATER_BANK_M    # the lake surface, 58.0 m


# --- the survey and the roads -----------------------------------------------------
# The Public Land Survey System, which is why the roads on a map of the American midwest
# look the way they do: the ground was subdivided into one-mile sections before anyone
# built on it, and the roads went on the section lines. So the grid is not a design
# choice here - it is a mile, exactly, and everything else has to fit around it.
#
# Two through roads run the length of the map a mile in from the east and the west edge,
# which is what was asked for and also happens to put both of them clear of the water:
# the river swings between x = 2791 and x = 5089 and its valley reaches 545 m either
# side of that, so neither trunk road ever meets it. The service roads are the section
# lines between them, and each one crosses the river exactly once - a river that is a
# single-valued function of northing cannot be crossed twice by an east-west line.
#
# The one thing the mile grid has to be told about is the lake. It is 1290 m across
# north to south and the sections are 1609 m apart, so there are only 319 m of anchor to
# choose from that keep a section line out of the water. PLSS_EW_ANCHOR_M centres that
# window, which leaves the two nearest section roads running 160 m off the north and
# south shores - a road along a lake shore, which is a thing that exists, rather than a
# mile and a half of causeway.
MILE_M = 1609.344

ROAD_MAIN_INSET_M = MILE_M                      # the trunk roads, a mile in from the edge
ROAD_W_X = ROAD_MAIN_INSET_M                    # 1609.344
ROAD_E_X = PLAYABLE_M - ROAD_MAIN_INSET_M       # 6582.656
PLSS_EW_ANCHOR_M = 1080.5
PLSS_EW_Y = [PLSS_EW_ANCHOR_M + k * MILE_M for k in range(5)]

ROAD_PRIMARY = dict(half_width_m=5.5, feather_m=14.0, grade_max=0.050)
ROAD_SECTION = dict(half_width_m=4.0, feather_m=11.0, grade_max=0.070)

# How far off the centreline of a watercourse a road has to be carried on a deck. The
# river is 90 m wide and a section line crosses it square or nearly so, so this sets a
# span of about 130 m - the ground under it is left exactly as the water cut it.
BRIDGE_CLEAR_M = RIVER_HALF_W_M + 20.0

# Which section lines get a bridge, by index into PLSS_EW_Y. Only the outer two: the
# three in the middle run down to the river and stop, which is what a section-line road
# without a bridge does - the survey put a road allowance on every mile line whether or
# not anyone ever built a crossing on it, and in river country most of them dead-end at
# the bank and the traffic goes round by the trunk roads. It also means the two roads
# that do cross carry the through traffic, which is what makes a trunk road a trunk road.
PLSS_BRIDGED = (0, len(PLSS_EW_Y) - 1)
# Where a road that is not going to cross stops: on the top of the bank, clear of the
# water but close enough to read as a road that ran out rather than one that was aimed
# somewhere else.
ROAD_STUB_SETBACK_M = RIVER_HALF_W_M + BANK_RUN_M


def water_crossings(axis, clearance=None):
    """Arc-length spans where an alignment runs within `clearance` of open water.

    What the DEM leaves alone and the OSM tags `bridge=yes`, computed once here so the
    two cannot disagree about where the deck starts. Grading the ground under a bridge
    is how a channel gets filled in by a road that was supposed to cross it.
    """
    clearance = BRIDGE_CLEAR_M if clearance is None else clearance
    dense = densify(axis, 10.0)
    riv = river_axis()
    ring = lake_ring()
    # Two cheap restrictions, because this is called for every alignment on the map and
    # walks a densified one against 322 river vertices and 97 lake ones: it was 29 of the
    # 36 seconds it took to import this module.
    #
    # The river is a single-valued function of northing, so the axis comes out sorted in
    # y and a point can only be within `clearance` of the stretch inside its own band of
    # northing - a slice, with a vertex of margin at each end. That is exact here and
    # would not be on a watercourse that doubled back.
    rys = [q[1] for q in riv]
    lx = min(q[0] for q in ring) - clearance, max(q[0] for q in ring) + clearance
    ly = min(q[1] for q in ring) - clearance, max(q[1] for q in ring) + clearance
    arc, acc = [0.0], 0.0
    for i in range(1, len(dense)):
        acc += math.dist(dense[i - 1], dense[i])
        arc.append(acc)
    out, run = [], None
    for i, p in enumerate(dense):
        a = max(0, bisect.bisect_left(rys, p[1] - clearance) - 1)
        b = min(len(riv), bisect.bisect_right(rys, p[1] + clearance) + 1)
        near = b - a >= 2 and dist_to_polyline(p, riv[a:b]) < clearance
        if not near and lx[0] <= p[0] <= lx[1] and ly[0] <= p[1] <= ly[1]:
            near = (point_in_ring(p, ring)
                    or min(math.dist(p, q) for q in ring) < clearance)
        if near and run is None:
            run = arc[i]
        elif not near and run is not None:
            out.append((run, arc[i]))
            run = None
    if run is not None:
        out.append((run, arc[-1]))
    return out


def _ns_road(rid, name, x, spec):
    axis = [(x, EDGE_MIN), (x, EDGE_MAX)]
    return dict(id=rid, name=name, kind='primary', axis=axis,
                bridge_spans=water_crossings(axis), **spec)


def _ew_road(rid, name, y, spec, bridged):
    """A section-line road. Bridged ones are one alignment; the rest are two stubs that
    stop on the bank, and the gap between them is the crossing nobody built."""
    axis = [(EDGE_MIN, y), (EDGE_MAX, y)]
    if bridged:
        return [dict(id=rid, name=name, kind='section', axis=axis,
                     bridge_spans=water_crossings(axis), **spec)]
    stop = water_crossings(axis, ROAD_STUB_SETBACK_M)
    if not stop:
        return [dict(id=rid, name=name, kind='section', axis=axis,
                     bridge_spans=[], **spec)]
    x0 = EDGE_MIN + stop[0][0]
    x1 = EDGE_MIN + stop[-1][1]
    return [dict(id=f'{rid}_west', name=f'{name} Oeste', kind='section',
                 axis=[(EDGE_MIN, y), (x0, y)], bridge_spans=[], **spec),
            dict(id=f'{rid}_east', name=f'{name} Este', kind='section',
                 axis=[(x1, y), (EDGE_MAX, y)], bridge_spans=[], **spec)]


# --- the towns --------------------------------------------------------------------
# Four towns, one on each corner where a trunk road meets a section line that carries a
# bridge. That is not a decoration on the road grid, it is the only placement the road
# grid allows: the two bridged section lines are the only east-west roads that get
# across the river, so their junctions with the two trunk roads are the four points on
# the map where two through routes cross, and a town in the midwest stands where two
# through routes cross. The three unbridged section lines dead-end on the bank and carry
# nobody, which is exactly why nothing is built on them.
#
# Each town is a grid of TOWN_COLS x TOWN_ROWS blocks of TOWN_BLOCK_W_M by
# TOWN_BLOCK_H_M, and the trunk road runs up the middle of it - the middle *street line*
# of the grid is the trunk road rather than a street of the town's own, so main street
# is the highway, which is what a section-line town looks like from the air. The section
# line does the same thing across it, so the junction the town is named for is the
# crossroads in its centre and there are equal numbers of blocks on all four sides.
#
# The block sizes are the brief and everything else falls out of them. What a road takes
# out of the grid is its own nominal feather either side of the centreline - the same
# clearance the parcelling uses, and the reason a primary takes 28 m of the grid where a
# town street takes 16 - so the blocks come out at exactly the size they are quoted at
# whatever class of road happens to bound them. `validate()` measures the rings rather
# than trusting the arithmetic.
TOWN_COLS, TOWN_ROWS = 4, 4
TOWN_BLOCK_W_M = 100.0        # east-west, across the trunk road
TOWN_BLOCK_H_M = 100.0        # north-south, along it
TOWN_STREET = dict(half_width_m=3.0, feather_m=8.0, grade_max=0.080)

# The platform the whole town is levelled onto. It reaches TOWN_PAD_MARGIN_M past the
# outermost street so the graded ground runs out beyond the last kerb rather than at it,
# and it is not dead flat: TOWN_DRAIN_GRADE of fall to the south is a third of a percent,
# far under any road's ruling grade and enough that the ground drains instead of
# terracing. The feather is nominal only - the DEM widens it to 1.5*|dz|/tan(4 deg)
# wherever the cut is deep, the same identity that sets every bank on the map.
TOWN_PAD_MARGIN_M = 20.0
TOWN_PAD_FEATHER_M = 30.0
TOWN_DRAIN_GRADE = 0.003

TOWN_SITES = (
    ('town_nw', 'Ciudad del Noroeste', ROAD_W_X, PLSS_EW_Y[PLSS_BRIDGED[0]]),
    ('town_ne', 'Ciudad del Noreste', ROAD_E_X, PLSS_EW_Y[PLSS_BRIDGED[0]]),
    ('town_sw', 'Ciudad del Suroeste', ROAD_W_X, PLSS_EW_Y[PLSS_BRIDGED[-1]]),
    ('town_se', 'Ciudad del Sureste', ROAD_E_X, PLSS_EW_Y[PLSS_BRIDGED[-1]]),
)


def _street_lines(n, block, gap_mid, gap_side):
    """Offsets of the `n + 1` street centrelines that bound `n` blocks, measured from
    the through road in the middle of the grid.

    `n` has to be even: the through road is the middle line, so there are `n / 2` blocks
    either side of it. Each step is one block plus what the two lines bounding it take
    out of the grid, which is why the first step out from the middle is wider than the
    rest - a primary road's half-allowance against a street's.
    """
    out = [0.0]
    for k in range(n // 2):
        out.append(out[-1] + block + (gap_mid if k == 0 else gap_side) + gap_side)
    return [-v for v in reversed(out[1:])] + out


def _line_gaps(n, gap_mid, gap_side):
    """What each of those lines takes out of the grid on either side of itself."""
    return [gap_mid if i == n // 2 else gap_side for i in range(n + 1)]


TOWN_COL_LINES = _street_lines(TOWN_COLS, TOWN_BLOCK_W_M,
                               ROAD_PRIMARY['feather_m'], TOWN_STREET['feather_m'])
TOWN_COL_GAPS = _line_gaps(TOWN_COLS, ROAD_PRIMARY['feather_m'],
                           TOWN_STREET['feather_m'])
TOWN_ROW_LINES = _street_lines(TOWN_ROWS, TOWN_BLOCK_H_M,
                               ROAD_SECTION['feather_m'], TOWN_STREET['feather_m'])
TOWN_ROW_GAPS = _line_gaps(TOWN_ROWS, ROAD_SECTION['feather_m'],
                           TOWN_STREET['feather_m'])

TOWN_HALF_W_M = TOWN_COL_LINES[-1] + TOWN_PAD_MARGIN_M
TOWN_HALF_H_M = TOWN_ROW_LINES[-1] + TOWN_PAD_MARGIN_M


def town_streets(tid, name, cx, cy):
    """The town's own streets: every line of the block grid except the two the trunk
    road and the section line already stand on.

    Each one runs from the outermost cross line to the outermost cross line, so both its
    ends land on another alignment and the grid is a connected network rather than eight
    sticks laid beside each other. `validate()` holds that: a street is the one class of
    corridor allowed to stop inside the canvas, and only on a road it meets.
    """
    out = []
    y0, y1 = cy + TOWN_ROW_LINES[0], cy + TOWN_ROW_LINES[-1]
    x0, x1 = cx + TOWN_COL_LINES[0], cx + TOWN_COL_LINES[-1]
    for i, dx in enumerate(TOWN_COL_LINES):
        if i == TOWN_COLS // 2:
            continue                              # the trunk road is this line
        out.append(dict(id=f'{tid}_ns{i}', name=f'{name} Calle {i + 1}', kind='street',
                        axis=[(cx + dx, y0), (cx + dx, y1)], bridge_spans=[],
                        **TOWN_STREET))
    for j, dy in enumerate(TOWN_ROW_LINES):
        if j == TOWN_ROWS // 2:
            continue                              # and the section line is this one
        out.append(dict(id=f'{tid}_ew{j}', name=f'{name} Avenida {j + 1}',
                        kind='street', axis=[(x0, cy + dy), (x1, cy + dy)],
                        bridge_spans=[], **TOWN_STREET))
    return out


def town_blocks(tid, name, cx, cy):
    """The `TOWN_COLS * TOWN_ROWS` blocks, as tagged rings. Each one is the ground
    between two street lines, inset by what each of them takes out of the grid."""
    out = []
    for r in range(TOWN_ROWS):
        y0 = cy + TOWN_ROW_LINES[r] + TOWN_ROW_GAPS[r]
        y1 = cy + TOWN_ROW_LINES[r + 1] - TOWN_ROW_GAPS[r + 1]
        for c in range(TOWN_COLS):
            x0 = cx + TOWN_COL_LINES[c] + TOWN_COL_GAPS[c]
            x1 = cx + TOWN_COL_LINES[c + 1] - TOWN_COL_GAPS[c + 1]
            out.append({'id': f'{tid}_b{r}{c}',
                        'name': f'{name} Cuadra {r + 1}-{c + 1}',
                        'ring': rect_ring(x0, y0, x1, y1),
                        'tags': {'landuse': 'farmyard'}})
    return out


def town_pad(tid, name, cx, cy):
    """The platform the whole town stands on. It carries no tags of its own: what is
    drawn here is the blocks, the same way the island is an `AREAS` ring rather than
    something hanging off the lake record."""
    w, h = 2.0 * TOWN_HALF_W_M, 2.0 * TOWN_HALF_H_M
    return {'id': f'{tid}_pad', 'name': name, 'kind': 'town', 'centre': (cx, cy),
            'size': (w, h),
            'ring': rect_ring(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
            'feather_m': TOWN_PAD_FEATHER_M, 'drain_grade': TOWN_DRAIN_GRADE}


# --- the industrial zones ---------------------------------------------------------
# Ten two-hectare aprons hung off the trunk roads and the section lines. An industrial
# yard is a thing that wants a road at its gate and flat ground under it, and both of
# those are relationships rather than coordinates - so a site is recorded as *which road,
# how far along it, and which side*, and the rectangle is worked out from the road's own
# record. Write the corners out as four numbers instead and the day a road moves the
# yards stay where they were, ten aprons in the middle of a field with no way in.
#
# The setback is measured from the edge of the running surface and not from the
# centreline, because that is what "ten metres off the road" means to anyone standing on
# it. Against a primary that puts the fence 15.5 m from the centreline and against a
# section line 14 m, both of which clear the per-class road clearance the parcelling uses
# (14 m and 11 m) - so a yard placed this way is inside the letter of the brief and
# outside the verge of the road, which is the only way both can be true at once.
#
# The yard is square and it is a stated number of hectares, so the side is neither of
# those numbers - it is what falls out of them. The area is the requirement and the shape
# is the requirement; a side rounded to a whole metre would satisfy the shape and quietly
# miss the area, which is the kind of thing nothing downstream would ever mention. This
# map already runs on 1609.344 m section lines, so an awkward number is not a problem
# here. Two sizes: ten small yards and two large ones.
INDUSTRY_SMALL_HA = 2.0       # -> 141.42 m a side
INDUSTRY_LARGE_HA = 5.0       # -> 223.61 m a side
FARM_AREA_HA = 20.0           # -> 447.21 m a side

# `ROADSIDE_SETBACK_M` is shared by everything that stands beside a road - the aprons, the
# granjas and the woods - because a fence ten metres off the road is ten metres off the
# road whatever is behind it. The other two are the yards': a drain grade is a grade and
# does not want to be re-argued per size, it just falls further across a bigger yard.
ROADSIDE_SETBACK_M = 10.0     # from the *edge of the running surface*, not the centreline
YARD_FEATHER_M = 25.0         # nominal; the DEM widens it with the cut like any other
YARD_DRAIN_GRADE = 0.004      # 57 cm across a 2 ha apron, 1.8 m across a 20 ha farm


def yard_side(area_ha):
    """The side of a square yard of `area_ha`."""
    return math.sqrt(area_ha * 10000.0)

# Where they are: (id, name, road, station along it, side, hectares). `side` is +1
# towards increasing x or y - east of a trunk road, south of a section line - and -1 the
# other way. The stations are spread over the whole map, four small yards on the two
# trunk roads and six on the section lines so that every one of the five has one, and the
# two large yards on the trunk roads, which is where the traffic a five-hectare operation
# generates has somewhere to go. None of them is anywhere near the water: `validate()`
# holds every corner a full valley half-width off the river and the lake, because an
# apron on a valley side is a cut nobody would make.
INDUSTRY_SITES = (
    ('ind_w1', 'Zona Industrial del Oeste 1', 'road_west', 2000.0, +1,
     INDUSTRY_SMALL_HA),
    ('ind_w2', 'Zona Industrial del Oeste 2', 'road_west', 5200.0, -1,
     INDUSTRY_SMALL_HA),
    ('ind_e1', 'Zona Industrial del Este 1', 'road_east', 3400.0, +1,
     INDUSTRY_SMALL_HA),
    ('ind_e2', 'Zona Industrial del Este 2', 'road_east', 6600.0, -1,
     INDUSTRY_SMALL_HA),
    ('ind_s1', 'Zona Industrial de Servicio 1', 'section_1', 2800.0, -1,
     INDUSTRY_SMALL_HA),
    ('ind_s2', 'Zona Industrial de Servicio 2', 'section_1', 5800.0, +1,
     INDUSTRY_SMALL_HA),
    ('ind_s3', 'Zona Industrial de Servicio 3', 'section_2_west', 900.0, +1,
     INDUSTRY_SMALL_HA),
    ('ind_s4', 'Zona Industrial de Servicio 4', 'section_3_east', 7400.0, -1,
     INDUSTRY_SMALL_HA),
    ('ind_s5', 'Zona Industrial de Servicio 5', 'section_4_west', 2600.0, +1,
     INDUSTRY_SMALL_HA),
    ('ind_s6', 'Zona Industrial de Servicio 6', 'section_5', 5700.0, -1,
     INDUSTRY_SMALL_HA),
    ('ind_b1', 'Zona Industrial Mayor del Oeste', 'road_west', 3400.0, +1,
     INDUSTRY_LARGE_HA),
    ('ind_b2', 'Zona Industrial Mayor del Este', 'road_east', 4900.0, -1,
     INDUSTRY_LARGE_HA),
)

# The farms. Same shape of thing as an apron - a square yard hung off a road, ten metres
# off the kerb - and built by the same code from the same table shape. Two differences,
# and both of them are data rather than another implementation: a granja is twenty
# hectares, and it stands **only on a section line**. That last one is not decoration.
# A section-line road is the road a farm in this country actually fronts onto: the trunk
# roads carry the through traffic, and 447 m of yard gate opening onto one of them is a
# thing nobody builds. `validate()` holds it off the record itself, so a farm moved onto
# a trunk road is a complaint and not a surprise in the editor.
#
# 447 m a side is a large square, and where it can go is decided almost entirely by what
# is already on the map: the river's valley reaches 500 m either side of a channel that
# swings between x = 2791 and x = 5089, so the whole middle third of every section line
# is out; a trunk road runs up each side of that; and the four towns and the twelve
# aprons take their own ground. What is left is a window either side of each trunk road,
# and these six sit in it.
FARM_SITES = (
    ('farm_1', 'Granja del Noroeste', 'section_1', 800.0, -1),
    ('farm_2', 'Granja del Oeste', 'section_3_west', 900.0, +1),
    ('farm_3', 'Granja del Suroeste', 'section_5', 800.0, -1),
    ('farm_4', 'Granja del Noreste', 'section_2_east', 7200.0, +1),
    ('farm_5', 'Granja del Este', 'section_4_east', 7300.0, -1),
    ('farm_6', 'Granja del Sureste', 'section_5', 7300.0, -1),
)


def corridor_by_id(cid):
    for c in CORRIDORS:
        if c['id'] == cid:
            return c
    raise KeyError(f"no corridor {cid!r}")


def roadside_geometry(road_id, station, side, area_ha, setback_m):
    """`(centre, size)` for a square yard on `road_id`, from the road's own record.

    The offset is half the running surface plus the setback plus half the yard, so the
    near fence lands exactly `setback_m` off the edge of the road whatever class of road
    it is and whatever size the yard is. The yard being square, `side` only decides which
    way it is put and the size is the same either way.
    """
    c = corridor_by_id(road_id)
    ax = c['axis']
    a = yard_side(area_ha)
    off = c['half_width_m'] + setback_m + a / 2.0
    if abs(ax[0][0] - ax[-1][0]) < abs(ax[0][1] - ax[-1][1]):
        return (ax[0][0] + side * off, station), (a, a)
    return (station, ax[0][1] + side * off), (a, a)


def roadside_pad(pid, name, road_id, station, side, area_ha, setback_m, drain_grade,
                 feather_m, tags, road_kinds, kind):
    """A levelled yard that draws its own footprint - which is the difference between
    this and a town pad. A yard *is* the thing standing on the platform, so it carries
    `tags` and `emit_pads` draws it; a town's platform carries none, because what is
    drawn there is its blocks.

    `road_kinds` travels with the record so `validate()` can hold "a farm only stands on
    a section line" without knowing what a farm is: the placement rules are one loop over
    every roadside yard, and the differences between an apron and a granja are data.
    """
    (cx, cy), (w, h) = roadside_geometry(road_id, station, side, area_ha, setback_m)
    return {'id': pid, 'name': name, 'kind': kind, 'road': road_id,
            'station': station, 'side': side, 'area_ha': area_ha,
            'setback_m': setback_m, 'road_kinds': road_kinds,
            'centre': (cx, cy), 'size': (w, h),
            'ring': rect_ring(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
            'feather_m': feather_m, 'drain_grade': drain_grade, 'tags': dict(tags)}


def industry_pad(pid, name, road_id, station, side, area_ha):
    return roadside_pad(pid, name, road_id, station, side, area_ha,
                        ROADSIDE_SETBACK_M, YARD_DRAIN_GRADE, YARD_FEATHER_M,
                        {'landuse': 'farmyard', 'building': 'industrial'},
                        ('primary', 'section'), 'industry')


def farm_pad(pid, name, road_id, station, side):
    return roadside_pad(pid, name, road_id, station, side, FARM_AREA_HA,
                        ROADSIDE_SETBACK_M, YARD_DRAIN_GRADE, YARD_FEATHER_M,
                        {'landuse': 'farmyard'}, ('section',), 'farm')


# --- the roadside woods -----------------------------------------------------------
# Five twenty-hectare woods along the trunk roads and the section lines: rectangles laid
# long-side-on along the road, `WOOD_ASPECT` times as long as they are deep, which is
# what a planted woodlot beside a road looks like. The two sides fall out of the area and
# the aspect and nothing else - `sqrt(area/aspect)` deep by `aspect` times that long, so
# 632 by 316 m - and there is no rounding anywhere in it.
#
# A wood is the first thing on this map that is *only* vectors. Nobody levels ground to
# grow trees on, so there is no pad, no feather and no drain: these live in `AREAS`
# rather than in `PADS`, and the heightmap does not change when one is added. That is the
# whole difference between a granja and a bosque here, and it is the reason a wood may
# sit on ground a yard could not.
WOOD_AREA_HA = 20.0
WOOD_ASPECT = 2.0             # long side along the road, so it reads as a belt
# What is growing in them. `leaf_type` is not a tag that makes a way drawable - the
# closed vocabulary in RENDERED_TAGS is, and `natural=wood` is what gets these rings on
# the map - but it is not dead weight either: both renderers colour a needleleaf wood
# apart from a broadleaf one, the same way `visualize_osm` tells an industrial apron from
# a farmyard. A tag no renderer reads is the one thing not worth emitting. The woods and
# the island are the conifer; the shelterbelts are hardwood (`SHELTER_LEAF_TYPE`), and
# that difference is the whole reason this tag earns its place on the ring. It is not
# written out here: `wood_tags` carries it along with the `landuse=farmyard` every
# planting stands on, so all five kinds of timber are tagged the same way.
WOOD_LEAF_TYPE = 'needleleaved'
# Same table shape as the yards: which road, how far along, which side. The setback is
# the shared roadside one, measured from the edge of the running surface - so the trees
# come down to the verge and stop where the road's graded platform begins.
WOOD_SITES = (
    ('wood_1', 'Bosque del Oeste', 'road_west', 3400.0, -1),
    ('wood_2', 'Bosque del Este', 'road_east', 5100.0, +1),
    ('wood_3', 'Bosque del Norte', 'section_1', 2600.0, +1),
    ('wood_4', 'Bosque del Este Bajo', 'section_4_east', 6000.0, +1),
    ('wood_5', 'Bosque del Suroeste', 'section_5', 2700.0, -1),
)


def wood_ring_at_origin(along_x, area_ha):
    """The wood's outline about (0, 0), long side east-west if `along_x`."""
    d = math.sqrt(area_ha * 10000.0 / WOOD_ASPECT)      # the short side
    w, h = (WOOD_ASPECT * d, d) if along_x else (d, WOOD_ASPECT * d)
    return rect_ring(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)


def wood_geometry(road_id, station, side, area_ha):
    """`(centre, ring)` for a wood on `road_id`.

    The centre is put wherever it has to be for the ring's *own* nearest point to land
    `ROADSIDE_SETBACK_M` off the edge of the road, read off the ring rather than assumed
    from a half-side. It comes to the same thing for a rectangle and it did not for the
    lobed outline these were drawn with first, which is reason enough to leave it
    measuring the ring: the shape is a constant away from changing again.
    """
    c = corridor_by_id(road_id)
    ax = c['axis']
    vertical = abs(ax[0][0] - ax[-1][0]) < abs(ax[0][1] - ax[-1][1])
    ring0 = wood_ring_at_origin(not vertical, area_ha)
    d = [p[0] for p in ring0] if vertical else [p[1] for p in ring0]
    reach = -min(d) if side > 0 else max(d)
    off = c['half_width_m'] + ROADSIDE_SETBACK_M + reach
    cx, cy = ((ax[0][0] + side * off, station) if vertical
              else (station, ax[0][1] + side * off))
    return (cx, cy), [(cx + x, cy + y) for x, y in ring0]


def wood_area(wid, name, road_id, station, side):
    (cx, cy), ring = wood_geometry(road_id, station, side, WOOD_AREA_HA)
    return {'id': wid, 'name': name, 'road': road_id, 'station': station, 'side': side,
            'area_ha': WOOD_AREA_HA, 'centre': (cx, cy), 'ring': ring,
            'tags': wood_tags(WOOD_LEAF_TYPE)}


# --- the shelterbelts -------------------------------------------------------------
# Rompevientos: strips of hardwood SHELTER_W_M across, between the fields and along the
# primary roads. There is no site table. Where a belt can stand is read off what is
# already on the map, the same way the riverside timber was: the fields say where the
# boundaries are, the roads say where the frontage is, and everything else says where a
# belt has to stop.
#
# Two kinds, one piece of code:
#
#   * **Along a primary**, on both sides: the belt stands ROADSIDE_SETBACK_M off the
#     running surface, like a yard does, and runs the length of the road. It is cut
#     wherever something crosses or stands in the way - a field road joining the primary,
#     a yard, the town, a wood, the lake, the clean strip - and each stretch that is left
#     is a belt of its own.
#   * **Between two fields**, wherever two fields face each other across a gap of
#     SHELTER_GAP_MAX_M or less. On this map nearly every such gap has a field road down
#     the middle of it (an 18 m gap with a tertiary on its axis and 9 m of verge either
#     side), and a belt cannot be planted on a road, so where a road runs in the gap the
#     belt stands *beside* it, on the SHELTER_ROAD_SIDE side, with the same setback as
#     along a primary. Where nothing runs in the gap the belt is centred on it.
#
# The fields then give way. A belt is 50 m across and the gap it stands in is 18, so the
# field on the far side of it comes back to SHELTER_CLEAR_M off the trees - the same
# headland the twelve woods already on the map keep from the fields next to them, which
# is where the number comes from. That is done by cutting the field, not by moving the
# belt: a field's edge is pulled straight back to the clearance line, and the corners the
# cut makes are rounded to the FIELD_CORNER_R_M the fields were drawn with, so a trimmed
# field looks like every other field and not like something with a bite out of it.
#
# What a cut may not do is leave a field that is not a field. FIELD_MIN_HA and
# FIELD_MIN_SIDE_M are the floors: a belt that would push the field beside it under
# either one is not planted on that side. Along a primary the belt is planted anyway and
# the field it would ruin is treated as an obstacle instead, so the belt stops short of
# it; between fields the belt tries the other side of the road first and is dropped if
# that fails too. The north row of fields is why - 104 m deep between the clean strip
# and the first field road, and a belt on the windward side of that road would have left
# 35 m of it.
#
# The ends of a belt are not sampled positions. The walk along the line goes in steps of
# SHELTER_STEP_M, but each run is then extended by bisection to exactly the clearance
# from whatever stopped it, so a transversal belt meets the north-south belt it runs into
# at SHELTER_JOIN_M and not at some fraction of a step short of it. Which of the two
# gives way at a crossing is decided by order: the belts along the primaries are laid
# first, then the north-south field belts, then the east-west ones, and each later belt
# stops at the earlier. A transversal belt crossing a north-south one, drawn twice over
# the same ground, is the overlap case the old map found the hard way.
SHELTER_W_M = 25.0
# What is planted in them, and it is deliberately not what is in the woods: a windbreak
# on a field boundary is a row of hardwood, the woods are conifer, and both renderers
# colour the two apart.
SHELTER_LEAF_TYPE = 'broadleaved'
SHELTER_CLEAR_M = 15.0            # to a field, a yard, a wood - what the woods keep now
SHELTER_WATER_CLEAR_M = 60.0      # to the lake shore, where the fields already stop
SHELTER_JOIN_M = 1.0              # where one belt runs into another
SHELTER_GAP_MAX_M = 40.0          # two fields further apart than this do not share a boundary
SHELTER_MIN_LEN_M = 100.0         # shorter than this is a clump, not a belt
SHELTER_STEP_M = 5.0              # how finely a line is walked for obstacles
SHELTER_MERGE_STEPS = 25          # ... and how finely the gap between two runs is re-walked before they are joined
SHELTER_EDGE_MIN_M = 40.0         # a straight edge shorter than this is a corner fillet
# Which side of a field road the belt stands on: -1 is west of a north-south road and
# north of an east-west one, the windward side for the north-westerlies that do the
# damage on this ground. +1 is the other side, and it is what a belt falls back to when
# the windward field would go under the floor.
SHELTER_ROAD_SIDE = -1
# How many there may be. The map could carry two hundred; twenty is what was asked for,
# and the twenty that stay are the longest, because a belt shelters the frontage it runs
# along and a long one shelters more of it. `validate()` holds the cap.
SHELTER_MAX_COUNT = 20
# Field roads that carry no belt, by corridor id. Two fields facing each other across
# one of these keep their whole depth: the lane along the north band (way 119) has
# Campo 2 to 6 on one side and Campo 64 on the other, and a 50 m belt beside it would
# take 65 m off five fields to shelter the strip that was just put under the plough.
SHELTER_NO_BELT_ROADS = {'road_119'}
# The corner radius the fields were drawn with, fitted to the arcs in the input. A cut
# corner is rounded back to it so the trimmed fields match the untouched ones.
FIELD_CORNER_R_M = 9.0


def simplify_polyline(pts, eps):
    """Ramer-Douglas-Peucker. A closed ring works too: the first split falls on the
    vertex farthest from the shared endpoint, and the ring stays closed."""
    if len(pts) < 3:
        return list(pts)
    a, b = pts[0], pts[-1]
    closed = math.dist(a, b) < 1e-9
    dmax, idx = -1.0, 0
    for i in range(1, len(pts) - 1):
        d = math.dist(pts[i], a) if closed else seg_point_dist(pts[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return (simplify_polyline(pts[:idx + 1], eps)[:-1]
                + simplify_polyline(pts[idx:], eps))
    return [a, b]


def ring_bbox(ring, grow=0.0):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs) - grow, min(ys) - grow, max(xs) + grow, max(ys) + grow)


def boxes_apart(a, b):
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def seg_seg_dist(a, b, c, d):
    if segs_cross(a, b, c, d):
        return 0.0
    return min(seg_point_dist(a, c, d), seg_point_dist(b, c, d),
               seg_point_dist(c, a, b), seg_point_dist(d, a, b))


def seg_polyline_dist(a, b, pts, stop=0.0, within=None):
    """Distance from segment ab to a polyline; gives up early once under `stop`. With
    `within`, a segment of the polyline is only measured if it reaches into that box -
    the distance comes back as at least `within`'s margin otherwise, which is all a
    clearance test needs to know."""
    best = 1e18
    if within is not None:
        x0, y0, x1, y1 = within
        for i in range(len(pts) - 1):
            c, d = pts[i], pts[i + 1]
            if (max(c[0], d[0]) < x0 or min(c[0], d[0]) > x1
                    or max(c[1], d[1]) < y0 or min(c[1], d[1]) > y1):
                continue
            dd = seg_seg_dist(a, b, c, d)
            if dd < best:
                best = dd
                if best <= stop:
                    break
        return best
    for i in range(len(pts) - 1):
        d = seg_seg_dist(a, b, pts[i], pts[i + 1])
        if d < best:
            best = d
            if best <= stop:
                break
    return best


def seg_ring_dist(a, b, ring, within=None):
    """0 if the segment enters the ring, else the distance to its boundary."""
    if point_in_ring(a, ring) or point_in_ring(b, ring):
        return 0.0
    return seg_polyline_dist(a, b, ring, within=within)


def ring_ring_dist(p, q, stop=0.0):
    """Distance between two closed rings: 0 if they share any ground."""
    if any(point_in_ring(v, q) for v in p[:-1]) or any(point_in_ring(v, p) for v in q[:-1]):
        return 0.0
    best = 1e18
    for i in range(len(p) - 1):
        d = seg_polyline_dist(p[i], p[i + 1], q, stop)
        if d < best:
            best = d
            if best <= stop:
                break
    return best


def clip_halfplane(poly, q, n):
    """Sutherland-Hodgman against one half-plane: keep the part of the open polygon
    `poly` where (p - q) . n >= 0."""
    out = []
    if not poly:
        return out
    prev = poly[-1]
    dprev = (prev[0] - q[0]) * n[0] + (prev[1] - q[1]) * n[1]
    for p in poly:
        d = (p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1]
        if d >= 0.0:
            if dprev < 0.0:
                t = dprev / (dprev - d)
                out.append((prev[0] + t * (p[0] - prev[0]), prev[1] + t * (p[1] - prev[1])))
            out.append(p)
        elif dprev >= 0.0:
            t = dprev / (dprev - d)
            out.append((prev[0] + t * (p[0] - prev[0]), prev[1] + t * (p[1] - prev[1])))
        prev, dprev = p, d
    return out


def poly_area_m2(poly):
    s = 0.0
    m = len(poly)
    for i in range(m):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % m]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def convex_halfplanes(box):
    """(point, outward normal) for every edge of a convex ring, whichever way it winds."""
    pts = box[:-1] if math.dist(box[0], box[-1]) < 1e-9 else list(box)
    s = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        s += x0 * y1 - x1 * y0
    sgn = 1.0 if s > 0 else -1.0
    out = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ll = math.hypot(dx, dy) or 1.0
        out.append((a, (sgn * dy / ll, -sgn * dx / ll)))
    return out


def shrink_convex(box, eps):
    """The same convex ring pulled in by `eps` on every side."""
    pts = box[:-1] if math.dist(box[0], box[-1]) < 1e-9 else list(box)
    poly = list(pts)
    for q, n in convex_halfplanes(box):
        poly = clip_halfplane(poly, (q[0] - n[0] * eps, q[1] - n[1] * eps), (-n[0], -n[1]))
    return poly + [poly[0]] if poly else []


def fillet_ring(ring, r, arc_pts=6):
    """Round every sharp corner of a ring to radius `r`, where both edges are long
    enough to take the tangent. Vertices along an existing arc turn by too little to be
    touched, so a ring can go through this twice and come out the same."""
    pts = ring[:-1] if math.dist(ring[0], ring[-1]) < 1e-9 else list(ring)
    m = len(pts)
    out = []
    for i in range(m):
        p0, v, p1 = pts[i - 1], pts[i], pts[(i + 1) % m]
        ux, uy = v[0] - p0[0], v[1] - p0[1]
        wx, wy = p1[0] - v[0], p1[1] - v[1]
        lu, lw = math.hypot(ux, uy), math.hypot(wx, wy)
        if lu < 1e-9 or lw < 1e-9:
            continue
        ux, uy, wx, wy = ux / lu, uy / lu, wx / lw, wy / lw
        theta = math.acos(max(-1.0, min(1.0, -(ux * wx + uy * wy))))   # interior angle
        turn = math.degrees(math.pi - theta)
        if turn < 45.0 or turn > 135.0:
            out.append(v)
            continue
        t = r / math.tan(theta / 2.0)
        if t > lu / 2.0 - 0.5 or t > lw / 2.0 - 0.5:
            out.append(v)
            continue
        s = 1.0 if ux * wy - uy * wx > 0 else -1.0
        t1 = (v[0] - ux * t, v[1] - uy * t)
        t2 = (v[0] + wx * t, v[1] + wy * t)
        cx, cy = t1[0] - uy * r * s, t1[1] + ux * r * s
        a0 = math.atan2(t1[1] - cy, t1[0] - cx)
        a1 = math.atan2(t2[1] - cy, t2[0] - cx)
        sweep = a1 - a0
        while sweep > math.pi:
            sweep -= 2 * math.pi
        while sweep < -math.pi:
            sweep += 2 * math.pi
        for j in range(arc_pts + 1):
            a = a0 + sweep * j / arc_pts
            out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out + [out[0]]


# --- obstacles: what a belt has to stop for ---------------------------------------
class _Obstacles:
    """Everything a belt is held off, bucketed on a coarse grid so a station only looks
    at what is near it. Each entry is a ring or an axis with its own clearance."""
    CELL = 250.0

    def __init__(self):
        self.items = []
        self.grid = {}

    def add(self, ring=None, axis=None, clear=0.0, tag=''):
        pts = ring if ring is not None else axis
        bbox = ring_bbox(pts, clear + SHELTER_STEP_M)
        idx = len(self.items)
        self.items.append({'ring': ring, 'axis': axis, 'clear': clear, 'tag': tag,
                           'bbox': bbox})
        c = self.CELL
        for i in range(int(bbox[0] // c), int(bbox[2] // c) + 1):
            for j in range(int(bbox[1] // c), int(bbox[3] // c) + 1):
                self.grid.setdefault((i, j), []).append(idx)

    def blocked(self, a, b, margin, skip=None):
        """Is the cross-section a-b inside the clean strip and clear of everything, by
        each thing's own clearance plus `margin`?"""
        if playable_sdf(*a) > -EDGE_CLEAR_M or playable_sdf(*b) > -EDGE_CLEAR_M:
            return True
        sb = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        c = self.CELL
        seen = set()
        for i in range(int(sb[0] // c), int(sb[2] // c) + 1):
            for j in range(int(sb[1] // c), int(sb[3] // c) + 1):
                for k in self.grid.get((i, j), ()):
                    if k in seen:
                        continue
                    seen.add(k)
                    o = self.items[k]
                    if o['tag'] == skip or boxes_apart(sb, o['bbox']):
                        continue
                    need = o['clear'] + margin
                    box = (sb[0] - need, sb[1] - need, sb[2] + need, sb[3] + need)
                    if o['ring'] is not None:
                        d = seg_ring_dist(a, b, o['ring'], box)
                    else:
                        d = seg_polyline_dist(a, b, o['axis'], need, box)
                    if d < need:
                        return True
        return False


def _static_obstacles():
    obs = _Obstacles()
    for p in PADS:
        obs.add(ring=simplify_polyline(p['ring'], 0.05), clear=SHELTER_CLEAR_M, tag=p['id'])
    for c in CORRIDORS:
        obs.add(axis=c['axis'], clear=c['half_width_m'] + ROADSIDE_SETBACK_M, tag=c['id'])
    for w in WATER:
        if w.get('ring'):
            obs.add(ring=simplify_polyline(w['ring'], 0.2), clear=SHELTER_WATER_CLEAR_M,
                    tag=w['id'])
    for a in AREAS:
        if a.get('tags', {}).get('natural') == 'wood':
            obs.add(ring=simplify_polyline(a['ring'], 0.2), clear=SHELTER_CLEAR_M, tag=a['id'])
    return obs


def _runs(flags):
    runs, start = [], None
    for i, ok in enumerate(flags):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _walk(section, n, obstacles, skip):
    """Walk a line of `n` stations; `section(s)` gives the cross-section at a station
    parameter (integers are stations, fractions lie between them). Returns runs as
    (s0, s1) in that parameter, each end bisected to exactly the clearance."""
    margin = SHELTER_STEP_M / 2.0
    flags = [not obstacles.blocked(*section(float(k)), margin, skip) for k in range(n)]
    out = []
    for i0, i1 in _runs(flags):
        s0, s1 = float(i0), float(i1)
        if i0 > 0:
            lo, hi = float(i0 - 1), s0          # blocked at lo, clear at hi
            for _ in range(12):
                mid = (lo + hi) / 2.0
                if obstacles.blocked(*section(mid), 0.0, skip):
                    lo = mid
                else:
                    hi = mid
            s0 = hi
        if i1 < n - 1:
            lo, hi = s1, float(i1 + 1)          # clear at lo, blocked at hi
            for _ in range(12):
                mid = (lo + hi) / 2.0
                if obstacles.blocked(*section(mid), 0.0, skip):
                    hi = mid
                else:
                    lo = mid
            s1 = lo
        out.append((s0, s1))
    # Two runs with nothing but clear ground between them are one run. The walk flags a
    # station with SHELTER_STEP_M / 2 of margin, and a road *ending* at the primary
    # just across from the belt clears the setback by less than that margin: the
    # station over it reads blocked, both runs are bisected up to it, and a 10 m gap is
    # left in a belt with no obstacle in it. The gap is walked again at the fine step
    # with no margin, and closed if it is clear the whole way.
    merged = []
    for s0, s1 in out:
        if merged and all(not obstacles.blocked(*section(merged[-1][1] + (s0 - merged[-1][1]) * k / SHELTER_MERGE_STEPS), 0.0, skip)
                          for k in range(1, SHELTER_MERGE_STEPS)):
            merged[-1] = (merged[-1][0], s1)
        else:
            merged.append((s0, s1))
    return merged


# --- along the primaries ------------------------------------------------------------
def _dense_with_normals(axis):
    """The axis densified to SHELTER_STEP_M, with a unit normal at every station and
    the mitre factor that keeps an offset along that normal a true distance off both
    adjacent segments: on the bisector at a bend the offset is foreshortened by the
    cosine of half the turn, and without the factor a belt on the inside of a 20 degree
    bend sat 13.8 m off a road it was meant to clear by 14."""
    # Not `densify`: that resamples at a uniform step and drops the bends, and a belt
    # drawn through the chords of a curve cuts inside its own offset at every one.
    dense = [axis[0]]
    for i in range(len(axis) - 1):
        a, b = axis[i], axis[i + 1]
        k = max(1, int(math.ceil(math.dist(a, b) / SHELTER_STEP_M)))
        dense += [(a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k)
                  for j in range(1, k + 1)]
    dense = [q for i, q in enumerate(dense) if i == 0 or math.dist(q, dense[i - 1]) > 1e-9]
    n = len(dense)
    seg = []                                   # unit normal of each segment
    for i in range(n - 1):
        dx, dy = dense[i + 1][0] - dense[i][0], dense[i + 1][1] - dense[i][1]
        ll = math.hypot(dx, dy) or 1.0
        seg.append((-dy / ll, dx / ll))
    normals, mitre = [], []
    for i in range(n):
        # The bisector of the two segment normals, not the normal of the chord between
        # the neighbours: those agree only when the two segments are the same length,
        # and on a curve drawn with short segments they are not.
        p, q = seg[max(0, i - 1)], seg[min(n - 2, i)]
        nx, ny = p[0] + q[0], p[1] + q[1]
        ll = math.hypot(nx, ny) or 1.0
        nv = (nx / ll, ny / ll)
        normals.append(nv)
        cos_half = abs(nv[0] * q[0] + nv[1] * q[1])
        mitre.append(1.0 / max(cos_half, 0.5))
    return dense, normals, mitre


def _at(dense, normals, mitre, s):
    """Point, normal and mitre factor at fractional station `s`."""
    i = int(s)
    if i >= len(dense) - 1:
        return dense[-1], normals[-1], mitre[-1]
    f = s - i
    p, q = dense[i], dense[i + 1]
    n0, n1 = normals[i], normals[i + 1]
    nx, ny = n0[0] + f * (n1[0] - n0[0]), n0[1] + f * (n1[1] - n0[1])
    ll = math.hypot(nx, ny) or 1.0
    return ((p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])), (nx / ll, ny / ll),
            mitre[i] + f * (mitre[i + 1] - mitre[i]))


def _primary_belt_runs(road, side, obstacles):
    """The stretches along one side of a primary a belt can stand on, as rings."""
    inner = road['half_width_m'] + ROADSIDE_SETBACK_M
    outer = inner + SHELTER_W_M
    dense, normals, mitre = _dense_with_normals(road['axis'])

    def at(s, d):
        p, nv, m = _at(dense, normals, mitre, s)
        return (p[0] + side * nv[0] * d * m, p[1] + side * nv[1] * d * m)

    def section(s):
        return at(s, inner), at(s, outer)

    # The ring is drawn through the very sections the walk tested, so what was measured
    # clear is what is drawn: an offset built afresh from the run's own polyline takes
    # its end normals one-sided and lands up to a metre off the section at a bend.
    out = []
    for s0, s1 in _walk(section, len(dense), obstacles, road['id']):
        stations = [s0] + [float(k) for k in range(int(math.ceil(s0)), int(s1) + 1)
                           if s0 + 1e-9 < k < s1 - 1e-9] + [s1]
        axis = simplify_polyline([at(s, inner + SHELTER_W_M / 2.0) for s in stations], 0.02)
        if polyline_length(axis) < SHELTER_MIN_LEN_M:
            continue
        lo = [at(s, inner) for s in stations]
        hi = [at(s, outer) for s in stations]
        ring = simplify_polyline(close_ring(lo + hi[::-1]), 0.02)
        # A belt is an offset of the road, and an offset past the bend's radius of
        # curvature folds through itself: the outer edge of a 50 m belt round a
        # tight bend comes back over the inner one. A folded ring is not a belt.
        if not ring_is_simple(ring):
            continue
        out.append({'ring': ring, 'axis': axis,
                    'road': road['id'], 'side': side, 'inner': inner,
                    'road_axis': simplify_polyline(
                        [_at(dense, normals, mitre, s)[0] for s in stations], 0.02)})
    return out


# --- between the fields -------------------------------------------------------------
def _field_edges(field):
    """(orient, coord, lo, hi, side) for every long axis-aligned edge of a field, with
    `side` +1 if the field lies on the greater-coordinate side of it (east or south)."""
    s = simplify_polyline(field['ring'], 0.5)
    out = []
    for i in range(len(s) - 1):
        a, b = s[i], s[i + 1]
        if abs(a[0] - b[0]) < 0.5 and abs(a[1] - b[1]) >= SHELTER_EDGE_MIN_M:
            x = (a[0] + b[0]) / 2.0
            lo, hi = sorted((a[1], b[1]))
            side = 1 if point_in_ring((x + 1.0, (lo + hi) / 2.0), field['ring']) else -1
            out.append(('v', x, lo, hi, side))
        elif abs(a[1] - b[1]) < 0.5 and abs(a[0] - b[0]) >= SHELTER_EDGE_MIN_M:
            y = (a[1] + b[1]) / 2.0
            lo, hi = sorted((a[0], b[0]))
            side = 1 if point_in_ring(((lo + hi) / 2.0, y + 1.0), field['ring']) else -1
            out.append(('h', y, lo, hi, side))
    return out


def _road_in_gap(o, mid, lo, hi, half_gap):
    """The corridor running down this gap, if one does: on the midline at both ends
    and the middle of the stretch."""
    for c in CORRIDORS:
        for s in (lo, (lo + hi) / 2.0, hi):
            p = (mid, s) if o == 'v' else (s, mid)
            if dist_to_polyline(p, c['axis']) > half_gap:
                break
        else:
            return c
    return None


def _boundary_lines(fields):
    """Every stretch of ground two fields face each other across, merged along each
    line: (orient, gap midline, lo, hi, host road or None). Where a road runs in the
    gap the belt will stand beside it; the line carries the road so the caller can."""
    edges = []
    for f in fields:
        edges += _field_edges(f)
    lines = []
    for o in ('v', 'h'):
        near = [e for e in edges if e[0] == o and e[4] == -1]     # field on the low side
        far = sorted((e for e in edges if e[0] == o and e[4] == +1), key=lambda e: e[1])
        cands = []
        for e in near:
            for f in far:
                gap = f[1] - e[1]
                if gap <= 0.0 or gap > SHELTER_GAP_MAX_M:
                    continue
                lo, hi = max(e[2], f[2]), min(e[3], f[3])
                if hi - lo < 1.0:
                    continue
                mid = (e[1] + f[1]) / 2.0
                road = _road_in_gap(o, mid, lo, hi, gap / 2.0)
                if road is not None and road['id'] in SHELTER_NO_BELT_ROADS:
                    continue
                cands.append((mid, lo, hi, None if road is None else road['id']))
        cands.sort(key=lambda c: (c[0], c[1]))
        i = 0
        while i < len(cands):
            j = i
            while j + 1 < len(cands) and cands[j + 1][0] - cands[i][0] <= 2.0:
                j += 1
            group = cands[i:j + 1]
            mid = sum(g[0] for g in group) / len(group)
            cur = None
            for lo, hi, rid in sorted((g[1], g[2], g[3]) for g in group):
                if cur is not None and lo <= cur[1] + SHELTER_GAP_MAX_M and rid == cur[2]:
                    cur[1] = max(cur[1], hi)
                else:
                    if cur is not None:
                        lines.append((o, mid, cur[0], cur[1], cur[2]))
                    cur = [lo, hi, rid]
            lines.append((o, mid, cur[0], cur[1], cur[2]))
            i = j + 1
    return lines


def _field_belt_runs(o, coord, lo, hi, host, obstacles):
    """The stretches of one boundary line a belt can stand on, as rings."""
    half = SHELTER_W_M / 2.0
    n = max(2, int(math.ceil((hi - lo) / SHELTER_STEP_M)) + 1)

    def station(s):
        return lo + (hi - lo) * s / (n - 1)

    def section(s):
        at = station(s)
        return ((coord - half, at), (coord + half, at)) if o == 'v' \
            else ((at, coord - half), (at, coord + half))

    out = []
    for s0, s1 in _walk(section, n, obstacles, host):
        a0, a1 = station(s0), station(s1)
        if a1 - a0 < SHELTER_MIN_LEN_M:
            continue
        if o == 'v':
            ring, axis = rect_ring(coord - half, a0, coord + half, a1), [(coord, a0), (coord, a1)]
        else:
            ring, axis = rect_ring(a0, coord - half, a1, coord + half), [(a0, coord), (a1, coord)]
        out.append({'ring': ring, 'axis': axis, 'road': host, 'side': 0, 'inner': None})
    return out


# --- trimming the fields back -------------------------------------------------------
def _belt_boxes(belt, clear):
    """The convex boxes a field has to stay out of for this belt, grown by `clear`:
    one for a straight belt, one per straight stretch of a belt along a bend."""
    if belt['side'] == 0:
        return [rect_ring(*ring_bbox(belt['ring'], clear))]
    ax, inner, side = belt['road_axis'], belt['inner'], belt['side']
    d0, d1 = inner - clear, inner + SHELTER_W_M + clear
    reach = clear + SHELTER_W_M          # room for the mitre at a bend
    boxes = []
    for i in range(len(ax) - 1):
        a, b = ax[i], ax[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ll = math.hypot(dx, dy) or 1.0
        ux, uy = dx / ll, dy / ll
        nx, ny = -uy * side, ux * side
        p0 = (a[0] - ux * reach, a[1] - uy * reach)
        p1 = (b[0] + ux * reach, b[1] + uy * reach)
        boxes.append([(p0[0] + nx * d0, p0[1] + ny * d0), (p1[0] + nx * d0, p1[1] + ny * d0),
                      (p1[0] + nx * d1, p1[1] + ny * d1), (p0[0] + nx * d1, p0[1] + ny * d1),
                      (p0[0] + nx * d0, p0[1] + ny * d0)])
    return boxes


def _clear_of(poly, box_shrunk):
    return not poly or not box_shrunk or not rings_overlap(poly + [poly[0]], box_shrunk)


def _trim_poly(poly, box):
    """Cut the least off an open polygon that keeps it out of a convex box: one
    half-plane of the box if one will do, two if not. None if nothing is left."""
    shrunk = shrink_convex(box, 0.01)
    if _clear_of(poly, shrunk):
        return poly, False
    hps = convex_halfplanes(box)
    best = None
    for q, n in hps:
        cand = clip_halfplane(poly, q, n)
        if len(cand) >= 3 and _clear_of(cand, shrunk):
            a = poly_area_m2(cand)
            if best is None or a > best[0]:
                best = (a, cand)
    if best is None:
        for i in range(len(hps)):
            for j in range(i + 1, len(hps)):
                cand = clip_halfplane(clip_halfplane(poly, *hps[i]), *hps[j])
                if len(cand) >= 3 and _clear_of(cand, shrunk):
                    a = poly_area_m2(cand)
                    if best is None or a > best[0]:
                        best = (a, cand)
    return (None, True) if best is None else (best[1], True)


def _trim_ring(ring, belts, clear):
    """A field ring held `clear` off every belt in `belts`: (ring, touched)."""
    poly = ring[:-1]
    fb = ring_bbox(ring)
    touched = False
    for belt in belts:
        if boxes_apart(fb, ring_bbox(belt['ring'], clear + 0.1)):
            continue
        for box in _belt_boxes(belt, clear):
            if boxes_apart(ring_bbox(poly), ring_bbox(box)):
                continue
            poly, cut = _trim_poly(poly, box)
            touched = touched or cut
            if poly is None:
                return None, True
    if not touched:
        return ring, False
    return fillet_ring(simplify_polyline(poly + [poly[0]], 0.05), FIELD_CORNER_R_M), True


def field_floors_ok(ring):
    """Is this still a field? Judged on what reaches the map - the ring clipped to the
    clean strip - against FIELD_MIN_HA and FIELD_MIN_SIDE_M."""
    if ring is None:
        return False
    m = EDGE_CLEAR_M
    r = clip_ring_to_rect(ring, m, m, PLAYABLE_M - m, PLAYABLE_M - m)
    if len(r) < 4 or ring_area_ha(r) < FIELD_MIN_HA:
        return False
    x0, y0, x1, y1 = ring_bbox(r)
    return min(x1 - x0, y1 - y0) >= FIELD_MIN_SIDE_M


def _trim_fields(fields, belts, commit):
    """Trim every field off `belts`. With `commit` the rings are replaced; without it
    the fields that would go under the floors are returned instead."""
    failed = []
    if not belts:
        return failed
    reach = [ring_bbox(b['ring'], SHELTER_CLEAR_M + 0.1) for b in belts]
    near = (min(r[0] for r in reach), min(r[1] for r in reach),
            max(r[2] for r in reach), max(r[3] for r in reach))
    for f in fields:
        if boxes_apart(near, ring_bbox(f['ring'])):
            continue
        ring, touched = _trim_ring(f['ring'], belts, SHELTER_CLEAR_M)
        if not touched:
            continue
        if not field_floors_ok(ring):
            failed.append(f)
            continue
        if commit:
            f['ring'] = ring
    return failed


def _lay_candidates(fields, obstacles, lay):
    """Every belt the map could carry, laid in order onto `obstacles` through `lay`.

    Order is the whole of the crossing rule: primaries first, then the north-south
    field belts, then the east-west, each stopping at what came before. The fields are
    trimmed as it goes, so a later belt is judged against the ground the earlier ones
    left; pass a scratch copy of the fields to keep that from reaching the map.
    """
    # Along the primaries, both sides. A field the belt would ruin becomes an obstacle
    # and the belt is laid again around it.
    for road in [c for c in CORRIDORS if c['kind'] == 'primary']:
        for side in (-1, +1):
            source = ('primary', road['id'], side)
            for _ in range(4):
                runs = _primary_belt_runs(road, side, obstacles)
                failed = _trim_fields(fields, runs, commit=False)
                if not failed:
                    break
                for f in failed:
                    obstacles.add(ring=simplify_polyline(f['ring'], 0.05),
                                  clear=SHELTER_CLEAR_M, tag=f['id'])
            _trim_fields(fields, runs, commit=True)
            lay(runs, source)

    # Between the fields: north-south lines first, then east-west. Beside a road the
    # windward side is tried first, the other side if the windward field would go under
    # the floor, and the line is left alone if neither will do.
    lines = _boundary_lines(fields)
    for orient in ('v', 'h'):
        for o, mid, lo, hi, host in sorted(l for l in lines if l[0] == orient):
            if host is None:
                choices = [mid]
            else:
                road = corridor_by_id(host)
                at = sum(p[0] if o == 'v' else p[1] for p in road['axis']) / len(road['axis'])
                off = road['half_width_m'] + ROADSIDE_SETBACK_M + SHELTER_W_M / 2.0
                choices = [at + SHELTER_ROAD_SIDE * off, at - SHELTER_ROAD_SIDE * off]
            for coord in choices:
                runs = _field_belt_runs(o, coord, lo, hi, host, obstacles)
                if not runs:
                    continue
                if _trim_fields(fields, runs, commit=False):
                    continue
                _trim_fields(fields, runs, commit=True)
                lay(runs, ('field', o, coord, lo, hi, host))
                break


def _runs_from(source, obstacles):
    if source[0] == 'primary':
        return _primary_belt_runs(corridor_by_id(source[1]), source[2], obstacles)
    return _field_belt_runs(*source[1:], obstacles)


def build_shelterbelts(fields):
    """The shelterbelts on the map, as `AREAS` records - and the fields cut back to
    make room for them, in place.

    Two passes. The first lays every belt the map could carry on a scratch copy of the
    fields, which is the only way to know what there is to choose from; the
    SHELTER_MAX_COUNT longest are kept, longest first because a belt shelters the
    frontage it runs along and nothing else. The second pass lays only those, in the
    original order, so each is traced again against the belts actually kept rather
    than stopping short of one that was dropped - and only then are the real fields
    cut, so no field gives ground to a belt that is not there.
    """
    scratch = [{'id': f['id'], 'name': f['name'], 'ring': f['ring']} for f in fields]
    cands = []

    def note(runs, source):
        for r in runs:
            obstacles.add(ring=r['ring'], clear=SHELTER_JOIN_M, tag='belt')
            cands.append((len(cands), source, r))

    obstacles = _static_obstacles()
    _lay_candidates(scratch, obstacles, note)
    keep = sorted(cands, key=lambda c: -polyline_length(c[2]['axis']))[:SHELTER_MAX_COUNT]
    keep.sort(key=lambda c: c[0])

    obstacles = _static_obstacles()
    belts = []
    for _, source, was in keep:
        centre = was['axis'][len(was['axis']) // 2]
        run = None
        for _ in range(4):
            runs = _runs_from(source, obstacles)
            if not runs:
                run = None
                break
            run = min(runs, key=lambda r: dist_to_polyline(centre, r['axis']))
            failed = _trim_fields(fields, [run], commit=False)
            if not failed:
                break
            for f in failed:
                obstacles.add(ring=simplify_polyline(f['ring'], 0.05),
                              clear=SHELTER_CLEAR_M, tag=f['id'])
        else:
            # Four attempts and the last one still fails a field's floors: a belt
            # that is not laid, rather than one laid over the failure.
            run = None
        if run is None:
            continue
        _trim_fields(fields, [run], commit=True)
        obstacles.add(ring=run['ring'], clear=SHELTER_JOIN_M, tag='belt')
        belts.append(run)

    out = []
    for k, b in enumerate(belts):
        out.append({'id': f'belt_{k + 1:03d}', 'kind': 'shelterbelt',
                    'name': f'Rompevientos {k + 1:03d}', 'ring': b['ring'],
                    'axis': b['axis'], 'road': b['road'], 'side': b['side'],
                    'area_ha': ring_area_ha(b['ring']), 'tags': wood_tags(SHELTER_LEAF_TYPE)})
    return out


def _validate_shelterbelts(bad):
    """Every rule a belt has to meet, measured on the drawn rings."""
    belts = [a for a in AREAS if a.get('kind') == 'shelterbelt']
    if not belts:
        return
    if len(belts) > SHELTER_MAX_COUNT:
        bad.append(f"{len(belts)} shelterbelts, over the cap of {SHELTER_MAX_COUNT}")
    fields = [a for a in AREAS if a.get('kind') == 'farmland']
    woods = [a for a in AREAS if a.get('kind') == 'wood']
    tol = 0.05
    for b in belts:
        ring = b['ring']
        if len(ring) < 4 or math.dist(ring[0], ring[-1]) > 1e-6:
            bad.append(f"{b['id']}: ring does not close on its first point")
            continue
        if not ring_is_simple(ring):
            bad.append(f"{b['id']}: ring crosses itself")
        if b['tags'] != wood_tags(SHELTER_LEAF_TYPE):
            bad.append(f"{b['id']}: tagged {b['tags']}, not a {SHELTER_LEAF_TYPE} wood")
        if max(playable_sdf(x, y) for x, y in ring) > -EDGE_CLEAR_M + tol:
            bad.append(f"{b['id']}: inside the {EDGE_CLEAR_M:.0f} m clean strip")
        length = polyline_length(b['axis'])
        if length < SHELTER_MIN_LEN_M - tol:
            bad.append(f"{b['id']}: {length:.0f} m long, under {SHELTER_MIN_LEN_M:.0f}")
        width = ring_area_ha(ring) * 10000.0 / length if length else 0.0
        if abs(width - SHELTER_W_M) > 0.5:
            bad.append(f"{b['id']}: {width:.1f} m across, not {SHELTER_W_M:.0f}")
        bb = ring_bbox(ring, SHELTER_WATER_CLEAR_M + 1.0)
        for f in fields:
            if boxes_apart(bb, ring_bbox(f['ring'])):
                continue
            d = ring_ring_dist(ring, f['ring'], SHELTER_CLEAR_M)
            if d < SHELTER_CLEAR_M - tol:
                bad.append(f"{b['id']}: {d:.1f} m from field {f['name']}, "
                           f"under {SHELTER_CLEAR_M:.0f}")
        for p in PADS:
            if boxes_apart(bb, ring_bbox(p['ring'])):
                continue
            d = ring_ring_dist(ring, p['ring'], SHELTER_CLEAR_M)
            if d < SHELTER_CLEAR_M - tol:
                bad.append(f"{b['id']}: {d:.1f} m from {p['name']}, under {SHELTER_CLEAR_M:.0f}")
        for w in woods:
            if boxes_apart(bb, ring_bbox(w['ring'])):
                continue
            d = ring_ring_dist(ring, w['ring'], SHELTER_CLEAR_M)
            if d < SHELTER_CLEAR_M - tol:
                bad.append(f"{b['id']}: {d:.1f} m from wood {w['name']}, under {SHELTER_CLEAR_M:.0f}")
        for w in WATER:
            if not w.get('ring') or boxes_apart(bb, ring_bbox(w['ring'])):
                continue
            d = ring_ring_dist(ring, w['ring'], SHELTER_WATER_CLEAR_M)
            if d < SHELTER_WATER_CLEAR_M - tol:
                bad.append(f"{b['id']}: {d:.1f} m from {w['name']}, under {SHELTER_WATER_CLEAR_M:.0f}")
        for c in CORRIDORS:
            need = c['half_width_m'] + ROADSIDE_SETBACK_M
            if boxes_apart(bb, ring_bbox(c['axis'])):
                continue
            d = min(seg_polyline_dist(ring[i], ring[i + 1], c['axis'], need)
                    for i in range(len(ring) - 1))
            if d < need - tol:
                bad.append(f"{b['id']}: {d:.1f} m from {c['name']}, under {need:.1f}")
    for i in range(len(belts)):
        for j in range(i + 1, len(belts)):
            if boxes_apart(ring_bbox(belts[i]['ring']), ring_bbox(belts[j]['ring'])):
                continue
            if rings_overlap(belts[i]['ring'], belts[j]['ring']):
                bad.append(f"{belts[i]['id']} and {belts[j]['id']} overlap")
    for f in fields:
        if not ring_is_simple(f['ring']):
            bad.append(f"{f['id']}: field ring crosses itself")
        if not field_floors_ok(f['ring']):
            bad.append(f"{f['id']}: under {FIELD_MIN_HA:.0f} ha or {FIELD_MIN_SIDE_M:.0f} m "
                       f"across after trimming")


# --- the timber along the water ----------------------------------------------------
# Bosque de ribera: the trees along the river and round the lake. It is the one planting
# on this map that is not placed on the survey - it follows the water, which is the only
# thing here that does not run on a section line - and it is the timber this country
# actually has. The uplands were prairie and were ploughed; what was left standing was
# the ground along the water that nobody could plough, and that is exactly the ground
# this fills: from GALLERY_SETBACK_M off the drawn waterline out across the valley side,
# stopping GALLERY_CLEAR_M short of the nearest field.
#
# **How it is built is the whole of it, because the obvious construction does not work.**
# A band along a curve is an offset of that curve, and the first entry in the list of
# things that have gone wrong here is that offsetting a polyline by more than its radius
# of curvature folds the ring through itself. These meanders bend to 255 m and the far
# edge of this wood is 485 m out: the offset at that distance is not even monotone in
# northing any more, which is measurable in one line and is what the first version of
# this - a 45 m strip that stopped at the top of the bank - was sized to avoid.
#
# So the outer edge is not an offset of the water at all. Every station on the waterline
# **casts a ray outwards** along its own normal and the wood runs as far as that ray gets
# before it meets something: a field, a yard, a road's clearance, the other body of
# water, the clean strip, or GALLERY_MAX_W_M of valley side, whichever comes first. The
# inner edge stays a true offset, which is safe because GALLERY_INNER_M is 60 m against a
# 255 m radius. What that buys is a shape whose width answers to what is actually beside
# it - wide where the fields stand back, pinched to nothing at a bridge - and a ring that
# cannot fold, because a ray never crosses its neighbour while the cast stays under the
# radius of curvature. `validate()` sweeps the drawn rings for crossings rather than
# taking that on trust.
#
# It is also why this is laid out **after** the fields and not before. The parcelling
# does not need to know the timber is there - it is inside the reserve the water already
# keeps the fields out of - but the timber is defined by where the fields stopped, so the
# order is: parcel the ground, then plant what the parcelling left along the water.
GALLERY_SETBACK_M = 15.0          # from the drawn waterline to the first tree
GALLERY_CLEAR_M = 15.0            # from the last tree to whatever stopped it
# How far up the valley side it may go. The valley reaches VALLEY_HALF_W_M from the
# waterline and the ground is till plain beyond that, so past this the wood would be
# standing on the flats rather than on the side - and the fields are the ones that are
# supposed to stop it, so this is only the backstop for a station that has no field
# opposite it at all.
GALLERY_MAX_W_M = VALLEY_HALF_W_M - GALLERY_SETBACK_M
GALLERY_MIN_W_M = 40.0            # thinner than this is a verge, not a wood
GALLERY_MIN_LEN_M = 250.0         # and shorter than this is a fragment between crossings
GALLERY_STEP_M = 20.0             # how finely the waterline is walked
# How much of the way to where two converging rays meet the wood is allowed to go, and
# how hard a panel that is not clear pulls its two rays back. Neither is a tuning knob so
# much as a margin: the first keeps the outer edge short of the point where it would
# cross itself, the second only has to converge.
GALLERY_MEET_FRAC = 0.80
GALLERY_RELAX_FRAC = 0.80
GALLERY_RELAX_PASSES = 16
# Riparian timber is hardwood - cottonwood, willow, silver maple - so it reads with the
# shelterbelts rather than with the planted conifer woodlots, and both renderers colour
# the two apart.
GALLERY_LEAF_TYPE = 'broadleaved'

GALLERY_INNER_M = RIVER_HALF_W_M + GALLERY_SETBACK_M       # 60 m off the river's axis


def _river_dense():
    """The axis at GALLERY_STEP_M, with the arc length at every vertex."""
    dense = densify(river_axis(), GALLERY_STEP_M)
    arc = [0.0]
    for i in range(1, len(dense)):
        arc.append(arc[-1] + math.dist(dense[i - 1], dense[i]))
    return dense, arc


_RIVER_AXIS = river_axis()
_RIVER_YS = [q[1] for q in _RIVER_AXIS]
_LAKE_RING = lake_ring()
_WATER_BAND_M = GALLERY_MAX_W_M + RIVER_HALF_W_M + GALLERY_SETBACK_M
_LAKE_BOX = (min(q[0] for q in _LAKE_RING), min(q[1] for q in _LAKE_RING),
             max(q[0] for q in _LAKE_RING), max(q[1] for q in _LAKE_RING))


def gallery_blocks(planted, fields):
    """Everything a ray stops at, as rectangles already grown by GALLERY_CLEAR_M.

    A road goes in as the bounding box of its axis grown by the clearance it keeps, which
    for an axis-aligned alignment is exactly its own reserve and a little over at the two
    ends. That is the right direction to be wrong in: it stops the wood short of a road
    rather than letting it onto one, and the ends of an alignment are where a bridge
    abutment or a stub's turning head is anyway.
    """
    g = GALLERY_CLEAR_M
    out = []
    for p in PADS:
        cx, cy = p['centre']
        w, h = p['size']
        out.append((cx - w / 2.0 - g, cy - h / 2.0 - g,
                    cx + w / 2.0 + g, cy + h / 2.0 + g))
    for a in list(planted) + list(fields):
        xs = [q[0] for q in a['ring']]
        ys = [q[1] for q in a['ring']]
        out.append((min(xs) - g, min(ys) - g, max(xs) + g, max(ys) + g))
    for c in CORRIDORS:
        keep = c['half_width_m'] + ROADSIDE_SETBACK_M
        xs = [q[0] for q in c['axis']]
        ys = [q[1] for q in c['axis']]
        out.append((min(xs) - keep, min(ys) - keep, max(xs) + keep, max(ys) + keep))
    m = EDGE_CLEAR_M
    far = 10.0 * PLAYABLE_M
    out += [(-far, -far, m, far), (PLAYABLE_M - m, -far, far, far),
            (-far, -far, far, m), (-far, PLAYABLE_M - m, far, far)]
    return out


def _cast(p, n, limit, rects):
    """How far from `p` along the unit vector `n` the wood may run before it meets one of
    `rects`. The slab test, which is exact for an axis-aligned box and is the reason
    everything above was turned into one."""
    t = limit
    bx0, bx1 = min(p[0], p[0] + limit * n[0]), max(p[0], p[0] + limit * n[0])
    by0, by1 = min(p[1], p[1] + limit * n[1]), max(p[1], p[1] + limit * n[1])
    for x0, y0, x1, y1 in rects:
        # The ray's own box first. The slab test below is exact and costs twenty
        # operations; four comparisons throw out the several hundred rectangles on the
        # map that are nowhere near this station.
        if x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1:
            continue
        lo, hi = 0.0, t
        for k, (a, b) in enumerate(((x0, x1), (y0, y1))):
            if abs(n[k]) < 1e-12:
                if not (a <= p[k] <= b):
                    lo = 1e18
                    break
            else:
                u, v = (a - p[k]) / n[k], (b - p[k]) / n[k]
                if u > v:
                    u, v = v, u
                lo, hi = max(lo, u), min(hi, v)
        if lo <= hi and lo < t:
            t = max(0.0, lo)
    return t


def water_dist(q, cutoff=None):
    """Distance from `q` to the nearest drawn waterline, 0 inside the water.

    `cutoff` is how far the caller still cares: anything further comes back as at least
    that, which is what lets the search be sliced. The river is a single-valued function
    of northing, so its axis is sorted in y and only the stretch inside `q`'s own band of
    northing can be within the cutoff - a bisect with a vertex of margin either end,
    exact here and *not* exact on a watercourse that doubled back. The lake is skipped
    outright unless `q` is inside its box grown by the same. Without both of those this
    walks 322 river vertices and 97 lake ones for every sample of every ray, which was
    most of what the timber cost to lay out.
    """
    band = _WATER_BAND_M if cutoff is None else cutoff + RIVER_HALF_W_M
    lo = bisect.bisect_left(_RIVER_YS, q[1] - band)
    hi = bisect.bisect_right(_RIVER_YS, q[1] + band)
    seg = _RIVER_AXIS[max(0, lo - 1):hi + 1]
    d = (dist_to_polyline(q, seg) - RIVER_HALF_W_M) if len(seg) > 1 else 1.0e18
    if _LAKE_BOX[0] - band <= q[0] <= _LAKE_BOX[2] + band \
            and _LAKE_BOX[1] - band <= q[1] <= _LAKE_BOX[3] + band:
        if point_in_ring(q, _LAKE_RING):
            return 0.0
        d = min(d, min(math.dist(q, r) for r in _LAKE_RING))
    return max(0.0, d)


def _cast_medial(p, n, t, s):
    """Pull a cast back where it reaches ground that belongs to a different piece of
    waterline, which is the rule that keeps one stand of timber out of another.

    A ray starts `s` off its own waterline and walks away from it, so `s + d` is how far
    the point at `d` stands from the water it belongs to. The moment some *other* piece of
    water is nearer than that, the ground under the ray is nearer that water instead, and
    the stand growing out of it is the one that owns it. Stopping there is the medial
    axis between the two, and it is one statement that does three jobs: it keeps the two
    banks of a meander neck from growing into each other, it keeps the lake's timber and
    the river's apart where the river runs in, and it keeps any ray out of open water,
    because water is at distance zero and zero is under `s + d` for every d.

    Sampled rather than solved. The step is the one the waterline itself is walked at, so
    nothing wider than a station can be stepped over, and the metre of slack is for the
    curvature the `s + d` reading ignores on the inside of a bend.
    """
    d = GALLERY_STEP_M
    while d <= t:
        q = (p[0] + d * n[0], p[1] + d * n[1])
        if water_dist(q, s + d) < s + d - 1.0:
            return max(0.0, d - GALLERY_STEP_M)
        d += GALLERY_STEP_M
    return t


def _gallery_runs(width, closed):
    """Runs of consecutive stations wide enough to be a wood, as index spans."""
    ok = [w >= GALLERY_MIN_W_M for w in width]
    n = len(ok)
    if closed:
        if all(ok):
            return [(0, n - 1)]
        k = next(i for i in range(n) if not ok[i])
        order = [(k + i) % n for i in range(n)]
    else:
        order = list(range(n))
    runs, run = [], []
    for i in order:
        if ok[i]:
            run.append(i)
        elif run:
            runs.append((run[0], run[-1]))
            run = []
    if run:
        runs.append((run[0], run[-1]))
    return runs


def _gallery_ring(inner, normal, width, i0, i1, closed):
    """Out along the ray at one end, down the outer edge, back along the waterline."""
    n = len(inner)
    idx = [(i0 + k) % n for k in range((i1 - i0) % n + 1)] if closed \
        else list(range(i0, i1 + 1))
    edge = [(inner[i][0] + width[i] * normal[i][0],
             inner[i][1] + width[i] * normal[i][1]) for i in idx]
    return close_ring([inner[i] for i in idx] + edge[::-1])


def _ray_meet(p, n, q, m):
    """Where two rays converge, as a distance along the first, or infinity if they do not.

    Rays cast off the inside of a bend point at each other and meet at the centre of
    curvature. A wood drawn out past that point has an outer edge that crosses itself, so
    this is the distance the pair of them has to stop short of, and it is solved rather
    than estimated from a radius: two lines, one determinant.
    """
    den = n[0] * m[1] - n[1] * m[0]
    if abs(den) < 1e-12:
        return float('inf')
    t = ((q[0] - p[0]) * m[1] - (q[1] - p[1]) * m[0]) / den
    return t if t > 0.0 else float('inf')


def _gallery_side(inner, normal, blocks, closed, tag, name, out):
    """Cast from every station, pull the casts back until what is drawn between them is
    as clear as they are, take the runs that are wide enough, draw them."""
    n = len(inner)
    width = [_cast_medial(p, nv, _cast(p, nv, GALLERY_MAX_W_M, blocks),
                          GALLERY_SETBACK_M)
             for p, nv in zip(inner, normal)]
    pairs = [(i, (i + 1) % n) for i in range(n if closed else n - 1)]
    # Converging rays first, because that failure is in the pair and not in the ground.
    #
    # Every pair, and not the next ray along or a window of frontage either side of it:
    # the pair that had one stand still crossing itself was two rays 1020 m apart along
    # the river and 733 m apart on the ground, on the two arms of a meander that comes
    # back on itself, meeting 468 m out where both wanted 485. Frontage says nothing
    # about that. What does is how far apart the two stations *stand*, and no two rays
    # capped at GALLERY_MAX_W_M can meet at all unless that is under twice it - which is
    # the pre-filter, and is why this stays a sweep rather than a cost.
    span = 2.0 * GALLERY_MAX_W_M
    for i in range(n):
        px, py = inner[i]
        for j in range(i + 1, n):
            qx, qy = inner[j]
            if abs(qx - px) > span or abs(qy - py) > span:
                continue
            t = _ray_meet(inner[i], normal[i], inner[j], normal[j])
            if t < float('inf'):
                width[i] = min(width[i], GALLERY_MEET_FRAC * t)
                width[j] = min(width[j], GALLERY_MEET_FRAC * t)
    # Then the ground. A ray is a line and what is drawn between two of them is a
    # quadrilateral, and a rectangle can be missed by both rays and still be clipped by
    # the panel that joins them - neither ray has a point in it and no corner of it is on
    # a ray, which is the crossing case a vertex test passes every time. So the panels
    # are tested, and the two casts behind one that is not clear come back together until
    # it is. It terminates because at zero width the panel is a segment of the waterline
    # offset, which is clear by construction.
    near = {}
    for i, j in pairs:
        x0 = min(inner[i][0], inner[j][0]) - GALLERY_MAX_W_M
        x1 = max(inner[i][0], inner[j][0]) + GALLERY_MAX_W_M
        y0 = min(inner[i][1], inner[j][1]) - GALLERY_MAX_W_M
        y1 = max(inner[i][1], inner[j][1]) + GALLERY_MAX_W_M
        near[i] = [b for b in blocks
                   if not (b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1)]
    for _ in range(GALLERY_RELAX_PASSES):
        moved = False
        for i, j in pairs:
            if width[i] < 1e-6 and width[j] < 1e-6:
                continue
            quad = close_ring([inner[i], inner[j],
                               (inner[j][0] + width[j] * normal[j][0],
                                inner[j][1] + width[j] * normal[j][1]),
                               (inner[i][0] + width[i] * normal[i][0],
                                inner[i][1] + width[i] * normal[i][1])])
            for b in near[i]:
                if rings_overlap(quad, rect_ring(*b)):
                    width[i] *= GALLERY_RELAX_FRAC
                    width[j] *= GALLERY_RELAX_FRAC
                    moved = True
                    break
        if not moved:
            break
    span = width
    for k, (i0, i1) in enumerate(_gallery_runs(span, closed)):
        idx = [(i0 + j) % n for j in range((i1 - i0) % n + 1)] if closed \
            else list(range(i0, i1 + 1))
        length = sum(math.dist(inner[idx[j]], inner[idx[j + 1]])
                     for j in range(len(idx) - 1))
        if length < GALLERY_MIN_LEN_M:
            continue
        r = _gallery_ring(inner, normal, span, i0, i1, closed)
        xs = [q[0] for q in r]
        ys = [q[1] for q in r]
        out.append({'id': f'{tag}_{k + 1}', 'name': f'{name} {k + 1}',
                    'reach': (idx[0], idx[-1]), 'length_m': length,
                    'width_m': (min(span[i] for i in idx),
                                max(span[i] for i in idx)),
                    'area_ha': ring_area_ha(r),
                    'centre': ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0),
                    'ring': r,
                    'tags': wood_tags(GALLERY_LEAF_TYPE)})


def gallery_areas(planted, fields):
    """Every stand of timber along the water: both banks of the river, and the lake."""
    blocks = gallery_blocks(planted, fields)
    dense, _ = _river_dense()
    lake = lake_ring()
    out = []
    for sgn, bank in ((-1.0, 'Oeste'), (+1.0, 'Este')):
        off = offset_polyline(dense, sgn * GALLERY_INNER_M)
        # Which sign of the offset is the east bank depends on which way the axis runs,
        # so it is read off the geometry rather than argued about.
        if (off[0][0] - dense[0][0]) * sgn < 0:
            off = offset_polyline(dense, -sgn * GALLERY_INNER_M)
        normal = [((q[0] - p[0]) / GALLERY_INNER_M, (q[1] - p[1]) / GALLERY_INNER_M)
                  for p, q in zip(dense, off)]
        _gallery_side(off, normal, blocks, False,
                      f'ribera_{"w" if sgn < 0 else "e"}',
                      f'Bosque de Ribera {bank}', out)
    # The lake. Its shore is walked as the same closed ring the water is drawn from, so
    # the setback is off the drawn shore and not off some finer sampling of the same
    # level set. The outward normal is taken from the ring itself: which way is out of a
    # lobed ellipse is not something to assume.
    pts = lake[:-1]
    inner, normal = [], []
    for i, p in enumerate(pts):
        a, b = pts[i - 1], pts[(i + 1) % len(pts)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ll = math.hypot(dx, dy) or 1.0
        nv = (-dy / ll, dx / ll)
        if point_in_ring((p[0] + nv[0], p[1] + nv[1]), lake):
            nv = (-nv[0], -nv[1])
        inner.append((p[0] + GALLERY_SETBACK_M * nv[0],
                      p[1] + GALLERY_SETBACK_M * nv[1]))
        normal.append(nv)
    _gallery_side(inner, normal, blocks, True,
                  'ribera_lago', 'Bosque de Ribera del Lago', out)
    return out



# --- the parcelling ---------------------------------------------------------------
# The fields, and they are laid out the way the ground was actually subdivided: by
# *aliquot* parts of a section. A section is a mile square - 259.0 ha - and the survey
# halves it and halves it again, so the parcel sizes are not chosen, they fall out:
#
#     a quarter section    804.7 m square    64.75 ha    ("a hundred and sixty acres")
#     a quarter-quarter    402.3 m square    16.19 ha    ("a forty")
#     a quarter of that    201.2 m square     4.05 ha    ("a ten")
#
# Those three land where the brief asked for large (20-100 ha) and medium (10-20 ha)
# exactly, and a hectare under it on the small (5 ha): a ten-acre parcel is 4.05 ha and
# there is no aliquot part of a section that is 5.00. Breaking the ladder to hit the
# round number would cost the one thing that makes this a survey rather than a grid -
# that every parcel nests inside the one above it - so it is not broken. What closes the
# gap instead is the merge below: two tens worked as one field come to 8.09 ha, so the
# ground round a town comes out between four and eight hectares with a mean of six, which
# is the five the brief asked for read as a class rather than as a number.
#
# The nesting is also what fills the map. A cell is offered at the size its position
# calls for; if anything is in the way - the water's valley, a road and its verge, a
# yard, a wood, the clean strip - it is quartered and the four children are offered
# instead, down to a ten. So open country comes out in quarter sections, the ground
# around a town in tens, and the awkward corners against a river valley or a shelterbelt
# fill with whatever aliquot fits. That is what real parcelling looks like, and it is
# one rule rather than a special case per obstacle.
#
# Size by distance to a town, which is the brief and is also how it works: land near a
# settlement is worth more, gets sold in smaller pieces and stays that way.
#
# **The count is a constraint and not an outcome**, and it is the one number here that
# is not derived from the survey: at most `FIELD_MAX_COUNT` fields on the map. That
# binds. The playable square is 6710 ha, the water's valley takes a fifth of it and the
# roads, the towns, the yards and the plantings take their own, so what is left to
# cultivate is about 3200 ha - which over 120 fields is a mean of 28 ha and means the
# country has to be worked in quarter sections and pairs of them, not in forties. Every
# band below was set by running the parcelling and counting, because the count falls out
# of the geometry rather than out of any one constant, and it is not even monotone in
# the obvious direction - merging harder cuts twenty fields and covers very nearly the
# same ground. There is no arithmetic short of building it that says so. `validate()`
# holds the cap, so the next feature added to the map cannot quietly push it over.
#
# What the ground costs to reach the cap is worth stating, because it is almost nothing:
# 150 laid out 3252 ha and 120 lays out 3189, a difference of 63 ha on 6710 - under one
# per cent of the map. Coverage is flat in every constant here and what they buy is how
# finely that ground is cut up, so the cap is a decision about field *size* and not
# about how much of the map is farmed. The two levers that were turned for it are
# `FIELD_MERGE_STEPS`, which joins parcels that are already flush, and
# `FIELD_SPLIT_GAIN`, which stops cutting cells that would go on splitting - in that
# order, because joining what the survey merely wrote down twice costs no ground at all
# and declining to split does.
FIELD_MAX_COUNT = 120
FIELD_SECTION_M = MILE_M
FIELD_ANCHOR = (ROAD_W_X, PLSS_EW_ANCHOR_M)   # the survey's own corner
FIELD_SPLITS = 3                  # how many halvings a side may take: S/8 = 201.2 m
# The two radii, and they are tight because the count is capped. A town platform is
# 516 x 510 m, so its own half-diagonal is 363 m: a 750 m band is the first ring or two
# of tens outside the kerb and no more, and a 1300 m band the forties beyond it. Set them
# at the 900 and 1800 m that read naturally on a map this size and it comes to well over
# 150 for *less* ground: the inner bands cut the same acres finer and the country loses
# the quarter sections that were paying for them. The bands are narrow because a field is
# a scarce thing here.
FIELD_SMALL_M = 750.0             # inside this of a town centre, parcels are tens
FIELD_MEDIUM_M = 1300.0           # ... and forties out to here; quarter sections beyond
FIELD_CLEAR_M = 10.0              # headland between a field and anything built
# And the headland between a field and the *next field*. The aliquot grid cuts flush -
# two quarter sections either side of a halving line share that line exactly - so without
# this every parcel the merge did not join touches its neighbour along a full edge, and
# there is nowhere to turn a machine round or drive from one to the other. Half of it
# comes off each side of every field, which is the only way the gap is 5 m whichever two
# fields are looked at: take the whole 5 off one side and it depends which of the pair
# was cut first, and a field with a neighbour on each side ends up 10 m short across.
#
# It is applied *after* the merge and not before, because the merge is what decides that
# two parcels are one field, and it does that by finding a whole shared edge. Inset first
# and there are no shared edges left, the merge stops joining anything at all, and the
# count goes back over two hundred. So the grid stays flush all the way through the
# planning and every rectangle comes in by 2.5 m at the end.
FIELD_GAP_M = 5.0

# How close cultivation comes to the water, and it is the full `VALLEY_HALF_W_M`: the
# fields stay out of the water's valley altogether and start where the ground is back on
# the till plain. That is the same clearance every yard, wood and shelterbelt on this map
# is already held off the water by, so the rule is now one rule rather than a general one
# with the fields exempted from it, and `validate()` reads the same for all of them.
#
# It is also the most expensive constant here. The valley is a kilometre across and runs
# the length of the map: cultivating down its side to the top of the bank - which the
# ground allows, the section falling 20 m over 440 m at under five degrees - is 901 ha,
# a fifth of everything the parcelling lays out. What that would buy is fields on a
# hillside, in a corridor whose whole purpose is that it is not the till plain. The
# ground is left as floodplain, which carries no tag: nothing is drawn on it and nothing
# is grown on it.
#
# Two numbers because the two bodies are measured from different things: the river's
# distance is to its *centreline* and the lake's to its *shore*, which is exactly the
# convention the yard and planting rules below already use.
FIELD_RIVER_CLEAR_M = VALLEY_HALF_W_M
FIELD_LAKE_CLEAR_M = VALLEY_HALF_W_M

# The aliquot ladder, in hectares, from halving a section alternately in each direction.
# Halves matter as much as quarters and were missing at first: with only square cells the
# far country came out in forties, because an 805 m square does not fit between a
# shelterbelt and a river valley while the 402 x 805 m half of one does. `N1/2 NE1/4` is
# as real a parcel as `NE1/4` and the brief allows a rectangle, so both are cut.
#
#     259.0  129.5  64.75  32.37  16.19  8.09  4.05 ha
#     section  1/2  1/4    1/8    1/16   1/32  1/64
FIELD_SECTION_HA = FIELD_SECTION_M ** 2 / 10000.0
FIELD_CAP_LARGE_HA = FIELD_SECTION_HA / 4.0        # 64.75, a quarter section
FIELD_CAP_MEDIUM_HA = FIELD_SECTION_HA / 16.0      # 16.19, a forty
FIELD_CAP_SMALL_HA = FIELD_SECTION_HA / 64.0       # 4.05, a ten
# What a merge may reach out in the country. The aliquot above a quarter section is a
# half at 129.5 ha, over the hundred the brief allows, so merging is the one step here
# that is *not* aliquot: two parcels that share a whole edge become one field, and a
# farmer working two fields as one does not consult the survey. The generating grid stays
# aliquot; this is what is done with it afterwards.
FIELD_MERGE_MAX_HA = 100.0
# How far a merge may climb over the band the survey sold in, in halvings. The bands say
# how the ground was *subdivided* - tens against a town, forties beyond, quarter sections
# in the country - and holding a merge to the same ceiling made the rule dead everywhere
# but the far country: a ten that merges is 8.09 ha, over the ten the band allows, so no
# two parcels near a town were ever joined and the near ground came out as ten-acre
# slivers.
#
# Two halvings - one aliquot square step - is what the 120-field cap wants, and it is
# the cheap half of paying for it: sixteen fields for *six hectares more* ground, because
# the union of two flush rectangles is exactly the two of them. It takes the ceiling on
# a near-town parcel up to 16.19 ha, which was the argument against it while the cap was
# 150: on paper there would be nothing small left. Measured, there is - the near-town
# band comes out 7.0 to 15.8 ha and averages 9.0 against the 29 and 31 of the forties
# and the country, so the three bands still read apart, which is what the worry was
# actually about. Three changes nothing: the merge is out of flush pairs by then.
#
# Drop it to 0 instead and the merge only ever joins the two halves of a quarter section
# again, and the count goes over two hundred for the same ground. One - what this was -
# reaches 122 at its hardest setting and cannot make the cap without throwing away
# 150 ha, which is the whole reason it moved.
FIELD_MERGE_STEPS = 2

# What a field is allowed to be. The ceiling is the largest aliquot the levels above can
# produce; the floor is *not* the smallest, because a parcel trimmed off a road comes out
# under its aliquot and that is the point of trimming. It is a floor on what is worth
# drawing at all: under this, or under a hundred metres on its short side, a parcel is a
# headland rather than a field and the ground is better left out of cultivation than cut
# into slivers.
#
# The floor is also where the field count is bought back, which is why it is 3.0 and not
# the 2.5 that reaches furthest. A town stands on the corner four sections meet at, and
# its platform is 516 x 510 m centred there, so it takes 258 m out of a section in x and
# 255 m in y - one whole ten and a bite of 67 m out of the next. What is left of that
# second ten is 134 x 201 m, 2.70 ha, and it cannot be subdivided because a ten is
# already the smallest cell the survey cuts. There are twenty-odd of those against the
# four towns and the plantings, and they are about 120 ha of ground: under two per cent
# of the map for a quarter of the field budget, on parcels a third the size of the smallest class
# the brief asks for. At 3.0 they go and the fields they were spending are laid out where
# the ground is worth working; the towns keep the ring of tens outside them, which is the
# thing the floor is really guarding.
FIELD_MAX_HA = FIELD_MERGE_MAX_HA
FIELD_MIN_HA = 3.0
FIELD_MIN_SIDE_M = 100.0
# How much of a side a parcel has to keep for trimming to be the right answer. Past this
# the obstacle is not along an edge, it is *in* the cell, and the cell wants quartering
# so its children can take the ground on both sides of it - trim instead and one side is
# thrown away.
FIELD_KEEP_FRAC = 0.50
# How much more ground quartering a cell has to recover before it is preferred to
# trimming it. Without the hysteresis every cell with anything along an edge splits, and
# the map comes out as ten-acre parcels from end to end whatever the size bands say.
#
# It is also the finest control there is over the count, and it costs almost nothing to
# turn: 0.90 covers 3258 ha against 0.80's 3189 - two per cent more ground - and spends
# fourteen more fields on it, which is over the cap. So the ground is nearly all
# reachable at any setting and what the setting buys is how finely it is cut up.
#
# 0.80 is not quite the most splitting the cap will carry - 0.82 lands 119 - and that is
# deliberate. 115 of 120 leaves the same five fields of headroom the old 145 of 150 had,
# and the headroom is the point: `validate()` holds the cap so the *next* feature added
# to the map cannot quietly push it over, and a margin of one is a cap that fails on the
# next shelterbelt. Fifteen hectares is what that costs.
FIELD_SPLIT_GAIN = 0.80


_RIVER_Y = [p[1] for p in river_axis()]


def _pt_rect_dist(p, x0, y0, x1, y1):
    return math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1))


def _seg_rect_dist(a, b, x0, y0, x1, y1):
    """Exact distance from a segment to an axis-aligned rectangle.

    Both are convex, so the closest pair is realised at a vertex of one of them - which
    is why taking the rectangle's corners against the segment *and* the segment's ends
    against the rectangle is exact, and why either one on its own is not.
    """
    if (x0 <= a[0] <= x1 and y0 <= a[1] <= y1) or (x0 <= b[0] <= x1 and y0 <= b[1] <= y1):
        return 0.0
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    for i in range(4):
        if segs_cross(a, b, corners[i], corners[(i + 1) % 4]):
            return 0.0
    return min(min(seg_point_dist(c, a, b) for c in corners),
               _pt_rect_dist(a, x0, y0, x1, y1), _pt_rect_dist(b, x0, y0, x1, y1))


def _polyline_rect_dist(pts, rect, stop):
    """Min distance from a polyline to a rectangle, giving up as soon as it is under
    `stop`. The caller only ever asks "is this closer than the clearance", so there is no
    reason to walk eight kilometres of river once the answer is known."""
    x0, y0, x1, y1 = rect
    best = 1.0e18
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if (min(a[0], b[0]) - x1 > best or x0 - max(a[0], b[0]) > best
                or min(a[1], b[1]) - y1 > best or y0 - max(a[1], b[1]) > best):
            continue
        d = _seg_rect_dist(a, b, x0, y0, x1, y1)
        if d < best:
            best = d
            if best < stop:
                return best
    return best


def _rects_apart(rect, cx, cy, w, h, gap):
    x0, y0, x1, y1 = rect
    return not (x0 < cx + w / 2.0 + gap and x1 > cx - w / 2.0 - gap
                and y0 < cy + h / 2.0 + gap and y1 > cy - h / 2.0 - gap)


def _trim_one(rect, axis, lo, hi, keep_frac):
    """Pull `rect` clear of the band `lo..hi` on `axis` (0 = x, 1 = y).

    Returns the trimmed rectangle, or None if the band sits far enough into the cell that
    clearing it would eat more than `keep_frac` of that side. That case is not a failure -
    it is a cell that wants quartering, and its children will find the ground on both
    sides of the obstacle instead of the parcelling throwing one side away.
    """
    r = list(rect)
    a, b = r[axis], r[axis + 2]
    if hi <= a or lo >= b:
        return rect
    span = b - a
    cut_lo, cut_hi = hi - a, b - lo          # what each side would cost
    if cut_lo <= cut_hi:
        if cut_lo > (1.0 - keep_frac) * span:
            return None
        r[axis] = hi
    else:
        if cut_hi > (1.0 - keep_frac) * span:
            return None
        r[axis + 2] = lo
    return tuple(r)


def _trim_water(rect, water):
    """Pull `rect` clear of the water's valley.

    The valley is not a box - it is a 1 km band following a meander - so it cannot go in
    with the yards and the woods. What can be said about it *for one cell* is how far it
    reaches in x over that cell's own range of y, which is a band, and a band is what
    `_trim_one` takes. Taking the extreme over the range makes it a superset of the true
    intrusion, so this can only ever cut too much; `_field_clear` measures the real
    distance afterwards.

    Trimming rather than rejecting is what "avoid the valley" has to mean if the ground
    beside it is to be used at all. Rejecting whole cells cost a fifth of the map: a
    quarter section is 805 m and the valley is wider than that, so every parcel within
    reach of the river came back as tens or as nothing.
    """
    for pts, bbox, kind, clear in water:
        if _rects_apart(rect, *bbox, 0.0):
            continue
        if kind == 'axis':
            # The river axis is a single-valued function of northing, so it comes out of
            # `river_axis()` sorted in y and the band can be sliced rather than scanned.
            lo = rect[1] - clear
            hi = rect[3] + clear
            i0 = bisect.bisect_left(_RIVER_Y, lo)
            i1 = bisect.bisect_right(_RIVER_Y, hi)
            xs = [p[0] for p in pts[i0:i1]]
            if not xs:
                continue
            rect = _trim_one(rect, 0, min(xs) - clear, max(xs) + clear,
                             FIELD_KEEP_FRAC)
        else:
            best = None
            for axis in (0, 1):
                lo = bbox[axis] - bbox[axis + 2] / 2.0
                cand = _trim_one(rect, axis, lo, lo + bbox[axis + 2], FIELD_KEEP_FRAC)
                if cand is None:
                    continue
                area = (cand[2] - cand[0]) * (cand[3] - cand[1])
                if best is None or area > best[0]:
                    best = (area, cand)
            rect = None if best is None else best[1]
        if rect is None:
            return None
    return rect


def _trim(rect, water, corridors, boxes):
    """Pull a parcel's edges in off the water, off anything built and off any road that
    runs along them.

    An aliquot part of a section is flush with the section lines, and the section lines
    are where the roads are - so *every* quarter section on this map has a road along one
    edge and a straight clearance test rejects the lot of them. It did: the first run
    came back with 342 ten-acre parcels, no forties on a boundary and not one quarter
    section anywhere, because the only cells that passed were the ones buried in the
    middle of a section.

    A parcel and a field are not the same thing. The parcel is the forty; the field is
    the forty less the road allowance along it, which is why a field abutting a road is a
    little smaller than one that does not. Trimming is that. A road through the *middle*
    of a cell is a different thing and comes back as None - the cell is quartered and its
    children are offered instead, which is how the parcelling finds the two halves either
    side of a road on its own.

    The order is load-bearing where a road stops beside a platform, which is every town
    street on the map. A box knows its own extent in both directions and can only reach a
    cell it truly touches; a corridor is tested as a band along its axis with the reach
    it has *along* that axis, so a street that runs up to the town and stops still counts
    against a cell whose only overlap with it is ground the town's own platform takes
    out. Trimmed in that order the street ate a full-width band out of the ten north of
    every town - 201 x 137 m of it, down to 1.26 ha and under any floor - and the towns
    came out with 147 m of nothing along two sides and a field at the ten-metre headland
    along the others. Take the platform off first and the street no longer reaches.
    """
    rect = _trim_water(rect, water)
    if rect is None:
        return None
    for cx, cy, w, h, gap in boxes:
        if _rects_apart(rect, cx, cy, w, h, gap):
            continue
        # Take whichever axis costs less, so a yard against a corner takes the corner
        # off and not half the field.
        best = None
        for axis, c, s in ((0, cx, w), (1, cy, h)):
            cand = _trim_one(rect, axis, c - s / 2.0 - gap, c + s / 2.0 + gap,
                             FIELD_KEEP_FRAC)
            if cand is None:
                continue
            area = (cand[2] - cand[0]) * (cand[3] - cand[1])
            if best is None or area > best[0]:
                best = (area, cand)
        if best is None:
            return None
        rect = best[1]
    for pts, keep in corridors:
        vertical = abs(pts[0][0] - pts[-1][0]) < abs(pts[0][1] - pts[-1][1])
        along = min((p[1] if vertical else p[0]) for p in pts), \
            max((p[1] if vertical else p[0]) for p in pts)
        axis = 0 if vertical else 1
        if along[1] < rect[1 - axis] or along[0] > rect[3 - axis]:
            continue                     # the road does not reach this far along
        at = pts[0][0] if vertical else pts[0][1]
        rect = _trim_one(rect, axis, at - keep, at + keep, FIELD_KEEP_FRAC)
        if rect is None:
            return None
    return rect


def _field_clear(rect, obstacles):
    """Is this rectangle ground a field can be laid on?

    `obstacles` is everything already on the map, gathered once by the caller: walking
    `CORRIDORS` and `river_axis()` afresh for each of a couple of thousand candidate
    cells is the difference between a layout that loads in a second and one that does
    not.
    """
    x0, y0, x1, y1 = rect
    m = EDGE_CLEAR_M
    if x0 < m or y0 < m or x1 > PLAYABLE_M - m or y1 > PLAYABLE_M - m:
        return False
    # The floors are on the field as it is *drawn*, so they are tested on the cell less
    # the gap that comes off it at the end. Tested on the cell itself they would pass a
    # 3.00 ha parcel that reaches the map as 2.99, and `validate()` - which measures the
    # ring - would then fail a layout the parcelling thought was fine.
    w, h = x1 - x0 - FIELD_GAP_M, y1 - y0 - FIELD_GAP_M
    if w * h < FIELD_MIN_HA * 10000.0:
        return False
    if min(w, h) < FIELD_MIN_SIDE_M:
        return False
    water, corridors, boxes = obstacles
    for pts, bbox, _, clear in water:
        if _rects_apart(rect, *bbox, 0.0):
            continue                     # nowhere near this body of water
        if _polyline_rect_dist(pts, rect, clear) < clear:
            return False
    for cx, cy, w, h, gap in boxes:
        if not _rects_apart(rect, cx, cy, w, h, gap):
            return False
    return True


def _field_cap(cx, cy, towns, merging=False):
    """The largest parcel allowed at this point: a ten near a town, a forty beyond it, a
    quarter section out in the country. Land near a settlement is worth more, gets sold
    in smaller pieces and stays that way.

    A merge is allowed `FIELD_MERGE_STEPS` halvings over that, up to the merge cap: what
    the band governs is how the ground was subdivided, and a merge is what is worked
    afterwards. Held to the band itself the rule can only ever join the two halves of the
    largest parcel the band allows, which is nothing at all wherever the aliquot grid
    already cuts at the band's own size.
    """
    d = min(math.dist((cx, cy), t) for t in towns) if towns else 1.0e18
    if d < FIELD_SMALL_M:
        cap = FIELD_CAP_SMALL_HA
    elif d < FIELD_MEDIUM_M:
        cap = FIELD_CAP_MEDIUM_HA
    else:
        cap = FIELD_CAP_LARGE_HA
    if not merging:
        return cap
    return min(cap * 2.0 ** FIELD_MERGE_STEPS, FIELD_MERGE_MAX_HA)


def _union_if_flush(a, b, tol=1e-6):
    """The rectangle covering both, if they share a whole edge - otherwise None.

    A *whole* edge and not a partial one: two rectangles that merely touch along part of
    a side have a union that is L-shaped, and an L is not a field. That is the only thing
    keeping this from producing a shape the brief rules out.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if abs(ay0 - by0) < tol and abs(ay1 - by1) < tol:
        if abs(ax1 - bx0) < tol:
            return (ax0, ay0, bx1, ay1)
        if abs(bx1 - ax0) < tol:
            return (bx0, ay0, ax1, ay1)
    if abs(ax0 - bx0) < tol and abs(ax1 - bx1) < tol:
        if abs(ay1 - by0) < tol:
            return (ax0, ay0, ax1, by1)
        if abs(by1 - ay0) < tol:
            return (ax0, by0, ax1, ay1)
    return None


def _merge_fields(rects, towns):
    """Join parcels that share a whole edge, while the result stays inside its band.

    The aliquot grid cuts a field wherever a halving line falls, whether or not anything
    on the ground calls for one: two tens either side of a line with nothing between them
    are one field that the survey happened to write down as two. This puts them back
    together, and it is safe to do after the fact rather than during - the union of two
    rectangles sharing a whole edge is exactly the two of them, no new ground, so
    whatever they were both clear of the union is clear of too.
    """
    rects = list(rects)
    again = True
    while again:
        again = False
        i = 0
        while i < len(rects):
            j = i + 1
            while j < len(rects):
                u = _union_if_flush(rects[i], rects[j])
                cap = None if u is None else _field_cap(
                    (u[0] + u[2]) / 2.0, (u[1] + u[3]) / 2.0, towns, merging=True)
                if u is not None and \
                        (u[2] - u[0]) * (u[3] - u[1]) / 10000.0 <= cap * 1.000001:
                    rects[i] = u
                    rects.pop(j)
                    again = True
                else:
                    j += 1
            i += 1
    return rects


def _plan(rect, obstacles, towns, memo):
    """The best parcelling of one cell, as `(hectares, [rectangles])`.

    Three things can be done with a cell an obstacle reaches into: trim it back and keep
    one large field, halve it north-south, or halve it east-west. None of them is right
    on its own. Trimming everything cost a twentieth of the map - a quarter section
    pulled back off the river's valley throws away whatever was on the far side of it -
    and splitting everything gave a map of nothing but ten-acre parcels, because near
    enough every cell on a map this full has *something* along an edge.

    So all three are worked out and the best is taken, with `FIELD_SPLIT_GAIN` of
    hysteresis in favour of the whole cell: splitting has to recover meaningfully more
    ground, not merely a hectare more, or the parcelling dissolves into the smallest
    aliquot everywhere and the size bands the brief asked for stop meaning anything.

    Halving alternately in each direction reaches the same cell by more than one route -
    north half then east half is the same quarter as east half then north half - so the
    results are memoised on the rectangle. Without it the tree is 4^6 nodes a section and
    takes minutes; with it there are 225 distinct cells in a section and it takes
    seconds.
    """
    if rect in memo:
        return memo[rect]
    x0, y0, x1, y1 = rect
    if x1 <= EDGE_CLEAR_M or y1 <= EDGE_CLEAR_M \
            or x0 >= PLAYABLE_M - EDGE_CLEAR_M or y0 >= PLAYABLE_M - EDGE_CLEAR_M:
        memo[rect] = (0.0, [])
        return memo[rect]

    area = (x1 - x0) * (y1 - y0) / 10000.0
    keep = None
    if area <= _field_cap((x0 + x1) / 2.0, (y0 + y1) / 2.0, towns) * 1.000001:
        cut = _trim(rect, *obstacles)
        if cut is not None and _field_clear(cut, obstacles):
            keep = cut
    kept = 0.0 if keep is None \
        else (keep[2] - keep[0]) * (keep[3] - keep[1]) / 10000.0

    best = (kept, [keep]) if keep else (0.0, [])
    floor = FIELD_SECTION_M / 2 ** FIELD_SPLITS
    for axis in (0, 1):
        side = (x1 - x0) if axis == 0 else (y1 - y0)
        if side / 2.0 < floor - 1e-6:
            continue                     # a ten is the smallest parcel this survey cuts
        mid = (x0 + x1) / 2.0 if axis == 0 else (y0 + y1) / 2.0
        halves = (((x0, y0, mid, y1), (mid, y0, x1, y1)) if axis == 0
                  else ((x0, y0, x1, mid), (x0, mid, x1, y1)))
        total, got = 0.0, []
        for half in halves:
            a, r = _plan(half, obstacles, towns, memo)
            total += a
            got += r
        if total * FIELD_SPLIT_GAIN > best[0]:
            best = (total, got)
    memo[rect] = best
    return best


def build_fields(planted):
    """Every field on the map, as `AREAS` records.

    `planted` is what is already drawn - the island, the town blocks, the woods and the
    belts - passed in rather than read from `AREAS`, because this is what fills `AREAS`
    and a registry cannot be its own input.
    """
    def bbox(pts, grow):
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        return cx, cy, max(xs) - min(xs) + 2 * grow, max(ys) - min(ys) + 2 * grow

    water = [(river_axis(), bbox(river_axis(), FIELD_RIVER_CLEAR_M), 'axis',
              FIELD_RIVER_CLEAR_M),
             (lake_ring(), bbox(lake_ring(), FIELD_LAKE_CLEAR_M), 'ring',
              FIELD_LAKE_CLEAR_M)]
    corridors = [(c['axis'], c['half_width_m'] + c['feather_m']) for c in CORRIDORS]
    boxes = [(p['centre'][0], p['centre'][1], p['size'][0], p['size'][1], FIELD_CLEAR_M)
             for p in PADS]
    for a in planted:
        # A planting goes in as its bounding box, which is exact because every one of
        # them is a rectangle. The timber along the water is not, and it is not here: it
        # is laid out after the fields rather than before them, off the edges they leave.
        xs = [q[0] for q in a['ring']]
        ys = [q[1] for q in a['ring']]
        boxes.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0,
                      max(xs) - min(xs), max(ys) - min(ys), FIELD_CLEAR_M))
    obstacles = (water, corridors, boxes)
    towns = [p['centre'] for p in PADS if p['kind'] == 'town']

    ax, ay = FIELD_ANCHOR
    i0 = int(math.floor(-ax / FIELD_SECTION_M))
    i1 = int(math.ceil((PLAYABLE_M - ax) / FIELD_SECTION_M))
    j0 = int(math.floor(-ay / FIELD_SECTION_M))
    j1 = int(math.ceil((PLAYABLE_M - ay) / FIELD_SECTION_M))
    rects = []
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            x0 = ax + i * FIELD_SECTION_M
            y0 = ay + j * FIELD_SECTION_M
            sec = (x0, y0, x0 + FIELD_SECTION_M, y0 + FIELD_SECTION_M)
            # Only the obstacles this section can reach. Every cell inside it tests the
            # list once, so cutting a hundred and forty-five down to the handful that are
            # actually nearby is most of the time this takes.
            near = (
                [w for w in water if not _rects_apart(sec, *w[1], 0.0)],
                [c for c in corridors
                 if not _rects_apart(sec, *bbox(c[0], c[1]), 0.0)],
                [b for b in boxes if not _rects_apart(sec, *b[:4], b[4])],
            )
            rects += _plan(sec, near, towns, {})[1]
    # North to south then west to east, so the numbering runs the way the survey does
    # and adding a feature renumbers the fields after it rather than shuffling them all.
    rects = _merge_fields(rects, towns)
    # Half the gap off every side, once everything else is decided. Two fields that were
    # flush now stand FIELD_GAP_M apart and nothing else about the layout has moved.
    g = FIELD_GAP_M / 2.0
    rects = [(r[0] + g, r[1] + g, r[2] - g, r[3] - g) for r in rects]
    rects.sort(key=lambda r: (r[1], r[0]))
    return [{'id': f'field_{n + 1:03d}', 'name': f'Campo {n + 1:03d}',
             'ring': rect_ring(*r), 'area_ha': (r[2] - r[0]) * (r[3] - r[1]) / 10000.0,
             'tags': {'landuse': 'farmland'}}
            for n, r in enumerate(rects)]


# --- the OSM vocabulary -----------------------------------------------------------
# Exactly what `osm_generator/visualize_osm.py` and `visualizer/create_3d_viewer.py` know
# how to draw. A way tagged with anything else is dropped by both renderers without a
# word, so it is worse than useless: it costs nodes and shows nothing. Both the generator
# and the checker read this list, so the two cannot drift apart.
RENDERED_TAGS = (('natural', 'water'), ('water', None), ('natural', 'wood'),
                 ('landuse', 'forest'), ('landuse', 'farmyard'),
                 ('landuse', 'farmland'), ('railway', None), ('highway', None))


# Every ring drawn as timber carries both tags. `natural=wood` is the one that draws it,
# and `landuse=farmyard` beside it is what makes the ground under it a parcel the game
# can own - the same tag the granjas and the town blocks stand on, because that is the
# only word in the closed vocabulary for "ground somebody holds". The pair is safe
# because all three renderers here - `visualize_osm`, `create_3d_viewer` and
# `render_pda` - test the wood first and never reach the landuse, so a wood is drawn as
# timber and not as a yard. Emitted through one helper rather than written out at each
# of the five sites, so the woods, the belts, the ribera and the island cannot drift.
def wood_tags(leaf_type):
    return {'natural': 'wood', 'landuse': 'farmyard', 'leaf_type': leaf_type}


# The road classes the renderers colour, keyed by the `kind` a corridor record carries.
HIGHWAY_CLASS = {'primary': 'primary', 'secondary': 'secondary', 'tertiary': 'tertiary',
                 'section': 'secondary', 'track': 'tertiary', 'street': 'tertiary'}


# ==================================================================================
# projection
# ==================================================================================
def local_to_global(x, y):
    """Playable metres -> (lat, lon). y grows southwards, so it subtracts."""
    return (LAT_CENTER - (y - HALF_M) / M_PER_DEG,
            LON_CENTER + (x - HALF_M) / M_PER_DEG_LON)


def global_to_local(lat, lon):
    """(lat, lon) -> playable metres. The inverse of local_to_global."""
    return (HALF_M + (lon - LON_CENTER) * M_PER_DEG_LON,
            HALF_M - (lat - LAT_CENTER) * M_PER_DEG)


def bounds():
    """The four values of the OSM `<bounds>` element, as (minlat, minlon, maxlat,
    maxlon). The south-west corner is local (0, PLAYABLE_M); the north-east is
    (PLAYABLE_M, 0)."""
    minlat, minlon = local_to_global(0.0, PLAYABLE_M)
    maxlat, maxlon = local_to_global(PLAYABLE_M, 0.0)
    return minlat, minlon, maxlat, maxlon


def to_canvas(x, y):
    """Playable metres -> canvas metres (the DEM's frame, origin at its NW corner)."""
    return x + OFFSET_M, y + OFFSET_M


def from_canvas(xc, yc):
    return xc - OFFSET_M, yc - OFFSET_M


# ==================================================================================
# geometry helpers
# ==================================================================================
def polyline_length(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def ring_area_ha(ring):
    """Shoelace area in hectares. The ring may be given open or closed."""
    pts = ring[:-1] if len(ring) > 2 and math.dist(ring[0], ring[-1]) < 1e-9 else ring
    if len(pts) < 3:
        return 0.0
    twice = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] -
                pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return abs(twice) / 2.0 / 10000.0


def ring_perimeter(ring):
    pts = ring if math.dist(ring[0], ring[-1]) < 1e-9 else list(ring) + [ring[0]]
    return polyline_length(pts)


def close_ring(ring):
    return ring if math.dist(ring[0], ring[-1]) < 1e-9 else list(ring) + [ring[0]]


def rect_ring(x0, y0, x1, y1):
    """Axis-aligned rectangle as a closed ring, counter-clockwise in screen terms."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def clip_ring_to_rect(ring, x0, y0, x1, y1):
    """Sutherland-Hodgman against an axis-aligned rectangle.

    Clamping a ring's coordinates instead folds whatever hangs over the edge onto the
    edge itself, which is how a strip of gallery timber ended up with a run of nodes
    lying across a river channel. Clipping only ever puts a new vertex on the ring's
    own boundary, and cuts a straight edge where the ring leaves the rectangle.

    Returns a closed ring, or [] if nothing of it is left inside.
    """
    poly = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else list(ring)
    for keep, cut in ((lambda p: p[0] >= x0, lambda a, b: _cut_x(a, b, x0)),
                      (lambda p: p[0] <= x1, lambda a, b: _cut_x(a, b, x1)),
                      (lambda p: p[1] >= y0, lambda a, b: _cut_y(a, b, y0)),
                      (lambda p: p[1] <= y1, lambda a, b: _cut_y(a, b, y1))):
        out = []
        for i, b in enumerate(poly):
            a = poly[i - 1]
            if keep(b):
                if not keep(a):
                    out.append(cut(a, b))
                out.append(b)
            elif keep(a):
                out.append(cut(a, b))
        poly = out
        if not poly:
            return []
    return poly + [poly[0]]


def _cut_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return (x, a[1] + t * (b[1] - a[1]))


def _cut_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def ellipse_ring(cx, cy, a, b, rot_deg=0.0, n=48):
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    out = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        u, v = a * math.cos(t), b * math.sin(t)
        out.append((cx + u * c - v * s, cy + u * s + v * c))
    out.append(out[0])
    return out


def point_in_ring(pt, ring):
    """Even-odd test. The ring may be open or closed."""
    x, y = pt
    pts = ring[:-1] if math.dist(ring[0], ring[-1]) < 1e-9 else ring
    inside = False
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xx = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xx:
                inside = not inside
    return inside


def segs_cross(a, b, c, d):
    """True if segment ab properly crosses segment cd."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return False
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


def rings_overlap(p, q):
    """True if two simple rings share any area.

    A vertex test on its own is not enough, and the case it misses is not exotic: two
    rectangles crossing in a plus sign have no vertex of either inside the other, and
    that is exactly the shape a transversal shelterbelt makes against a north-south one.
    Two belts came out overlapping, drawn twice and counted twice in the inventory, and
    the check that was supposed to catch it said nothing. Test the edges as well.
    """
    if any(point_in_ring(v, q) for v in p) or any(point_in_ring(v, p) for v in q):
        return True
    return any(segs_cross(p[i], p[i + 1], q[j], q[j + 1])
               for i in range(len(p) - 1) for j in range(len(q) - 1))


def ring_is_simple(ring):
    """True if no two non-adjacent edges of a closed ring cross.

    The one shape on this map that can fail it is the gallery timber, and it is the
    first mistake in the list of things that have gone wrong here: offset a polyline by
    more than its radius of curvature and the ring folds through itself, and an even-odd
    fill punches holes in the tightest bends. A ring of a couple of hundred vertices is
    cheap to sweep once the boxes are compared first, so the rule is checked rather than
    argued from the numbers.
    """
    pts = ring[:-1] if math.dist(ring[0], ring[-1]) < 1e-9 else list(ring)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                 # the closing edge is adjacent to the first
            c, d = pts[j], pts[(j + 1) % n]
            if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
                continue
            if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
                continue
            if segs_cross(a, b, c, d):
                return False
    return True


def seg_point_dist(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll < 1e-12:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_to_polyline(p, pts):
    return min(seg_point_dist(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def catmull_rom(pts, per_seg=16):
    """Centripetal-ish Catmull-Rom through the control points, endpoints duplicated."""
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for j in range(per_seg):
            t = j / per_seg
            t2, t3 = t * t, t * t * t
            out.append(tuple(
                0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t
                       + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                       + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
                for k in (0, 1)))
    out.append(tuple(pts[-1]))
    return out


def densify(pts, step):
    """Resample a polyline to roughly `step` between vertices, keeping the ends."""
    out = [pts[0]]
    carry = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = math.dist(a, b)
        if seg < 1e-9:
            continue
        d = step - carry
        while d < seg:
            t = d / seg
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            d += step
        carry = seg - (d - step)
    out.append(pts[-1])
    return out


def offset_polyline(pts, dist):
    """Offset a polyline sideways by `dist` (positive = left of travel).

    Only safe while `dist` stays under the radius of curvature: offset a meander by more
    than it bends and the ring folds through itself, and an even-odd fill then punches
    holes in the tightest bends. Reserves along water are stamped by distance to the
    centreline instead, never as offset polygons.
    """
    out = []
    n = len(pts)
    for i in range(n):
        a = pts[max(0, i - 1)]
        b = pts[min(n - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ll = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ll, dx / ll
        out.append((pts[i][0] + dist * nx, pts[i][1] + dist * ny))
    return out


def buffer_ring(pts, half_w):
    """Closed ring around a polyline: left side out, right side back."""
    left = offset_polyline(pts, half_w)
    right = offset_polyline(pts, -half_w)
    return close_ring(left + right[::-1])


def clip_polyline(pts, x0, y0, x1, y1):
    """Keep the vertices inside the box, splitting into runs. Coarse (vertex level),
    which is all that is needed at 40 m sampling."""
    runs, cur = [], []
    for p in pts:
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            cur.append(p)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 2]


def _smoothstep(t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def playable_sdf(x, y):
    """Signed distance to the playable boundary: negative inside, positive out in the
    border. The clean strip is everything between -EDGE_CLEAR_M and 0."""
    dx = max(-x, x - PLAYABLE_M)
    dy = max(-y, y - PLAYABLE_M)
    if dx <= 0.0 and dy <= 0.0:
        return max(dx, dy)
    return math.hypot(max(dx, 0.0), max(dy, 0.0))


def rng():
    """The layout's random stream. Seeded, so both halves of the pipeline draw the same
    numbers - and key any jitter to position rather than to iteration order, or adding
    one feature shifts every feature after it."""
    return random.Random(SEED)


# ==================================================================================
# the registries
# ==================================================================================
# Everything the world is made of, in four lists. Both generators read them and neither
# defines geometry of its own, so a feature added here appears in the terrain and in the
# vectors together, and a feature added to only one half of the pipeline is the bug this
# arrangement exists to prevent.

# Roads and railway. One record per alignment:
#     id             short stable key, used in messages and by the DEM's build order
#     name           what goes in the OSM `name` tag
#     kind           'primary' | 'section' | 'track' | 'street' | 'rail'
#                    - the class decides the OSM highway tag (HIGHWAY_CLASS), how wide
#                      the graded platform is and which corridor wins where two cross
#     axis           [(x, y), ...] centreline in playable metres, running from EDGE_MIN
#                    to EDGE_MAX if it leaves the map: an alignment that stops at the
#                    canvas edge ends in a cliff
#     half_width_m   half the running surface
#     feather_m      nominal width of the fill either side; the DEM widens it to
#                    max(feather_m, 1.5 * |dz| / tan(4 deg)) wherever the cut is deep,
#                    because in a smoothstep the steepest gradient is 1.5 * rise / run
#                    and a constant feather cuts a step under a deep platform
#     grade_max      ruling grade, as a fraction (rail is ~0.015, a section road ~0.06)
#     bridge_spans   [(s0, s1), ...] arc lengths along the axis carried on a deck, so
#                    the DEM leaves the channel alone there and the OSM tags bridge=yes
_ROOT = os.path.dirname(os.path.abspath(__file__))
_INPUT_OSM = os.path.join(_ROOT, 'input', 'custom_osm.osm')

# The ridge in the east of the map and the wood that stands on it. Both halves of the
# pipeline need to agree which feature this is - the OSM stretches the wood's ring out to
# the clean strip, the DEM runs the ground under it across the boundary and into the
# border range - so the way id is named once, here, and read from both.
EAST_RIDGE_WAY = 517
EAST_WOOD_STRETCH_M = 400.0     # how much of the wood's tip takes up the stretch

CORRIDORS = []
WATER = []
PADS = []
_PLANTED = []
FIELDS = []
GALLERY = []
AREAS = []

if os.path.exists(_INPUT_OSM):
    import xml.etree.ElementTree as ET
    _tree = ET.parse(_INPUT_OSM)
    _root = _tree.getroot()
    _b = _root.find('bounds')
    if _b is not None:
        _minlat = float(_b.get('minlat'))
        _maxlat = float(_b.get('maxlat'))
        _minlon = float(_b.get('minlon'))
        _maxlon = float(_b.get('maxlon'))
    else:
        _minlat, _minlon, _maxlat, _maxlon = bounds()

    _lat_c = (_minlat + _maxlat) / 2.0
    _lon_c = (_minlon + _maxlon) / 2.0
    _cos_lat = math.cos(math.radians(_lat_c))
    _m_lat = 111111.0
    _m_lon = 111111.0 * _cos_lat

    # The survey has to be drawn for this map's size: the loader places every node
    # from the bounds' north-west corner, so a survey of another size would load
    # without complaint and put every feature in the wrong place against the DEM.
    INPUT_EXTENT_M = ((_maxlon - _minlon) * _m_lon, (_maxlat - _minlat) * _m_lat)
    if max(abs(INPUT_EXTENT_M[0] - PLAYABLE_M), abs(INPUT_EXTENT_M[1] - PLAYABLE_M)) > 0.05:
        raise RuntimeError(f"{_INPUT_OSM}: bounds span {INPUT_EXTENT_M[0]:.2f} x "
                           f"{INPUT_EXTENT_M[1]:.2f} m, not the {PLAYABLE_M:.0f} m this map is")

    _nodes = {int(n.get('id')): (float(n.get('lat')), float(n.get('lon'))) for n in _root.findall('node')}

    def _node_xy(nid):
        lat, lon = _nodes[nid]
        return ((lon - _minlon) * _m_lon, (_maxlat - lat) * _m_lat)


    def _strip_extent(pts):
        """The extent of a ring, held inside the clean strip.

        A pad is a rectangle to the DEM, so it is the extent and not the ring that has
        to respect the strip."""
        lo, hi = EDGE_CLEAR_M, PLAYABLE_M - EDGE_CLEAR_M
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return ((min(max(min(xs), lo), hi), min(max(max(xs), lo), hi)),
                (min(max(min(ys), lo), hi), min(max(max(ys), lo), hi)))

    # Ways the survey draws that are not built. Empty since the conversion from the
    # 16x map, which left the dropped ways out of the survey it wrote; the mechanism
    # stays so a way can be taken out of the build by id without editing the survey.
    _TOWN_RESERVOIR_WAYS = set()

    # Ways dropped from the map. They are still drawn in the input file, which is the
    # authored survey - a way is taken out by id, the same way the town and reservoir
    # ways are, so the input stays the one record of what was surveyed and this module
    # stays the one record of what is built.
    #   190            Yard N5_1, the strip yard along the north lane
    #   187, 188, 189  three lane stubs that ran north off the lane into Campo 6
    #   204            the lane between the stubs and the primary; Yard N5_7 now
    #                  stretches east over it to the boundary lane, and way 203 on its
    #                  west side is the access that stays
    #   509 to 521     the eleven rectangular woods the survey drew between the fields
    #                  (517, the wood on the eastern ridge, stays)
    _DROPPED_WAYS = {190, 187, 188, 189, 204,
                     509, 510, 511, 512, 514, 515, 516, 518, 519, 520, 521}

    # Farmyards levelled by name rather than by the `m4fs:level` tag: every yard called
    # "Granja N" is a working farm and gets a platform. Kept here, like the dropped ways,
    # so the input stays the untouched survey; a new "Granja 4" drawn there is levelled
    # without anyone remembering to tag it.
    _LEVELLED_NAME = re.compile(r'^Granja \d+$')

    # Yards the survey drew that are worked as fields instead. The ring is read as
    # drawn and only its tags are replaced - it becomes farmland, joins `FIELDS` for
    # the shelterbelts, and stops being a pad, so the DEM leaves the ground under it
    # rolling like any other field's.
    _FARMLAND_WAYS = set()

    for _w in _root.findall('way'):
        _wid = int(_w.get('id'))
        if _wid <= 0 or _wid in _TOWN_RESERVOIR_WAYS or _wid in _DROPPED_WAYS:
            continue
        _tags = {t.get('k'): t.get('v') for t in _w.findall('tag')}
        _refs = [int(nd.get('ref')) for nd in _w.findall('nd')]
        _pts = [_node_xy(r) for r in _refs]
        _name = _tags.get('name', f'way_{_wid}')
        if _wid in _FARMLAND_WAYS:
            _tags = {'landuse': 'farmland', 'name': _name}

        if 'highway' in _tags:
            _kind = _tags['highway']
            _half_w = 4.0 if _kind == 'primary' else (3.0 if _kind == 'secondary' else 2.5)
            CORRIDORS.append({
                'id': f'road_{_wid}',
                'kind': _kind,
                'name': _name,
                'axis': _pts,
                'half_width_m': _half_w,
                'feather_m': 6.0,
                'grade_max': 0.08,
                'bridge_spans': [],
                'tags': _tags,
            })
        elif _tags.get('natural') == 'water' or _tags.get('water') == 'lake':
            WATER.append({
                'id': f'water_{_wid}',
                'kind': _tags.get('water', 'lake'),
                'name': _name,
                'ring': _pts,
                'tags': _tags,
            })
        elif _tags.get('natural') == 'wood' or _tags.get('landuse') == 'forest':
            AREAS.append({
                'id': f'wood_{_wid}',
                'kind': 'wood',
                'name': _name,
                'ring': _pts,
                'tags': _tags,
            })
        elif _tags.get('landuse') == 'farmland':
            AREAS.append({
                'id': f'field_{_wid}',
                'kind': 'farmland',
                'name': _name,
                'ring': _pts,
                'tags': _tags,
            })
            FIELDS.append(AREAS[-1])
        elif _tags.get('place') in ('town', 'village'):
            # A town's platform, and nothing else: it says where the ground was
            # levelled, not what stands on it. The town's drawn form is its blocks,
            # which are `AREAS` rings of their own, so this record carries no `tags`
            # key - `emit_pads` draws only the pads that have one, and a platform that
            # drew itself would put a second footprint over every block in the grid.
            # One platform for the whole town rather than one per block, so the streets
            # between them come out flat and continuous instead of stepping at a kerb.
            _xs, _ys = _strip_extent(_pts)
            _cx, _cy = (_xs[0] + _xs[1]) / 2.0, (_ys[0] + _ys[1]) / 2.0
            PADS.append({
                'id': f'town_pad_{_wid}',
                'kind': 'town',
                'name': _name,
                'centre': (_cx, _cy),
                'size': (_xs[1] - _xs[0], _ys[1] - _ys[0]),
                'ring': _pts,
                'feather_m': TOWN_PAD_FEATHER_M,
                'drain_grade': TOWN_DRAIN_GRADE,
                'level': True,
            })
        elif _tags.get('landuse') == 'farmyard':
            # `m4fs:level` is a build directive and not map data: it says this yard's
            # ground was graded, which is a fact about the terrain and not about the
            # ring. It is taken off the tags here rather than carried through, because
            # an attribute no renderer reads is the one thing not worth emitting - and
            # `check_osm` would be right to ask what draws it.
            _level = (_tags.pop('m4fs:level', None) == 'yes'
                      or bool(_LEVELLED_NAME.match(_name)))
            # The platform stops at the clean strip like everything else. The ring the
            # OSM draws is clipped to it, and a pad that went on grading to the boundary
            # would leave the strip clear in the vectors and levelled in the ground -
            # which is the disagreement between the two halves this module exists to
            # prevent. The feather still runs outwards past the platform edge, because a
            # platform has to come down to the ground it stands in somewhere.
            _xs, _ys = _strip_extent(_pts)
            _cx, _cy = (_xs[0] + _xs[1]) / 2.0, (_ys[0] + _ys[1]) / 2.0
            _w_size, _h_size = _xs[1] - _xs[0], _ys[1] - _ys[0]
            PADS.append({
                'id': f'yard_pad_{_wid}',
                'kind': 'farmyard',
                'name': _name,
                'centre': (_cx, _cy),
                'size': (_w_size, _h_size),
                'ring': _pts,
                'feather_m': YARD_FEATHER_M,
                'drain_grade': YARD_DRAIN_GRADE,
                'level': _level,
                'tags': _tags,
            })
            AREAS.append({
                'id': f'yard_area_{_wid}',
                'kind': 'farmyard',
                'name': _name,
                'ring': _pts,
                'tags': _tags,
            })


    # The eastern ridge runs out of the map - the DEM carries it across the boundary and
    # into the border range - and its timber runs with it as far as a vector is allowed
    # to go, which is the clean strip and not a metre further. The ring is *stretched*
    # rather than translated or clipped: the last EAST_WOOD_STRETCH_M of it take up the
    # whole of the gap on a smoothstep, so the rounded tip the way was drawn with is kept
    # and only moved, and everything west of that stands exactly where it was surveyed.
    for _a in AREAS:
        if _a['id'] != f'wood_{EAST_RIDGE_WAY}':
            continue
        _x_end = max(_p[0] for _p in _a['ring'])
        _gap = (PLAYABLE_M - EDGE_CLEAR_M) - _x_end
        if _gap > 0.0:
            _x_lo = _x_end - EAST_WOOD_STRETCH_M
            _ring = []
            for _px, _py in _a['ring']:
                _t = min(1.0, max(0.0, (_px - _x_lo) / EAST_WOOD_STRETCH_M))
                _ring.append((_px + _gap * _t * _t * (3.0 - 2.0 * _t), _py))
            _a['ring'] = _ring


# --- the railway ------------------------------------------------------------------
# One line, derived from the road it runs beside rather than drawn: the centreline is
# `RAIL_ALONG_ROAD`'s axis offset `RAIL_OFFSET_M` to one side, run straight on from
# both ends of the road to the playable boundary so the train leaves the map north and
# south. Derived, so that moving the road in JOSM moves the railway with it.
#
# The ground it takes is cut out of whatever was drawn there, the way the shelterbelts
# cut the fields: every ring within RAIL_HALF_W_M + RAIL_CLEAR_M of the centreline is
# trimmed back along the road, and where the line runs straight *through* a ring on its
# way to the edge the ring is split into the two fields either side. The belts are laid
# after this, so they keep off the railway on their own.
#
# Off: the map has no railway. `None` here makes `build_railway` return nothing, so no
# ring is cut back to its reserve and neither generator sees it. Naming a road again
# ('Mountain Pass Road' was the one) brings it back beside that road.
RAIL_ALONG_ROAD = None
RAIL_SIDE = +1                 # +1 = left of travel from the road's first node: the lake
                               # side, where the line meets only three yards and no
                               # levelled platform; -1 runs into Granja 3
RAIL_OFFSET_M = 10.0           # centreline to centreline; the beds are 3.5 m apart
RAIL_HALF_W_M = 2.5
RAIL_FEATHER_M = 6.0
RAIL_GRADE_MAX = 0.015
RAIL_CLEAR_M = 5.0             # from the edge of the bed to anything planted
RAIL_CORNER_R_M = 6.0          # fillet on a trimmed yard or wood; fields keep theirs


def _run_to_edge(p, d):
    """The point where a ray from `p` along `d` meets the playable boundary."""
    best = None
    for k, lim in ((0, 0.0), (0, PLAYABLE_M), (1, 0.0), (1, PLAYABLE_M)):
        if abs(d[k]) < 1e-12:
            continue
        t = (lim - p[k]) / d[k]
        if t > 1e-9 and (best is None or t < best):
            best = t
    return p if best is None else (p[0] + best * d[0], p[1] + best * d[1])


def _axis_frame(axis):
    """Segments of a polyline with unit tangent, unit normal and start arc length."""
    out, acc = [], 0.0
    for i in range(len(axis) - 1):
        a, b = axis[i], axis[i + 1]
        ll = math.dist(a, b)
        if ll < 1e-9:
            continue
        ux, uy = (b[0] - a[0]) / ll, (b[1] - a[1]) / ll
        out.append((a, b, ll, ux, uy, -uy, ux, acc))
        acc += ll
    return out


def _signed_to_axis(p, segs):
    """(signed distance, arc length, foot) of `p` against the nearest of `segs`:
    positive on the normal's side (left of travel)."""
    best = None
    for a, b, ll, ux, uy, nx, ny, s0 in segs:
        t = ((p[0] - a[0]) * ux + (p[1] - a[1]) * uy)
        t = 0.0 if t < 0.0 else (ll if t > ll else t)
        fx, fy = a[0] + ux * t, a[1] + uy * t
        d = math.hypot(p[0] - fx, p[1] - fy)
        if best is None or d < best[0]:
            sgn = 1.0 if (p[0] - fx) * nx + (p[1] - fy) * ny >= 0.0 else -1.0
            best = (d, sgn, s0 + t, (fx, fy), (nx, ny))
    d, sgn, s, foot, n = best
    return sgn * d, s, foot, n


def _densify_ring(ring, step):
    out = []
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        k = max(1, int(math.ceil(math.dist(a, b) / step)))
        out += [(a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k)
                for j in range(k)]
    return out


def _cut_ring_by_axis(ring, axis, half):
    """The pieces of `ring` left either side of a reserve `half` wide about `axis`.

    Each side is clipped on its own against the reserve's edge on that side: the ring
    is walked, every run of it that stands `half` or more off the line is kept, with
    the exact point where it entered and left that band at each end, and the run is
    closed back along the edge of the band between those two points. A ring that runs
    beside the line comes out following it with no wedge where it bends; a ring the
    line runs through comes out as one piece per side.
    """
    x0, y0, x1, y1 = ring_bbox(ring, half + 1.0)
    frame = _axis_frame(axis)
    segs = [sg for sg in frame
            if not boxes_apart((x0, y0, x1, y1), ring_bbox([sg[0], sg[1], sg[0]], 0.0))]
    if not segs:
        return [ring[:-1]], False
    pts = _densify_ring(ring, 3.0)
    sd = [_signed_to_axis(p, segs)[:2] for p in pts]
    if all(abs(d) >= half for d, _ in sd):
        return [ring[:-1]], False
    if all(abs(d) < half for d, _ in sd):
        return [], True

    # The band's edge on each side, as (s, point) along the line, within reach.
    edge = {1.0: [], -1.0: []}
    for q in densify(axis, 5.0):
        if not (x0 - 10.0 <= q[0] <= x1 + 10.0 and y0 - 10.0 <= q[1] <= y1 + 10.0):
            continue
        _, sq, foot, n = _signed_to_axis(q, segs)
        for side in edge:
            edge[side].append((sq, (foot[0] + side * half * n[0],
                                    foot[1] + side * half * n[1])))
    for side in edge:
        edge[side].sort()

    n = len(pts)
    pieces = []
    for side in (1.0, -1.0):
        v = [side * d for d, _ in sd]
        inside = [x >= half for x in v]
        if all(inside):
            pieces.append(list(pts))
            continue
        if not any(inside):
            continue
        starts = [i for i in range(n) if inside[i] and not inside[i - 1]]
        for i0 in starts:
            run = []
            i = i0
            while inside[i]:
                run.append(i)
                i = (i + 1) % n
            i_prev, i_next = (i0 - 1) % n, i

            def cross(a, b):
                t = (v[a] - half) / (v[a] - v[b])
                return ((pts[a][0] + t * (pts[b][0] - pts[a][0]),
                         pts[a][1] + t * (pts[b][1] - pts[a][1])),
                        sd[a][1] + t * (sd[b][1] - sd[a][1]))
            p_in, s_in = cross(run[0], i_prev)
            p_out, s_out = cross(run[-1], i_next)
            lo, hi = min(s_in, s_out), max(s_in, s_out)
            along = [q for sq, q in edge[side] if lo < sq < hi]
            if s_out > s_in:
                along = along[::-1]
            poly = [p_in] + [pts[k] for k in run] + [p_out] + along
            if len(poly) >= 3:
                pieces.append(poly)
    return pieces, True


def _finish_ring(poly, kind):
    r = FIELD_CORNER_R_M if kind == 'farmland' else RAIL_CORNER_R_M
    return fillet_ring(simplify_polyline(poly + [poly[0]], 0.05), r)


def _piece_ok(poly, kind):
    ha = poly_area_m2(poly) / 1.0e4
    if kind == 'farmland':
        return ha >= FIELD_MIN_HA
    per = ring_perimeter(poly + [poly[0]])
    return ha >= 0.15 and (2.0 * ha * 1.0e4 / per if per else 0.0) >= 8.0


def build_railway(areas, fields, pads, side=RAIL_SIDE):
    """The railway's corridor record, with `areas`, `fields` and `pads` trimmed to it in
    place. Returns `(corridor, report)`, or `(None, [])` if the road is not on the map;
    `report` lists (area id, hectares removed) for every ring touched."""
    road = next((c for c in CORRIDORS if c['name'] == RAIL_ALONG_ROAD), None)
    if road is None:
        return None, []
    beside = offset_polyline(road['axis'], side * RAIL_OFFSET_M)
    d_in = (beside[0][0] - beside[1][0], beside[0][1] - beside[1][1])
    d_out = (beside[-1][0] - beside[-2][0], beside[-1][1] - beside[-2][1])
    axis = [_run_to_edge(beside[0], d_in)] + beside + [_run_to_edge(beside[-1], d_out)]
    half = RAIL_HALF_W_M + RAIL_CLEAR_M
    report = []

    for a in list(areas):
        kind = a.get('kind')
        pieces, touched = _cut_ring_by_axis(a['ring'], axis, half)
        if not touched:
            continue
        kept = [p for p in pieces if _piece_ok(p, kind)]
        before = ring_area_ha(a['ring'])
        report.append((a['id'], before - sum(poly_area_m2(p) for p in kept) / 1.0e4))
        if not kept:
            areas.remove(a)
            if a in fields:
                fields.remove(a)
            continue
        kept.sort(key=lambda p: -poly_area_m2(p))
        a['ring'] = _finish_ring(kept[0], kind)
        for k, p in enumerate(kept[1:], 1):
            twin = dict(a, id=f"{a['id']}_{k}", ring=_finish_ring(p, kind))
            areas.insert(areas.index(a) + k, twin)
            if a in fields:
                fields.insert(fields.index(a) + k, twin)
        for pad in pads:
            if pad['id'] == a['id'].replace('area', 'pad'):
                pad['ring'] = a['ring']

    name = f'Ferrocarril ({RAIL_ALONG_ROAD})'
    return {'id': 'rail_1', 'kind': 'rail', 'name': name, 'axis': axis,
            'half_width_m': RAIL_HALF_W_M, 'feather_m': RAIL_FEATHER_M,
            'grade_max': RAIL_GRADE_MAX, 'bridge_spans': [],
            'tags': {'railway': 'rail', 'name': name}}, report


RAILWAY, RAIL_TRIMMED = build_railway(AREAS, FIELDS, PADS)
if RAILWAY is not None:
    CORRIDORS.append(RAILWAY)

# --- woods sold in parcels ----------------------------------------------------------
# A wood the player buys a piece at a time is drawn as that many rings, so each piece
# is its own parcel downstream. The cuts are straight and run across the wood's spine
# at equal-area stations, so the parcels are the same size and each one is a slice of
# the wood and not a sliver of it. The spine is derived from the ring itself - the
# midline between the two sides of the wood from tip to tip - so redrawing the wood in
# JOSM moves the cuts with it. The whole ring is kept in `EAST_RIDGE_RING` for the DEM,
# which runs the ridge under this wood out of the map and needs the wood's east tip.
WOOD_PARCELS = {EAST_RIDGE_WAY: 5}
WOOD_PARCEL_NAME = 'Bosque de la Sierra'


def _chain_at(chain, cum, t):
    """The point a fraction `t` of the way along a polyline with cumulative lengths."""
    target = t * cum[-1]
    i = bisect.bisect_left(cum, target)
    i = max(1, min(i, len(chain) - 1))
    f = (target - cum[i - 1]) / ((cum[i] - cum[i - 1]) or 1.0)
    return (chain[i - 1][0] + f * (chain[i][0] - chain[i - 1][0]),
            chain[i - 1][1] + f * (chain[i][1] - chain[i - 1][1]))


def split_ring_across(ring, n):
    """Cut a long ring into `n` pieces of equal area, across its spine. Returns the
    pieces as closed rings, from the ring's first tip to its second."""
    pts = _densify_ring(ring, 10.0)
    m = len(pts)
    # The tips: the two vertices farthest apart.
    best = (0.0, 0, 0)
    for i in range(0, m, 2):
        for j in range(i + 1, m, 2):
            d = math.dist(pts[i], pts[j])
            if d > best[0]:
                best = (d, i, j)
    _, i, j = best
    side_a = pts[i:j + 1]
    side_b = (pts[j:] + pts[:i + 1])[::-1]

    def cum(chain):
        out = [0.0]
        for k in range(1, len(chain)):
            out.append(out[-1] + math.dist(chain[k - 1], chain[k]))
        return out

    ca, cb = cum(side_a), cum(side_b)
    spine = []
    for k in range(201):
        t = k / 200.0
        a, b = _chain_at(side_a, ca, t), _chain_at(side_b, cb, t)
        spine.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    cs = cum(spine)

    def cut(t):
        """The cut across the spine at `t`: a point and the forward unit tangent."""
        p = _chain_at(spine, cs, t)
        q0 = _chain_at(spine, cs, max(0.0, t - 0.02))
        q1 = _chain_at(spine, cs, min(1.0, t + 0.02))
        ll = math.dist(q0, q1) or 1.0
        return p, ((q1[0] - q0[0]) / ll, (q1[1] - q0[1]) / ll)

    poly = ring[:-1]
    total = poly_area_m2(poly)

    def behind(t):
        p, u = cut(t)
        return clip_halfplane(poly, p, (-u[0], -u[1]))

    stations = []
    for k in range(1, n):
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if poly_area_m2(behind(mid)) < total * k / n:
                lo = mid
            else:
                hi = mid
        stations.append((lo + hi) / 2.0)

    pieces = []
    for k in range(n):
        piece = poly
        if k > 0:
            p, u = cut(stations[k - 1])
            piece = clip_halfplane(piece, p, u)
        if k < n - 1:
            p, u = cut(stations[k])
            piece = clip_halfplane(piece, p, (-u[0], -u[1]))
        pieces.append(simplify_polyline(close_ring(piece), 0.05))
    return pieces


EAST_RIDGE_RING = None
for _a in list(AREAS):
    _wid = int(_a['id'].split('_')[1]) if _a['kind'] == 'wood' and _a['id'].count('_') == 1 else None
    if _wid not in WOOD_PARCELS:
        continue
    if _wid == EAST_RIDGE_WAY:
        EAST_RIDGE_RING = _a['ring']
    _k = AREAS.index(_a)
    AREAS.remove(_a)
    for _i, _ring in enumerate(split_ring_across(_a['ring'], WOOD_PARCELS[_wid]), 1):
        AREAS.insert(_k + _i - 1, {
            'id': f"{_a['id']}_{_i}", 'kind': 'wood', 'name': f'{WOOD_PARCEL_NAME} {_i}',
            'ring': _ring, 'tags': dict(_a['tags'], name=f'{WOOD_PARCEL_NAME} {_i}')})


# The shelterbelts are derived from the input rather than read out of it, like the
# railway: where they stand is decided by the fields and the roads above, and the
# fields are cut back to make room for them. `SHELTERBELTS` is the list on its own;
# they are also in `AREAS`, which is what the OSM draws.
SHELTERBELTS = build_shelterbelts(FIELDS) if FIELDS else []
AREAS.extend(SHELTERBELTS)


def corridors():
    return list(CORRIDORS)


def water():
    return list(WATER)


def water_axes():
    """(polyline, name) for every watercourse that has a centreline."""
    return [(w['axis'], w['name']) for w in WATER if w.get('axis')]


def pads():
    return list(PADS)


def areas():
    return list(AREAS)


def load_roughness(path=None):
    """Read `terrain_stats.json` and return a roughness lookup, or None if it is not
    there yet."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'dem_generator', 'terrain_stats.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    n = data['n']
    cell = data['cell_m']
    x0 = data['origin'][0]
    y0 = data['origin'][1]
    rough = data['roughness']

    def lookup(x, y):
        c = int((x - x0) // cell)
        r = int((y - y0) // cell)
        c = 0 if c < 0 else (n - 1 if c >= n else c)
        r = 0 if r < 0 else (n - 1 if r >= n else r)
        return rough[r * n + c]
    return lookup


# ==================================================================================
# self-check
# ==================================================================================
def validate():
    """Check the invariants that keep placement mistakes from reaching Giants Editor."""
    bad = []

    # The rim.
    if RIM_RAMP_M <= 0.0:
        bad.append(f"rim: the apron ({RIM_APRON_M:.0f} m) and the shoulder "
                   f"({RIM_BACK_M:.0f} m) leave no border to climb the flank over")

    ids = [r['id'] for r in CORRIDORS + WATER + PADS + AREAS]
    for i in sorted(set(ids)):
        if ids.count(i) > 1:
            bad.append(f"{i}: used by {ids.count(i)} records - ids must be unique")

    for c in CORRIDORS:
        if c['kind'] not in HIGHWAY_CLASS and c['kind'] != 'rail':
            bad.append(f"{c['id']}: unknown corridor class {c['kind']!r}")
        if len(c['axis']) < 2:
            bad.append(f"{c['id']}: an alignment needs at least two points")
        for x, y in c['axis']:
            if not (-0.5 <= x <= PLAYABLE_M + 0.5 and -0.5 <= y <= PLAYABLE_M + 0.5):
                bad.append(f"{c['id']}: point ({x:.1f}, {y:.1f}) outside playable bounds")
                break

    for p in PADS:
        if len(p['ring']) < 4 or math.dist(p['ring'][0], p['ring'][-1]) > 1e-6:
            bad.append(f"{p['id']}: pad ring does not close on its first point")

    for w in WATER:
        if not w.get('tags'):
            bad.append(f"{w['id']}: untagged water body")
        elif not any(k in w['tags'] and (v is None or w['tags'][k] == v) for k, v in RENDERED_TAGS):
            bad.append(f"{w['id']}: tagged {w['tags']} - neither renderer draws that")
        if w.get('ring'):
            if len(w['ring']) < 4 or math.dist(w['ring'][0], w['ring'][-1]) > 1e-6:
                bad.append(f"{w['id']}: ring does not close on its first point")
            for x, y in w['ring']:
                if not (-0.5 <= x <= PLAYABLE_M + 0.5 and -0.5 <= y <= PLAYABLE_M + 0.5):
                    bad.append(f"{w['id']}: point ({x:.1f}, {y:.1f}) outside playable bounds")
                    break

    for a in AREAS:
        if not a.get('tags'):
            bad.append(f"{a['id']}: untagged - both renderers would drop it")
        elif not any(k in a['tags'] and (v is None or a['tags'][k] == v) for k, v in RENDERED_TAGS):
            bad.append(f"{a['id']}: tagged {a['tags']} - neither renderer draws that")
        if len(a['ring']) < 4 or math.dist(a['ring'][0], a['ring'][-1]) > 1e-6:
            bad.append(f"{a['id']}: ring does not close on its first point")
        for x, y in a['ring']:
            if not (-0.5 <= x <= PLAYABLE_M + 0.5 and -0.5 <= y <= PLAYABLE_M + 0.5):
                bad.append(f"{a['id']}: point ({x:.1f}, {y:.1f}) outside playable bounds")
                break

    for a in AREAS:
        if not ring_is_simple(a['ring']):
            bad.append(f"{a['id']}: ring crosses itself - a fold the renderers would fill")

    _validate_shelterbelts(bad)
    return bad


def summary():
    """One-line description of the layout, for the generators to print."""
    if not (CORRIDORS or WATER or PADS or AREAS):
        return (f"{PLAYABLE_M:.0f} m playable on a {CANVAS_M:.0f} m canvas, no features "
                f"yet, flat at {BASE_ELEV_M:.0f} m inside a valley rim rising to "
                f"{RIM_CREST_M:.0f} m")
    woods = [a for a in AREAS if a.get('kind') == 'wood']
    belts = [a for a in AREAS if a.get('kind') == 'shelterbelt']
    water_str = f", {len(WATER)} water body" if len(WATER) == 1 else (f", {len(WATER)} water bodies" if len(WATER) > 1 else "")
    return (f"{PLAYABLE_M:.0f} m playable on a {CANVAS_M:.0f} m canvas, "
            f"{len(CORRIDORS)} roads, {len(PADS)} farmyards, {len(FIELDS)} fields, "
            f"{len(woods)} woods, {len(belts)} shelterbelts{water_str}, "
            f"datum {BASE_ELEV_M:.1f} m, "
            f"rim to {RIM_CREST_M:.0f} m")


if __name__ == '__main__':
    print("=== map_layout self-check ===")
    print("  ", summary())
    sw = global_to_local(*local_to_global(0.0, PLAYABLE_M))
    ne = global_to_local(*local_to_global(PLAYABLE_M, 0.0))
    print(f"   centre {LAT_CENTER:.4f}, {LON_CENTER:.4f}; the projection round-trips "
          f"the corners to ({sw[0]:.3f}, {sw[1]:.3f}) and ({ne[0]:.3f}, {ne[1]:.3f})")
    print(f"   canvas metres run {-OFFSET_M:.0f} .. {PLAYABLE_M + OFFSET_M:.0f}, "
          f"alignments out to {EDGE_MIN:.0f} .. {EDGE_MAX:.0f}")
    problems = validate()
    if problems:
        print("\n   PROBLEMS")
        for p in problems:
            print("    -", p)
    else:
        print("\n   layout is sound")
