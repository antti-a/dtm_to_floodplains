#!/usr/bin/env python3
"""Classify the stage-6 floodplains by fill depth and stream contact.

Created on Fri Aug 7 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 7 (see README.md):
    reads   data/06_floodplains/floodplains.tif             (06_floodplains.py)
            data/04_accumulation/flow_accumulation_d8.tif   (04_flow_accumulation.py)
            data/01_carved/*.tif                            (01_carve_dem.py)
            data/02_filled/*.tif                            (02_fill_dem.py)
    writes  data/07_classified/floodplains_classified.tif

Stage 2's depression filling raises every closed basin to its pour point,
which connects every low area to the stream network - often only through
a narrow spill route. This stage classifies the stage-6 floodplain by two
measurements of what that conditioning did: fill depth (filled - carved
DEM, i.e. how much the ground was raised) and contact with the stream
network through low-fill ground. It makes no claim about which floodplain
is genuine - the classes report terrain measurements to be checked
against ground truth. Unlike stages 1-6, which reproduce published
methods, this classification is the author's own method.

Method
------
Let fp = (floodplains == 1), stream = (upstream area >= the
stream_threshold_km2 tag of the floodplain raster, so the classes are
always read against the network the floodplains were built with), and
low = (fill depth < --fill-split metres), all restricted to the valid
pixels of the floodplain raster.

1. A binary opening (disc of --radius pixels) of fp & low severs
   connections narrower than ~2*radius pixels; high-fill ground is
   excluded from the start.
2. Opened components (8-connectivity) that touch a low-fill stream pixel
   are kept; low-fill stream pixels are added back (streams are
   floodplain by definition in stage 6), and the rim the opening shaved
   off is restored by exactly `radius` geodesic dilations within
   fp & low (full reconstruction would regrow through the severed
   connections). The result is class 1.
3. Every other floodplain pixel is classified per 8-connected component
   by its minimum Euclidean distance to a low-fill stream pixel:
   <= --dmax metres -> class 2, farther -> class 3.

Holes are never filled - they are real islands. The morphological
operators are standard mathematical morphology (Soille, 2004 -
``MORPHOLOGY_CITATION`` below). Fill depth is measured against the
carved DEM, so it overstates basins wherever a real flow path (an
unmapped culvert, a bridge) is missing from the stage-1 carve data.

Degenerate cases: --fill-split inf treats all ground as low-fill (the
carved/filled tiles are then not read), and --radius 0 --fill-split inf
reproduces the stage-6 raster as class 1/0 plus nodata.

Output
------
``data/07_classified/floodplains_classified.tif`` - int8,
deflate-compressed GeoTIFF on the stage-6 grid and CRS (nothing is
resampled or reprojected; distances assume a projected metre CRS with
square pixels):

     0  upland (not floodplain)
     1  floodplain with fill depth < --fill-split, in contact with the
        stream network through low-fill ground
     2  floodplain without such contact - high-fill ground, or low-fill
        ground whose connection was severed - within --dmax metres of a
        low-fill stream pixel
     3  as class 2, but farther than --dmax
    -1  nodata (nodata in the stage-6 raster)

Credits
-------
* Source data: floodplain, flow-accumulation and DEM rasters from the
  earlier pipeline stages - presumed source: National Land Survey of
  Finland 2 m elevation model (KM2), CC BY 4.0.
* Tools that enabled this work: Python, NumPy (Harris et al., 2020),
  SciPy (Virtanen et al., 2020), rasterio (Gillies et al.) on GDAL
  (GDAL/OGR contributors, OSGeo).

Usage (inside the ``water`` conda environment, ``conda activate water``)
-----
    python 07_classify.py                   # defaults from USER SETTINGS below
    python 07_classify.py --radius 3        # opening disc radius, pixels
    python 07_classify.py --dmax 100        # class-2/3 distance split, metres
    python 07_classify.py --fill-split 0.5  # class-1/2 fill-depth split, metres
"""

from __future__ import annotations

# ===========================================================================
# USER SETTINGS - these feed the argparse defaults, so they apply to a
#                 no-argument run; any CLI flag overrides them
# ===========================================================================

FLOODPLAINS_RASTER = "data/06_floodplains/floodplains.tif"
                        # stage-6 binary floodplain raster (1/0/-1);
                        # relative paths are resolved next to this script
UPAREA_RASTER = "data/04_accumulation/flow_accumulation_d8.tif"
                        # D8 flow accumulation (04_flow_accumulation.py
                        # output); its m2/pixels units tag is honoured
CARVED_DIR = "data/01_carved"
                        # carved DEM tiles (01_carve_dem.py output)
FILLED_DIR = "data/02_filled"
                        # filled DEM tiles (02_fill_dem.py output); their
                        # difference to the carved tiles is the fill depth
                        # (both folders are only read when --fill-split
                        # is finite)
OUTPUTS_DIR = "data/07_classified"

# There is no --upa-min here: the stream threshold is read from the
# floodplain raster's stream_threshold_km2 tag (fallback: the shared
# UPA_MIN in pipeline_io.py), so the classes are always read against the
# network the floodplains were built with.

OPENING_RADIUS_PX = 3   # --radius: opening disc radius in pixels; severs
                        # floodplain connections narrower than ~2*radius
DMAX_M = 100.0          # --dmax: lateral distance to the nearest low-fill
                        # stream pixel splitting class 2 from class 3, m
FILL_SPLIT_M = 1.0      # --fill-split: the class-1/2 boundary in metres
                        # of fill depth (filled - carved DEM); only ground
                        # below it can carry stream contact; inf treats
                        # all ground as low-fill

# ========================== end of USER SETTINGS ===========================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

from pipeline_io import (
    SOURCE_DATA_CREDIT_KNOWN, SOURCE_DATA_CREDIT_PRESUMED, UPA_MIN,
    build_mosaic, collect_provenance, find_dems, load_uparea, resolve_near,
    validate_tiles, write_raster,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FP_NODATA = -1          # nodata of the stage-6 raster and of the output

CLS_UPLAND = 0          # not floodplain
CLS_CONTACT = 1         # low-fill floodplain in contact with the network
CLS_NEAR = 2            # no low-fill contact, within --dmax of a stream
CLS_FAR = 3             # no low-fill contact, farther than --dmax

EIGHT = np.ones((3, 3), dtype=bool)     # 8-connectivity, for label/dilation

MORPHOLOGY_CITATION = (
    "Soille, P. (2004) Morphological Image Analysis: Principles and "
    "Applications. 2nd edn. Berlin: Springer, "
    "doi:10.1007/978-3-662-05088-0."
)

TOOL_CREDITS_SCIPY = (
    "Python, NumPy (Harris et al., 2020, doi:10.1038/s41586-020-2649-2), "
    "SciPy (Virtanen et al., 2020, doi:10.1038/s41592-019-0686-2), "
    "rasterio (Gillies et al.), GDAL (GDAL/OGR contributors, OSGeo)."
)


# ---------------------------------------------------------------------------
# Fill depth
# ---------------------------------------------------------------------------

def load_fill_depth(carved_dir, filled_dir, transform, shape, crs):
    """Mosaic the carved and filled DEM tiles; return the fill depth grid.

    Fill depth = filled - carved elevation, in metres: how much stage 2's
    depression filling raised each pixel toward its pour point. float32
    is fine for the difference - the 1e-8 flat-resolution gradients are
    noise at --fill-split scale. Both mosaics must sit exactly on the
    floodplain raster's grid; fails hard otherwise. Returns
    ``(fill_depth, carved_paths, filled_paths)``.
    """
    mosaics, path_lists = [], []
    for what, folder in (("carved", carved_dir), ("filled", filled_dir)):
        paths = find_dems(None, folder)
        print(f"{what} DEM tiles ({len(paths)}): "
              + ", ".join(p.name for p in paths))
        validate_tiles(paths)
        elevtn, t, c = build_mosaic(paths)
        if (c != crs or elevtn.shape != shape
                or not t.almost_equals(transform, precision=1e-6)):
            sys.exit(f"{folder}: the {what} DEM mosaic is not on the "
                     f"floodplain raster's grid; the tiles and the "
                     f"floodplain raster must come from the same "
                     f"pipeline run")
        mosaics.append(elevtn)
        path_lists.append(paths)
    fill_depth = mosaics[1] - mosaics[0]
    del mosaics
    return fill_depth, path_lists[0], path_lists[1]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def disk(radius):
    """Boolean disc structuring element: x2 + y2 <= r2 on a (2r+1)2 grid."""
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def classify(fp, fp_low, stream_low, radius, dmax_m, pixel_m):
    """Classify floodplain pixels by fill depth and stream contact.

    ``fp``, ``fp_low`` and ``stream_low`` are boolean grids that must
    already exclude nodata pixels: ``fp_low``/``stream_low`` are the
    floodplain and stream masks restricted to low-fill ground (fill
    depth below the --fill-split threshold; pass ``fp_low = fp`` and
    ``stream_low = stream`` to disable the split). Only low-fill ground
    carries stream contact. Returns ``(classes, n_comp)``: an int8 grid
    of CLS_UPLAND/CLS_CONTACT/CLS_NEAR/CLS_FAR (the caller stamps
    nodata) and the per-class component counts ``{1: .., 2: .., 3: ..}``.
    """
    # 1. The opening severs connections narrower than ~2*radius pixels;
    #    it runs on the low-fill floodplain, so high-fill ground is
    #    excluded from the start.
    opened = ndi.binary_opening(fp_low, structure=disk(radius))

    # 2. Keep the opened components that touch a low-fill stream pixel;
    #    add the low-fill stream pixels back (streams are floodplain by
    #    definition in stage 6, even where the corridor was thinner than
    #    the disc); restore the shaved rim with exactly `radius` dilations
    #    constrained to the low-fill floodplain (full reconstruction would
    #    regrow through the severed connections).
    labels, n_labels = ndi.label(opened, structure=EIGHT)
    del opened
    touches = np.zeros(n_labels + 1, dtype=bool)
    touches[labels[stream_low]] = True
    touches[0] = False
    core = touches[labels]
    del labels
    core |= stream_low
    if radius > 0:
        core = ndi.binary_dilation(core, structure=EIGHT, iterations=radius,
                                   mask=fp_low)

    classes = np.zeros(fp.shape, dtype=np.int8)
    classes[core] = CLS_CONTACT
    n_comp = {CLS_CONTACT: int(ndi.label(core, structure=EIGHT)[1]),
              CLS_NEAR: 0, CLS_FAR: 0}

    # 3. Every other floodplain component: minimum Euclidean distance to
    #    a low-fill stream pixel <= dmax -> class 2, farther -> class 3.
    dist_m = ndi.distance_transform_edt(~stream_low) * pixel_m
    rest = fp & ~core
    del core
    labels, n_labels = ndi.label(rest, structure=EIGHT)
    del rest
    if n_labels:
        mins = np.asarray(ndi.minimum(dist_m, labels=labels,
                                      index=np.arange(1, n_labels + 1)))
        lookup = np.zeros(n_labels + 1, dtype=np.int8)
        lookup[1:] = np.where(mins <= dmax_m, CLS_NEAR, CLS_FAR)
        n_comp[CLS_NEAR] = int((mins <= dmax_m).sum())
        n_comp[CLS_FAR] = n_labels - n_comp[CLS_NEAR]
        in_rest = labels > 0
        classes[in_rest] = lookup[labels[in_rest]]
    return classes, n_comp


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # The USER SETTINGS block at the top of the script feeds the argparse
    # defaults directly, so there is exactly one source of truth per value.
    ap = argparse.ArgumentParser(
        description="Classify the stage-6 floodplains by fill depth "
                    "(filled - carved DEM) and contact with the stream "
                    "network through low-fill ground.")
    ap.add_argument("--floodplains", type=Path,
                    default=resolve_near(FLOODPLAINS_RASTER, HERE),
                    help="stage-6 floodplain raster (06_floodplains.py "
                         "output)")
    ap.add_argument("--uparea", type=Path,
                    default=resolve_near(UPAREA_RASTER, HERE),
                    help="D8 flow-accumulation raster "
                         "(04_flow_accumulation.py output)")
    ap.add_argument("--carved-dir", type=Path,
                    default=resolve_near(CARVED_DIR, HERE),
                    help="carved DEM tiles (01_carve_dem.py output; only "
                         "read when --fill-split is finite)")
    ap.add_argument("--filled-dir", type=Path,
                    default=resolve_near(FILLED_DIR, HERE),
                    help="filled DEM tiles (02_fill_dem.py output; only "
                         "read when --fill-split is finite)")
    ap.add_argument("--outputs-dir", type=Path,
                    default=resolve_near(OUTPUTS_DIR, HERE))
    ap.add_argument("--radius", type=int, default=OPENING_RADIUS_PX,
                    metavar="PX",
                    help="opening disc radius in pixels: severs floodplain "
                         "connections narrower than ~2*radius "
                         f"(default {OPENING_RADIUS_PX})")
    ap.add_argument("--dmax", type=float, default=DMAX_M, metavar="M",
                    help="a floodplain component without low-fill stream "
                         "contact is class 2 within this distance of a "
                         "low-fill stream pixel, class 3 farther away "
                         f"(default {DMAX_M:g})")
    ap.add_argument("--fill-split", type=float, default=FILL_SPLIT_M,
                    metavar="M",
                    help="the class-1/2 boundary in metres of fill depth "
                         "(filled - carved DEM): only ground below it can "
                         "carry stream contact; inf treats all ground as "
                         f"low-fill (default {FILL_SPLIT_M:g})")
    args = ap.parse_args(argv)
    if args.radius < 0:
        ap.error("--radius must be >= 0")
    if args.dmax < 0:
        ap.error("--dmax must be >= 0")
    if args.fill_split < 0:
        ap.error("--fill-split must be >= 0 (or inf)")

    t0 = time.perf_counter()
    fp_path = args.floodplains
    if not fp_path.is_file():
        sys.exit(f"Floodplain raster not found: {fp_path}. "
                 f"Run 06_floodplains.py first.")
    with rasterio.open(fp_path) as src:
        transform, crs = src.transform, src.crs
        shape = (src.height, src.width)
        fp_tags = src.tags()
        fp_nodata = FP_NODATA if src.nodata is None else int(src.nodata)
        fldpln = src.read(1)
    px, py = abs(transform.a), abs(transform.e)
    if abs(px - py) > 1e-9:
        sys.exit(f"{fp_path}: non-square pixels ({px:g} x {py:g} m); the "
                 f"distance classification assumes square pixels")
    print(f"floodplain raster {shape[1]} x {shape[0]} pixels, "
          f"pixel {px:g} x {py:g} m")

    # No --upa-min here: the classes are read against the network the
    # floodplains were built with, via the stage-6 raster's tag.
    routing_alg = fp_tags.get("flow_routing_algorithm")
    try:
        upa_min = float(fp_tags["stream_threshold_km2"])
    except (KeyError, ValueError):
        upa_min = UPA_MIN
        print(f"WARNING: {fp_path.name} carries no stream_threshold_km2 "
              f"tag; assuming the shared default {UPA_MIN:g} km2")

    valid = fldpln != fp_nodata
    fp = fldpln == 1
    del fldpln

    uparea = load_uparea(args.uparea, transform, shape, crs, routing_alg)
    stream = (uparea >= np.float32(upa_min)) & valid
    del uparea
    n_stream = int(stream.sum())
    print(f"stream pixels: {n_stream} (upstream area >= {upa_min:g} km2, "
          f"from the floodplain raster's stream_threshold_km2 tag)")
    if n_stream == 0:
        sys.exit(f"no pixel reaches the stream threshold {upa_min:g} km2; "
                 f"there is no stream network to classify against")

    carved_paths = filled_paths = None
    if np.isfinite(args.fill_split):
        fill_depth, carved_paths, filled_paths = load_fill_depth(
            args.carved_dir, args.filled_dir, transform, shape, crs)
        low = fill_depth < np.float32(args.fill_split)
        del fill_depth
        stream_low = stream & low
        fp_low = fp & low
        del low
        n_low = int(stream_low.sum())
        print(f"stream pixels on low-fill ground "
              f"(fill depth < {args.fill_split:g} m): {n_low} of {n_stream}")
        n_fp = int(fp.sum())
        n_high = n_fp - int(fp_low.sum())
        print(f"floodplain pixels on high-fill ground: {n_high} of {n_fp} "
              f"({100 * n_high / max(n_fp, 1):.1f}%)")
        if n_low == 0:
            sys.exit(f"--fill-split {args.fill_split:g} m leaves no "
                     f"low-fill stream pixel; nothing to classify against")
    else:
        stream_low = stream
        fp_low = fp
        print("fill-depth split disabled (--fill-split inf): all ground "
              "treated as low-fill")
    del stream

    classes, n_comp = classify(fp, fp_low, stream_low, args.radius,
                               args.dmax, px)
    classes[~valid] = FP_NODATA

    forwarded = collect_provenance([fp_path])
    tags = dict(
        title="Floodplain classification by fill depth and stream contact",
        classification_method=(
            "Fill depth = filled - carved DEM; ground below "
            "classify_fill_split_m is low-fill; a binary opening (disc of "
            "radius classify_radius_px) of the low-fill floodplain severs "
            "narrow connections; components touching a low-fill stream "
            "pixel, plus the low-fill stream pixels, geodesically "
            "re-dilated radius steps within the low-fill floodplain, form "
            "class 1; every other floodplain component is class 2 when "
            "its minimum Euclidean distance to a low-fill stream pixel "
            "is <= classify_dmax_m, else class 3."),
        morphology_citation=MORPHOLOGY_CITATION,
        parameters=f"radius={args.radius} px, dmax={args.dmax:g} m, "
                   f"fill_split={args.fill_split:g} m, "
                   f"upa_min={upa_min:g} km2",
        classify_radius_px=f"{args.radius}",
        classify_dmax_m=f"{args.dmax:g}",
        classify_fill_split_m=f"{args.fill_split:g}",
        stream_threshold_km2=f"{upa_min:g}",
        class_encoding="0 = upland; 1 = floodplain, fill depth < "
                       "fill_split, stream contact through low-fill "
                       "ground; 2 = floodplain without such contact, "
                       "within dmax of a low-fill stream pixel; 3 = as "
                       "2 but farther; -1 = nodata",
        source_floodplain_raster=fp_path.name,
        source_flow_accumulation_raster=args.uparea.name,
        source_data_credit=(SOURCE_DATA_CREDIT_KNOWN
                            if "dem_source_tiles" in forwarded
                            else SOURCE_DATA_CREDIT_PRESUMED),
        software_credits=TOOL_CREDITS_SCIPY,
        generated_by="07_classify.py",
        **forwarded,
    )
    if routing_alg is not None:
        tags["flow_routing_algorithm"] = routing_alg
    if carved_paths is not None:
        tags["source_carved_tiles"] = ", ".join(p.name for p in carved_paths)
        tags["source_filled_tiles"] = ", ".join(p.name for p in filled_paths)

    out_path = write_raster(
        args.outputs_dir / "floodplains_classified.tif", classes,
        transform, crs, nodata=FP_NODATA, dtype="int8", tags=tags,
    )

    pixel_ha = px * py / 1e4
    print(f"classification (radius={args.radius} px, dmax={args.dmax:g} m, "
          f"fill_split={args.fill_split:g} m, upa_min={upa_min:g} km2):")
    names = {CLS_UPLAND: "upland",
             CLS_CONTACT: "low fill, stream contact",
             CLS_NEAR: "no contact, within dmax",
             CLS_FAR: "no contact, beyond dmax"}
    for value, name in names.items():
        n = int((classes == value).sum())
        line = f"  class {value} ({name}): {n} pixels = {n * pixel_ha:.1f} ha"
        if value in n_comp:
            line += f", {n_comp[value]} components"
        print(line)
    print(f"  nodata: {int((~valid).sum())} pixels")
    print(f"-> {out_path}  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
