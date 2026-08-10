#!/usr/bin/env python3
"""Classify the stage-6 floodplains: floodplain/basin, blobs and lakes.

Created on Fri Aug 7 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 7 (see README.md):
    reads   data/06_floodplains/floodplains.tif             (06_floodplains.py)
            data/04_accumulation/flow_accumulation_d8.tif   (04_flow_accumulation.py)
            data/03_flows/flow_direction_d8.tif             (03_flow_router.py)
            data/01_carved/*.tif                            (01_carve_dem.py)
    writes  data/07_classified/floodplains_classified.tif
            data/07_classified/floodplains_clean.tif

Stage 2's depression filling connects every low area to the stream
network, often only through a narrow spill route, and stage 6 then marks
the route and the area as floodplain. This stage cleans that mask up:
floodplain that has (or nearly has) stream contact, floodplain too far
from any channel to matter, and open water. Unlike stages 1-6, which
reproduce published methods, this classification is the author's own
method.

Method
------
Let fp = (floodplains == 1) and stream = (upstream area >= the
stream_threshold_km2 tag of the floodplain raster, so the classes are
always read against the network the floodplains were built with), both
restricted to the valid pixels of the floodplain raster.

1. A binary opening (disc of --radius pixels) of fp severs connections
   narrower than ~2*radius pixels. Opened components (8-connectivity)
   that touch a stream pixel are kept; stream pixels are added back
   (streams are floodplain by definition in stage 6), and the rim the
   opening shaved off is restored by exactly `radius` geodesic dilations
   within fp (full reconstruction would regrow through the severed
   connections).
2. Kept floodplain, plus every severed component within --dmax metres of
   a stream pixel (minimum Euclidean distance per component), is class 1.
   Severed components farther away are class 2 (blobs).
3. Lakes: KM2 hydro-flattens each water body to one constant elevation,
   so open water is a connected region of bit-exact equal values in the
   carved DEM - nothing natural is that flat at centimetre quantization.
   Interior pixels (3x3 min == max) seed the mask, two value-matched
   dilations recover the one-pixel rim and reunite bodies split at
   narrows, and components of at least --lake-min-ha hectares (default
   1 ha, the Finnish convention separating a lake from a pond) become
   class 3.
4. Lake shores: a floodplain pixel inherited its stage-6 flood level
   from its controlling stream pixel - the first stream pixel downstream
   along the D8 paths. Where that pixel lies inside a lake, the
   floodplain exists because of the lake, not a river, so it joins
   class 3 (found with the stage-3 D8 raster and the same pyflwdir flow
   graph stages 5-6 use). River floodplain at a lake's inlet and outlet
   keeps its class - its controlling stream pixels are river pixels.
   Classes 1 and 2 therefore describe river floodplain only; class 3 is
   stamped last, over any class.

Holes are never filled - they are real islands. The morphological
operators are standard mathematical morphology (Soille, 2004 -
``MORPHOLOGY_CITATION`` below).

Degenerate case: --radius 0 --lake-min-ha 0 reproduces the stage-6
raster as class 1/0 plus nodata in both outputs (the opening is the
identity and every stage-6 component reaches its stream through the
spill routes).

Output
------
Two int8, deflate-compressed GeoTIFFs on the stage-6 grid and CRS
(nothing is resampled or reprojected; distances assume a projected metre
CRS with square pixels).

``data/07_classified/floodplains_classified.tif``:

     0  dry land (not floodplain)
     1  potential floodplain/basin: floodplain with stream contact after
        the opening, or within --dmax metres of a stream pixel
     2  blob: floodplain farther than --dmax from any stream pixel
     3  lake: hydro-flattened open water of at least --lake-min-ha, plus
        its shore floodplain (controlling stream pixel inside a lake)
    -1  nodata (nodata in the stage-6 raster)

``data/07_classified/floodplains_clean.tif`` - class 1 alone as a binary
raster in the stage-6 encoding (1 = floodplain, 0 = other, -1 = nodata):
the GFPLAIN floodplain minus lakes, shores and blobs.

Credits
-------
* Source data: floodplain, flow-accumulation and carved DEM rasters from
  the earlier pipeline stages - presumed source: National Land Survey of
  Finland 2 m elevation model (KM2), CC BY 4.0.
* Tools that enabled this work: Python, NumPy (Harris et al., 2020),
  SciPy (Virtanen et al., 2020), Numba (Lam, Pitrou and Seibert, 2015),
  pyflwdir (Eilander et al., 2021), rasterio (Gillies et al.) on GDAL
  (GDAL/OGR contributors, OSGeo).

Usage (inside the ``water`` conda environment, ``conda activate water``)
-----
    python 07_classify.py                   # defaults from USER SETTINGS below
    python 07_classify.py --radius 3        # opening disc radius, pixels
    python 07_classify.py --dmax 100        # class-1/2 distance limit, metres
    python 07_classify.py --lake-min-ha 0   # disable the lake class
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
D8_RASTER = "data/03_flows/flow_direction_d8.tif"
                        # D8 flow directions (03_flow_router.py output),
                        # used to trace lake shores (only read when
                        # --lake-min-ha > 0)
CARVED_DIR = "data/01_carved"
                        # carved DEM tiles (01_carve_dem.py output), used
                        # by the lake detector (only read when
                        # --lake-min-ha > 0)
OUTPUTS_DIR = "data/07_classified"

# There is no --upa-min here: the stream threshold is read from the
# floodplain raster's stream_threshold_km2 tag (fallback: the shared
# UPA_MIN in pipeline_io.py), so the classes are always read against the
# network the floodplains were built with.

OPENING_RADIUS_PX = 3   # --radius: opening disc radius in pixels; severs
                        # floodplain connections narrower than ~2*radius
DMAX_M = 100.0          # --dmax: lateral distance to the nearest stream
                        # pixel splitting class 1 from class 2, in metres
LAKE_MIN_HA = 1.0       # --lake-min-ha: minimum area of a constant-
                        # elevation water surface to classify as lake, in
                        # hectares (1 ha is the Finnish lake/pond limit);
                        # 0 disables the lake class

# ========================== end of USER SETTINGS ===========================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from numba import njit
from scipy import ndimage as ndi

from pipeline_io import (
    NODATA, SOURCE_DATA_CREDIT_KNOWN, SOURCE_DATA_CREDIT_PRESUMED, UPA_MIN,
    build_flwdir, build_mosaic, collect_provenance, find_dems, load_d8,
    load_uparea, resolve_near, validate_tiles, write_raster,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FP_NODATA = -1          # nodata of the stage-6 raster and of the output

CLS_DRY = 0             # not floodplain
CLS_FLOOD = 1           # stream contact, or within --dmax of a stream
CLS_BLOB = 2            # floodplain farther than --dmax from any stream
CLS_LAKE = 3            # hydro-flattened open water >= --lake-min-ha,
                        # plus its shore floodplain

EIGHT = np.ones((3, 3), dtype=bool)     # 8-connectivity, for label/dilation

MORPHOLOGY_CITATION = (
    "Soille, P. (2004) Morphological Image Analysis: Principles and "
    "Applications. 2nd edn. Berlin: Springer, "
    "doi:10.1007/978-3-662-05088-0."
)

TOOL_CREDITS_CLASSIFY = (
    "Python, NumPy (Harris et al., 2020, doi:10.1038/s41586-020-2649-2), "
    "SciPy (Virtanen et al., 2020, doi:10.1038/s41592-019-0686-2), "
    "Numba (Lam, Pitrou and Seibert, 2015, doi:10.1145/2833157.2833162), "
    "pyflwdir (Eilander et al., 2021, doi:10.5194/hess-25-5287-2021), "
    "rasterio (Gillies et al.), GDAL (GDAL/OGR contributors, OSGeo)."
)


# ---------------------------------------------------------------------------
# Lake detection
# ---------------------------------------------------------------------------

def detect_lakes(carved_dir, transform, shape, crs, min_ha):
    """Detect hydro-flattened water surfaces on the carved DEM mosaic.

    KM2 flattens each water body to one constant elevation, so open
    water is a connected region of bit-exact equal values - nothing
    natural is that flat at centimetre quantization. Interior pixels
    (3x3 min == max) seed the mask; two value-matched dilations recover
    the one-pixel rim and reunite bodies split at narrows (bounded, so
    coincidentally equal shore pixels cannot leak far); components
    smaller than ``min_ha`` hectares are dropped. The mosaic must sit
    exactly on the floodplain raster's grid; fails hard otherwise.
    Returns ``(lakes, n_lakes, carved_paths)``.
    """
    paths = find_dems(None, carved_dir)
    print(f"carved DEM tiles ({len(paths)}): "
          + ", ".join(p.name for p in paths))
    validate_tiles(paths)
    elevtn, t, c = build_mosaic(paths)
    if (c != crs or elevtn.shape != shape
            or not t.almost_equals(transform, precision=1e-6)):
        sys.exit(f"{carved_dir}: the carved DEM mosaic is not on the "
                 f"floodplain raster's grid; the tiles and the floodplain "
                 f"raster must come from the same pipeline run")
    valid = elevtn != NODATA
    flat = (ndi.maximum_filter(elevtn, size=3)
            == ndi.minimum_filter(elevtn, size=3)) & valid
    for _ in range(2):
        near = ndi.maximum_filter(
            np.where(flat, elevtn, -np.inf), size=3)
        flat |= valid & ~flat & (elevtn == near)
    del valid
    labels, n = ndi.label(flat, structure=EIGHT)
    del flat
    pixel_ha = abs(transform.a * transform.e) / 1e4
    counts = np.bincount(labels.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = counts[1:] * pixel_ha >= min_ha
    lakes = keep[labels]
    del labels
    return lakes, int(keep.sum()), paths


@njit(cache=True)
def _drains_to_lake(idxs_ds, seq, stream, lake):
    """1 where the first stream pixel downstream (itself included) is in a
    lake - i.e. where the controlling stream pixel that stage 6 took the
    flood level from lies inside a lake."""
    out = np.zeros(stream.size, dtype=np.uint8)
    for i in range(seq.size):  # down- to upstream
        idx0 = seq[i]
        if stream[idx0]:
            out[idx0] = 1 if lake[idx0] else 0
        else:
            out[idx0] = out[idxs_ds[idx0]]
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def disk(radius):
    """Boolean disc structuring element: x2 + y2 <= r2 on a (2r+1)2 grid."""
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def classify(fp, stream, radius, dmax_m, pixel_m):
    """Split floodplain pixels into class 1 (floodplain/basin) and 2 (blob).

    ``fp`` and ``stream`` are boolean grids that must already exclude
    nodata pixels. Returns ``(classes, n_comp)``: an int8 grid of
    CLS_DRY/CLS_FLOOD/CLS_BLOB (the caller stamps lakes and nodata) and
    the component counts ``{1: .., 2: ..}``.
    """
    # The opening severs connections narrower than ~2*radius pixels;
    # components that keep stream contact are floodplain, the stream
    # pixels are added back (floodplain by definition in stage 6), and
    # the shaved rim is restored by exactly `radius` dilations within fp
    # (full reconstruction would regrow through the severed connections).
    opened = ndi.binary_opening(fp, structure=disk(radius))
    labels, n_labels = ndi.label(opened, structure=EIGHT)
    del opened
    touches = np.zeros(n_labels + 1, dtype=bool)
    touches[labels[stream]] = True
    touches[0] = False
    core = touches[labels]
    del labels
    core |= stream
    if radius > 0:
        core = ndi.binary_dilation(core, structure=EIGHT, iterations=radius,
                                   mask=fp)

    classes = np.zeros(fp.shape, dtype=np.int8)
    classes[core] = CLS_FLOOD

    # Severed components join class 1 within dmax of a stream pixel
    # (minimum Euclidean distance per component); farther ones are blobs.
    dist_m = ndi.distance_transform_edt(~stream) * pixel_m
    rest = fp & ~core
    del core
    labels, n_labels = ndi.label(rest, structure=EIGHT)
    del rest
    n_blob = 0
    if n_labels:
        mins = np.asarray(ndi.minimum(dist_m, labels=labels,
                                      index=np.arange(1, n_labels + 1)))
        lookup = np.concatenate((
            [np.int8(0)],
            np.where(mins <= dmax_m, CLS_FLOOD, CLS_BLOB).astype(np.int8),
        ))
        in_rest = labels > 0
        classes[in_rest] = lookup[labels[in_rest]]
        n_blob = int((mins > dmax_m).sum())
    del dist_m, labels

    n_comp = {CLS_FLOOD: int(ndi.label(classes == CLS_FLOOD,
                                       structure=EIGHT)[1]),
              CLS_BLOB: n_blob}
    return classes, n_comp


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # The USER SETTINGS block at the top of the script feeds the argparse
    # defaults directly, so there is exactly one source of truth per value.
    ap = argparse.ArgumentParser(
        description="Classify the stage-6 floodplains into potential "
                    "floodplain/basin (1), blobs far from any stream (2) "
                    "and lakes (3).")
    ap.add_argument("--floodplains", type=Path,
                    default=resolve_near(FLOODPLAINS_RASTER, HERE),
                    help="stage-6 floodplain raster (06_floodplains.py "
                         "output)")
    ap.add_argument("--uparea", type=Path,
                    default=resolve_near(UPAREA_RASTER, HERE),
                    help="D8 flow-accumulation raster "
                         "(04_flow_accumulation.py output)")
    ap.add_argument("--d8", type=Path,
                    default=resolve_near(D8_RASTER, HERE),
                    help="D8 flow-direction raster (03_flow_router.py "
                         "output; only read when --lake-min-ha > 0)")
    ap.add_argument("--carved-dir", type=Path,
                    default=resolve_near(CARVED_DIR, HERE),
                    help="carved DEM tiles (01_carve_dem.py output; only "
                         "read when --lake-min-ha > 0)")
    ap.add_argument("--outputs-dir", type=Path,
                    default=resolve_near(OUTPUTS_DIR, HERE))
    ap.add_argument("--radius", type=int, default=OPENING_RADIUS_PX,
                    metavar="PX",
                    help="opening disc radius in pixels: severs floodplain "
                         "connections narrower than ~2*radius "
                         f"(default {OPENING_RADIUS_PX})")
    ap.add_argument("--dmax", type=float, default=DMAX_M, metavar="M",
                    help="a severed floodplain component stays class 1 "
                         "within this distance of a stream pixel and is a "
                         f"blob (class 2) farther away (default {DMAX_M:g})")
    ap.add_argument("--lake-min-ha", type=float, default=LAKE_MIN_HA,
                    metavar="HA",
                    help="minimum area of a constant-elevation "
                         "(hydro-flattened) water surface to classify as "
                         "lake, in hectares; 0 disables the lake class "
                         f"(default {LAKE_MIN_HA:g})")
    args = ap.parse_args(argv)
    if args.radius < 0:
        ap.error("--radius must be >= 0")
    if args.dmax < 0:
        ap.error("--dmax must be >= 0")
    if args.lake_min_ha < 0:
        ap.error("--lake-min-ha must be >= 0 (0 disables the lake class)")

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

    classes, n_comp = classify(fp, stream, args.radius, args.dmax, px)

    carved_paths = None
    n_lakes = 0
    if args.lake_min_ha > 0:
        lakes, n_lakes, carved_paths = detect_lakes(
            args.carved_dir, transform, shape, crs, args.lake_min_ha)
        print(f"lakes: {n_lakes} water bodies >= {args.lake_min_ha:g} ha")
        # Shores: floodplain whose controlling stream pixel is in a lake.
        d8u8, _ = load_d8(args.d8, transform, shape, crs)
        flw = build_flwdir(d8u8, transform)
        del d8u8
        shore = _drains_to_lake(
            flw.idxs_ds, flw.idxs_seq,
            stream.ravel().astype(np.uint8),
            lakes.ravel().astype(np.uint8),
        ).reshape(shape).astype(bool)
        del flw
        shore &= fp & ~lakes
        print(f"lake shores: {int(shore.sum())} floodplain pixels whose "
              f"controlling stream pixel lies in a lake")
        classes[(lakes | shore) & valid] = CLS_LAKE
        del lakes, shore
    else:
        print("lake detection disabled (--lake-min-ha 0)")
    del stream
    classes[~valid] = FP_NODATA

    forwarded = collect_provenance([fp_path])
    tags = dict(
        title="Floodplain classification: floodplain/basin, blobs, lakes",
        classification_method=(
            "A binary opening (disc of radius classify_radius_px) severs "
            "floodplain connections narrower than ~2*radius; floodplain "
            "keeping stream contact, or within classify_dmax_m of a "
            "stream pixel, is class 1; farther floodplain is class 2; "
            "connected regions of constant carved elevation of at least "
            "classify_lake_min_ha hectares (KM2 hydro-flattened water "
            "surfaces), plus floodplain whose controlling stream pixel "
            "(first stream pixel downstream along D8) lies in a lake, "
            "are class 3, stamped last - classes 1 and 2 describe river "
            "floodplain only."),
        morphology_citation=MORPHOLOGY_CITATION,
        parameters=f"radius={args.radius} px, dmax={args.dmax:g} m, "
                   f"lake_min_ha={args.lake_min_ha:g} ha, "
                   f"upa_min={upa_min:g} km2",
        classify_radius_px=f"{args.radius}",
        classify_dmax_m=f"{args.dmax:g}",
        classify_lake_min_ha=f"{args.lake_min_ha:g}",
        stream_threshold_km2=f"{upa_min:g}",
        class_encoding="0 = dry land; 1 = potential floodplain/basin; "
                       "2 = blob (floodplain farther than dmax from any "
                       "stream pixel); 3 = lake (hydro-flattened open "
                       "water and its shore floodplain); -1 = nodata",
        source_floodplain_raster=fp_path.name,
        source_flow_accumulation_raster=args.uparea.name,
        source_data_credit=(SOURCE_DATA_CREDIT_KNOWN
                            if "dem_source_tiles" in forwarded
                            else SOURCE_DATA_CREDIT_PRESUMED),
        software_credits=TOOL_CREDITS_CLASSIFY,
        generated_by="07_classify.py",
        **forwarded,
    )
    if routing_alg is not None:
        tags["flow_routing_algorithm"] = routing_alg
    if carved_paths is not None:
        tags["source_carved_tiles"] = ", ".join(p.name for p in carved_paths)
        tags["source_flow_direction_raster"] = args.d8.name

    out_path = write_raster(
        args.outputs_dir / "floodplains_classified.tif", classes,
        transform, crs, nodata=FP_NODATA, dtype="int8", tags=tags,
    )

    # The clean binary companion: class 1 alone, in the stage-6 encoding.
    clean = (classes == CLS_FLOOD).astype(np.int8)
    clean[~valid] = FP_NODATA
    clean_tags = dict(tags)
    clean_tags["title"] = ("Cleaned geomorphic floodplains (GFPLAIN minus "
                           "lakes, shores and blobs)")
    clean_tags["class_encoding"] = "1 = floodplain, 0 = other, -1 = nodata"
    clean_tags["derived_from"] = ("class 1 of floodplains_classified.tif, "
                                  "same run and parameters")
    clean_path = write_raster(
        args.outputs_dir / "floodplains_clean.tif", clean,
        transform, crs, nodata=FP_NODATA, dtype="int8", tags=clean_tags,
    )

    pixel_ha = px * py / 1e4
    print(f"classification (radius={args.radius} px, dmax={args.dmax:g} m, "
          f"lake_min_ha={args.lake_min_ha:g} ha, upa_min={upa_min:g} km2):")
    names = {CLS_DRY: "dry land",
             CLS_FLOOD: "potential floodplain/basin",
             CLS_BLOB: "blob",
             CLS_LAKE: "lake and shore"}
    for value, name in names.items():
        n = int((classes == value).sum())
        line = f"  class {value} ({name}): {n} pixels = {n * pixel_ha:.1f} ha"
        if value in n_comp:
            line += f", {n_comp[value]} components"
        elif value == CLS_LAKE and n_lakes:
            line += f", {n_lakes} water bodies"
        print(line)
    print(f"  nodata: {int((~valid).sum())} pixels")
    n1 = int((clean == 1).sum())
    print(f"clean floodplain (class 1 only): {n1} pixels = "
          f"{n1 * pixel_ha:.1f} ha")
    print(f"-> {out_path}")
    print(f"-> {clean_path}  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
