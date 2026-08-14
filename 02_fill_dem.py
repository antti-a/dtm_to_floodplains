#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrologically condition (fill or breach) DEMs.

Created on Fri Jul 3 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 2 (see README.md):
    reads   data/01_carved/*.tif        (01_carve_dem.py output)
    writes  data/02_filled/filled_<name>.tif            (--method fill, default)
        or  data/02_breached/breached_<name>.tif        (--method breach)
            data/02_breached/depth/breach_depth.tif     (breach diagnostic)
of which the fill output is the default input of the next stage
(03_flow_router.py); the breach output feeds the same stages through their
``--inputs-dir`` flags (see the A/B commands below).

For every GeoTIFF DEM in the input folder, removes the artefacts that
break downstream flow routing. With ``--method fill`` (the default):

  1. fill pits and depressions    (pysheds ``fill_depressions``, the
                                   priority-flood of Barnes, Lehman and
                                   Mulla (2014a); a single-pixel pit is just
                                   a one-pixel depression, so one pass
                                   removes both)
  2. resolve flats                (pysheds ``resolve_flats``, the flat-
                                   resolution method of Barnes, Lehman and
                                   Mulla (2014b): a tiny gradient is baked
                                   into the elevations of filled flats, so
                                   the output DEM itself drains -- not just
                                   some side-channel flow-direction raster)

With ``--method breach`` (experimental), depressions are instead removed by
complete breaching (the carving of Soille, Vogt and Colombo (2003); the
"complete breaching" of Lindsay (2016)): a priority flood from the outlets
records how it reached each pixel, and when it pops a pixel with no lower
neighbour -- an undrainable depression floor -- the pixels along the flood's
path back to that pixel are lowered to its elevation, cutting a level trench
through the depression's barrier. Filling raises a depression's floor to its
spill elevation; breaching keeps the floor and lowers the barrier instead.
That distinction is the mode's purpose: on a filled DEM the floodplain stage
sees basin floors at spill level and marks whole basins as floodplain, on a
breached DEM the basins keep their true depth. ``--fill-limit`` blends the
two: depressions at most that deep (spill minus floor) are filled, deeper
ones are breached, so centimetre-scale noise pits do not each cut a trench.
The trench pixels are recorded in ``depth/breach_depth.tif`` (metres
lowered, > 0 exactly where the breach carved) on the mosaic grid.

The inputs are mosaicked *virtually* with ``gdalbuildvrt`` and conditioned as
one surface, so depressions spanning tile edges are handled correctly
(depression removal is a global operation and cannot be done lazily inside
the VRT itself). The conditioned mosaic is then cropped back onto each
input's exact grid and written per tile -- the same DEMs, but conditioned.

The outputs are float64 on purpose. Source elevations are quantized to
the centimetre, so flat terrain is full of exactly tied pixels; the
flat-resolution gradient that makes those ties drain is far smaller than
float32 can represent at these elevations, and a float32 output silently
collapses the flats right back -- downstream flow routing then dies on a
DEM that merely looks conditioned. float64 keeps the gradients, and both
modes verify that the written mosaic actually drains.

Notes:
  * Breach trenches are carved level and rely on ``resolve_flats`` for
    their drainage gradient, exactly as filled depressions do (a per-step
    carve gradient would gouge the hydro-flattened lakes of the source
    DEM, polluting the breach-depth diagnostic).
  * Depressions draining across the mosaic's outer edge condition to the
    edge elevation ("outlets at edge"). The modes treat interior nodata
    holes differently: the fill is seeded from the outermost valid pixel
    of each row and column, so a depression that drains only into an
    interior nodata hole fills up to the hole's surrounding rim; the
    breach seeds every pixel adjacent to nodata, so the same depression
    keeps its floor and drains into the hole -- consistent with how this
    stage's drainage check and stage 3's treat nodata (as an outlet).
  * Pixels that are nodata in an input stay nodata in its output.
  * All inputs must share CRS, pixel size and grid alignment (validated).
  * Breach mode is experimental (an A/B against fill); downstream stages
    default to the fill outputs, so the breach variant is run manually:

        python 02_fill_dem.py --method breach --fill-limit 0.2
        python 03_flow_router.py --inputs-dir data/02_breached \\
            --outputs-dir data/03_flows_breach
        python 04_flow_accumulation.py --inputs-dir data/03_flows_breach \\
            --outputs-dir data/04_accumulation_breach
        python 05_hand.py --inputs-dir data/02_breached \\
            --outputs-dir data/05_hand_breach \\
            --d8 data/03_flows_breach/flow_direction_d8.tif \\
            --uparea data/04_accumulation_breach/flow_accumulation_d8.tif
        python 06_floodplains.py --inputs-dir data/02_breached \\
            --outputs-dir data/06_floodplains_breach \\
            --d8 data/03_flows_breach/flow_direction_d8.tif \\
            --uparea data/04_accumulation_breach/flow_accumulation_d8.tif

References:
  * Barnes, R., Lehman, C. and Mulla, D. (2014a) 'Priority-flood: an
    optimal depression-filling and watershed-labeling algorithm for
    digital elevation models', Computers & Geosciences, 62, pp. 117-127.
  * Barnes, R., Lehman, C. and Mulla, D. (2014b) 'An efficient assignment
    of drainage direction over flat surfaces in raster digital elevation
    models', Computers & Geosciences, 62, pp. 128-135.
  * Bartos, M. (2020) pysheds: simple and fast watershed delineation in
    python. doi:10.5281/zenodo.3822494
  * Lindsay, J.B. (2016) 'Efficient hybrid breaching-filling sink removal
    methods in raster digital elevation models', Hydrological Processes,
    30(6), pp. 846-857. doi:10.1002/hyp.10648
  * Soille, P., Vogt, J. and Colombo, R. (2003) 'Carving and adaptive
    drainage enforcement of grid digital elevation models', Water
    Resources Research, 39(12), 1366. doi:10.1029/2002WR001879

Run inside the ``water`` conda environment:

    conda activate water
    python 02_fill_dem.py
    python 02_fill_dem.py --method breach --fill-limit 0.2
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from numba import njit
from rasterio.windows import from_bounds

logger = logging.getLogger("fill_dem")

# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
NODATA = -9999.0             # KM2 convention; also stamped on the outputs
KEEP_INTERMEDIATE = False    # keep outputs/_work/ after a successful run

# Flat-resolution parameters (pysheds resolve_flats). Each flat pixel is
# raised by FLAT_EPS times its Barnes drainage-gradient count, which grows
# up to ~3 * FLAT_MAX_ITER steps. The values are chosen so even the widest
# resolvable flat inflates by at most 3 mm -- well under the source DEM's
# 1 cm quantization -- while each single step (1e-8 m) stays orders of
# magnitude above float64 resolution at Finnish elevations (~1e-13 m at
# 100 m). FLAT_MAX_ITER must exceed the widest flat in pixels; 100 000
# pixels = 200 km at 2 m, far beyond any filled lake here. (The pysheds
# defaults, eps=1e-5 and max_iter=1000, would inflate big flats by metres
# and leave flats wider than 2 km unresolved.)
FLAT_EPS = 1e-8
FLAT_MAX_ITER = 100_000

# The most resolve_flats can raise any pixel (see the comment above): the
# breach mode's crop-back guard allows raising up to this plus the fill
# limit, and anything more means the breach itself raised terrain -- a bug.
MAX_FLAT_INFLATION = 3 * FLAT_EPS * FLAT_MAX_ITER   # 3 mm

# Breach mode: depressions at most this deep (spill minus floor, metres)
# are filled instead of breached, so noise pits do not each cut a trench.
# 0 breaches every depression; the CLI flag --fill-limit overrides.
FILL_LIMIT = 0.0

# Used when the input carries stage-1 provenance tags (dem_source_tiles).
SOURCE_DATA_CREDIT = (
    "National Land Survey of Finland 2 m elevation model (KM2), CC BY 4.0, "
    "carved with the SYKE 'Tierumpujen uomakorjaus' culvert correction "
    "(CC BY 4.0); source tiles in the dem_source_tiles tag."
)
# Fallback for pre-convention inputs without tags.
SOURCE_DATA_CREDIT_PRESUMED = (
    "2 m DEM in EPSG:3067 (ETRS89 / TM35FIN), vertical datum N2000; "
    "presumed source: National Land Survey of Finland 2 m elevation model "
    "(KM2), CC BY 4.0."
)

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
INPUT_DIR = DATA_DIR / "01_carved"    # carved DEMs (01_carve_dem.py output)
OUT_DIR = DATA_DIR / "02_filled"      # filled DEMs -> 03_flow_router.py input
WORK_DIR = OUT_DIR / "_work"          # mosaic + fill intermediates
OUT_DIR_BREACH = DATA_DIR / "02_breached"   # breached DEMs (--method breach);
                                            # depth/ + _work/ subfolders stay
                                            # out of the *.tif tile globs


# --------------------------------------------------------------------------- #
# GDAL VRT helpers (mosaic the input DEMs virtually)
# --------------------------------------------------------------------------- #
def _gdalbuildvrt_exe() -> str:
    """Locate the ``gdalbuildvrt`` executable shipped with the conda env."""
    exe = shutil.which("gdalbuildvrt")
    if exe:
        return exe
    # Windows-only fallback: on conda/Windows the GDAL CLI tools live in
    # <env>/Library/bin. On Linux/macOS shutil.which() above finds them.
    cand = Path(sys.executable).parent / "Library" / "bin" / "gdalbuildvrt.exe"
    if cand.exists():
        return str(cand)
    raise RuntimeError(
        "gdalbuildvrt not found -- run inside the 'water' conda env."
    )


def build_vrt(
    tiles: Sequence[Path], vrt_path: Path, nodata: float = NODATA
) -> Path:
    """Mosaic ``tiles`` virtually into ``vrt_path`` (no physical merge)."""
    vrt_path = Path(vrt_path)
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _gdalbuildvrt_exe(), "-overwrite", str(vrt_path),
        *[str(t) for t in tiles],
    ]
    logger.info("Building VRT from %d tile(s) -> %s", len(tiles), vrt_path)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    with rasterio.open(vrt_path) as src:
        if src.nodata is None:
            logger.warning(
                "VRT has no nodata; downstream reads assume %s", nodata
            )
        elif src.nodata != nodata:
            logger.warning("VRT nodata %s != expected %s", src.nodata, nodata)
    return vrt_path


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_inputs(dems: Sequence[Path]) -> None:
    """Require one shared CRS, pixel size and grid lattice across ``dems``.

    Filling runs on a mosaic of all inputs, so a tile on a shifted grid or in
    another CRS would be silently resampled by the VRT -- fail instead. Also
    warns on overlapping tiles (gdalbuildvrt stacks them last-wins).
    """
    infos = []
    for path in dems:
        with rasterio.open(path) as src:
            infos.append((path.name, src.crs, src.transform, src.bounds))

    name0, crs0, t0, _ = infos[0]
    res = (abs(t0.a), abs(t0.e))
    for name, crs, t, _ in infos[1:]:
        if crs != crs0:
            raise ValueError(f"{name}: CRS {crs} != {crs0} ({name0})")
        if (abs(t.a), abs(t.e)) != res:
            raise ValueError(
                f"{name}: pixel size {(abs(t.a), abs(t.e))} != {res} ({name0})"
            )
        dx = (t.c - t0.c) / t.a
        dy = (t.f - t0.f) / t.e
        if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
            raise ValueError(
                f"{name}: grid origin misaligned with {name0} by "
                f"({dx % 1:.6f}, {dy % 1:.6f}) pixels"
            )

    for i, (name_a, _, _, ba) in enumerate(infos):
        for name_b, _, _, bb in infos[i + 1:]:
            if (ba.left < bb.right and bb.left < ba.right
                    and ba.bottom < bb.top and bb.bottom < ba.top):
                logger.warning(
                    "Tiles %s and %s overlap; VRT keeps the later one "
                    "in the overlap", name_a, name_b,
                )


# --------------------------------------------------------------------------- #
# Mosaic materialisation
# --------------------------------------------------------------------------- #
def materialize_mosaic(vrt_path: Path, out_tif: Path,
                       nodata: float = NODATA) -> Path:
    """Copy the virtual mosaic into a real GeoTIFF for the fill.

    Filling is a global operation over one in-memory surface anyway, so
    reading the mosaic once here costs nothing extra and normalizes the
    nodata value and any non-finite pixels before the fill sees them.
    """
    with rasterio.open(vrt_path) as src:
        dem = src.read(1).astype("float64")
        if src.nodata is not None and src.nodata != nodata:
            dem[dem == src.nodata] = nodata
        dem[~np.isfinite(dem)] = nodata
        logger.info(
            "Mosaic grid: %d x %d pixels @ %g m, bounds %s",
            src.width, src.height, src.transform.a, tuple(src.bounds),
        )
        return write_dem(
            out_tif, dem, src.transform, src.crs, nodata, dtype="float64",
        )


# --------------------------------------------------------------------------- #
# The fill (pysheds)
# --------------------------------------------------------------------------- #
def count_undrainable(dem_tif: Path, nodata: float = NODATA) -> tuple[int, int]:
    """Count interior pixels with no strictly lower neighbour.

    Such a pixel (a pit or a flat tie) stops every flow-routing algorithm.
    Nodata pixels count as outlets (like the fill treats them) and edge
    pixels drain off-grid, so a healthy conditioned DEM leaves only a
    scattered handful. Returns (undrainable_pixels, valid_pixels).
    """
    with rasterio.open(dem_tif) as src:
        z = src.read(1).astype("float64")
        if src.nodata is not None:
            nodata = src.nodata
    valid = np.isfinite(z) & (z != nodata)
    z[~valid] = -np.inf  # nodata is an outlet: neighbours drain into it
    centre = z[1:-1, 1:-1]
    has_lower = np.zeros(centre.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                has_lower |= (
                    z[1 + dr:z.shape[0] - 1 + dr,
                      1 + dc:z.shape[1] - 1 + dc] < centre
                )
    stuck = valid[1:-1, 1:-1] & ~has_lower
    return int(stuck.sum()), int(valid.sum())


def fill(mosaic_tif: Path, work_dir: Path = WORK_DIR) -> Path:
    """Fill pits + depressions and resolve flats; return the filled mosaic.

    ``fill_depressions`` is a priority-flood fill (Barnes, Lehman and
    Mulla, 2014a) that removes single-pixel pits along with larger
    depressions; ``resolve_flats`` (Barnes, Lehman and Mulla, 2014b)
    applies a tiny gradient (``FLAT_EPS`` per step, at most
    ``FLAT_MAX_ITER`` steps) across flats so the result drains
    everywhere. That gradient only survives the file round-trip in
    float64 (see the module docstring), so the written mosaic is checked
    here: this function fails loudly rather than hand a non-draining
    "filled" DEM downstream.

    The nodata mask is taken before filling (``fill_depressions`` mutates
    its input in place) and re-applied afterwards: ``resolve_flats`` does
    not exclude nodata regions, whose constant elevation reads as one big
    flat, so they come back slightly inflated and must not be mistaken
    for valid interior minima by the drainage check.
    """
    from pysheds.grid import Grid

    filled_tif = work_dir / "mosaic_filled.tif"

    grid = Grid.from_raster(str(mosaic_tif))
    dem = grid.read_raster(str(mosaic_tif))
    nodata_mask = (np.asarray(dem) == NODATA) | ~np.isfinite(np.asarray(dem))

    logger.info("pysheds fill_depressions (priority-flood) ...")
    flooded = grid.fill_depressions(dem)

    logger.info("pysheds resolve_flats (eps=%g, max_iter=%d) ...",
                FLAT_EPS, FLAT_MAX_ITER)
    inflated = grid.resolve_flats(flooded, eps=FLAT_EPS,
                                  max_iter=FLAT_MAX_ITER)

    band = np.asarray(inflated, dtype="float64")
    band[nodata_mask | ~np.isfinite(band)] = NODATA
    with rasterio.open(mosaic_tif) as src:
        write_dem(filled_tif, band, src.transform, src.crs, NODATA,
                  dtype="float64")

    stuck, valid = count_undrainable(filled_tif)
    logger.info("Drainage check: %d of %d valid pixels undrainable", stuck, valid)
    if stuck > max(1, valid // 1000):
        raise RuntimeError(
            f"Filled mosaic does not drain: {stuck} of {valid} pixels have no "
            f"lower neighbour. The flat-resolution gradients were lost - "
            f"check that every step keeps the DEM in float64, and that "
            f"FLAT_MAX_ITER exceeds the widest flat in pixels."
        )
    return filled_tif


# --------------------------------------------------------------------------- #
# The breach (numba complete breaching)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _heap_push(heap_z, heap_i, n, key, idx):
    """Push ``(key, idx)`` onto the binary min-heap; return the new size."""
    heap_z[n] = key
    heap_i[n] = idx
    child = n
    while child > 0:
        parent = (child - 1) // 2
        if heap_z[parent] <= heap_z[child]:
            break
        heap_z[parent], heap_z[child] = heap_z[child], heap_z[parent]
        heap_i[parent], heap_i[child] = heap_i[child], heap_i[parent]
        child = parent
    return n + 1


@njit(cache=True)
def _heap_pop(heap_z, heap_i, n):
    """Pop the minimum off the heap; return ``(key, idx, new size)``."""
    key = heap_z[0]
    idx = heap_i[0]
    n -= 1
    heap_z[0] = heap_z[n]
    heap_i[0] = heap_i[n]
    parent = 0
    while True:
        child = 2 * parent + 1
        if child >= n:
            break
        if child + 1 < n and heap_z[child + 1] < heap_z[child]:
            child += 1
        if heap_z[parent] <= heap_z[child]:
            break
        heap_z[parent], heap_z[child] = heap_z[child], heap_z[parent]
        heap_i[parent], heap_i[child] = heap_i[child], heap_i[parent]
        parent = child
    return key, idx, n


@njit(cache=True)
def _breach_kernel(z, depth, nodata, fill_limit):
    """Breach the depressions of ``z`` in place; see :func:`breach`.

    A priority flood from the outlets (grid-edge and nodata-adjacent
    pixels) pops pixels lowest-first and records for each pixel the
    neighbour it was reached from. A popped pixel with no strictly lower
    neighbour is an undrainable depression floor, and the flood reached
    it over the depression's lowest saddle, so the highest pixel on its
    chain of backlinks is the spill elevation. If spill minus floor is
    at most ``fill_limit`` the depression is left for the fill pass;
    otherwise the chain's pixels are lowered to the floor elevation,
    carving a level trench through the barrier (Soille, Vogt and
    Colombo, 2003; the "complete breaching" of Lindsay, 2016). Metres
    lowered accumulate into the flat float32 array ``depth``. Returns
    the (breached, left-for-filling) depression counts.
    """
    nrow, ncol = z.shape
    zf = z.reshape(-1)
    n = zf.size
    visited = np.zeros(n, dtype=np.uint8)
    backlink = np.full(n, -1, dtype=np.int32)
    heap_z = np.empty(n, dtype=np.float64)   # a pixel is pushed at most
    heap_i = np.empty(n, dtype=np.int32)     # once, so capacity n is exact
    heap_n = 0

    for i in range(n):
        if zf[i] == nodata:
            visited[i] = 1

    # Seed the flood at the outlets: valid pixels on the grid edge or
    # 8-adjacent to nodata (interior nodata holes drain, see docstring).
    for r in range(nrow):
        for c in range(ncol):
            i = r * ncol + c
            if visited[i]:
                continue
            seed = r == 0 or r == nrow - 1 or c == 0 or c == ncol - 1
            if not seed:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        if zf[(r + dr) * ncol + c + dc] == nodata:
                            seed = True
            if seed:
                visited[i] = 1
                heap_n = _heap_push(heap_z, heap_i, heap_n, zf[i], i)

    n_breached = 0
    n_shallow = 0
    while heap_n > 0:
        zc, i, heap_n = _heap_pop(heap_z, heap_i, heap_n)
        r = i // ncol
        c = i - r * ncol
        has_lower = False
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= nrow or cc < 0 or cc >= ncol:
                    continue
                j = rr * ncol + cc
                if zf[j] == nodata:
                    has_lower = True
                else:
                    if zf[j] < zc:
                        has_lower = True
                    if not visited[j]:
                        visited[j] = 1
                        backlink[j] = i
                        heap_n = _heap_push(heap_z, heap_i, heap_n, zf[j], j)
        if has_lower or backlink[i] == -1:
            continue
        # An undrainable depression floor: measure its depth, then breach
        # or leave for the fill. Backlink chains hold only already-popped
        # pixels, so carving them never invalidates a pending heap key.
        spill = zc
        j = backlink[i]
        while j != -1 and zf[j] > zc:
            if zf[j] > spill:
                spill = zf[j]
            j = backlink[j]
        if spill == zc:
            continue          # flat tie: an equal neighbour already leads out
        if spill - zc <= fill_limit:
            n_shallow += 1
            continue
        n_breached += 1
        j = backlink[i]
        while j != -1 and zf[j] > zc:
            depth[j] += zf[j] - zc
            zf[j] = zc
            j = backlink[j]
    return n_breached, n_shallow


def breach(
    mosaic_tif: Path,
    work_dir: Path,
    depth_out: Path | None = None,
    fill_limit: float = FILL_LIMIT,
) -> Path:
    """Breach depressions, fill shallow ones, resolve flats; return the mosaic.

    Complete breaching (see :func:`_breach_kernel`) carves a level trench
    from each depression floor deeper than ``fill_limit`` out through its
    barrier. pysheds ``fill_depressions`` then fills the shallow
    depressions the kernel skipped -- each raised by at most
    ``fill_limit``, its measured spill minus floor -- and
    ``resolve_flats`` gives the level trenches, like any other flat,
    their drainage gradient. The metres lowered are written to
    ``depth_out`` before those passes, so that raster is > 0 exactly
    where the breach carved. The float64 and drainage-check reasoning of
    :func:`fill` applies unchanged to the written mosaic.
    """
    from pysheds.grid import Grid

    raw_tif = work_dir / "mosaic_breached_raw.tif"
    breached_tif = work_dir / "mosaic_breached.tif"

    with rasterio.open(mosaic_tif) as src:
        z = src.read(1).astype("float64")
        transform, crs = src.transform, src.crs
    if z.size >= 2 ** 31:
        raise RuntimeError(
            f"Mosaic has {z.size} pixels; the breach kernel's int32 "
            f"backlinks support at most 2**31 - 1"
        )
    nodata_mask = (z == NODATA) | ~np.isfinite(z)
    z[nodata_mask] = NODATA

    logger.info("numba complete breaching (fill limit %g m) ...", fill_limit)
    depth = np.zeros(z.size, dtype=np.float32)
    n_breached, n_shallow = _breach_kernel(z, depth, NODATA, fill_limit)
    depth = depth.reshape(z.shape)
    carved = int(np.count_nonzero(depth > 0))
    logger.info(
        "Breached %d depression(s), carving %d pixel(s); left %d shallow "
        "depression(s) (<= %g m) for the fill",
        n_breached, carved, n_shallow, fill_limit,
    )

    if depth_out is not None:
        depth[nodata_mask] = NODATA
        write_dem(
            depth_out, depth, transform, crs, NODATA, dtype="float32",
            tags=dict(
                title="Breach depth: metres lowered by stage-2 complete "
                      "breaching; > 0 exactly on carved pixels",
                breach_method=(
                    "numba complete breaching (Lindsay, 2016; Soille, Vogt "
                    f"and Colombo, 2003), fill limit {fill_limit:g} m, on "
                    "the virtual mosaic of all input tiles"
                ),
                generated_by="02_fill_dem.py",
            ),
        )
    del depth

    write_dem(raw_tif, z, transform, crs, NODATA, dtype="float64")
    del z

    grid = Grid.from_raster(str(raw_tif))
    dem = grid.read_raster(str(raw_tif))
    if fill_limit > 0:
        logger.info("pysheds fill_depressions (shallow depressions) ...")
        dem = grid.fill_depressions(dem)
    logger.info("pysheds resolve_flats (eps=%g, max_iter=%d) ...",
                FLAT_EPS, FLAT_MAX_ITER)
    inflated = grid.resolve_flats(dem, eps=FLAT_EPS, max_iter=FLAT_MAX_ITER)

    band = np.asarray(inflated, dtype="float64")
    band[nodata_mask | ~np.isfinite(band)] = NODATA
    write_dem(breached_tif, band, transform, crs, NODATA, dtype="float64")

    stuck, valid = count_undrainable(breached_tif)
    logger.info("Drainage check: %d of %d valid pixels undrainable", stuck, valid)
    if stuck > max(1, valid // 1000):
        raise RuntimeError(
            f"Breached mosaic does not drain: {stuck} of {valid} pixels have "
            f"no lower neighbour. The flat-resolution gradients were lost - "
            f"check that every step keeps the DEM in float64, and that "
            f"FLAT_MAX_ITER exceeds the widest flat in pixels."
        )
    return breached_tif


# --------------------------------------------------------------------------- #
# Output helper
# --------------------------------------------------------------------------- #
def write_dem(
    path: Path,
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    nodata: float = NODATA,
    dtype: str = "float32",
    predictor: int | None = None,
    tags: dict | None = None,
) -> Path:
    """Write a single-band GeoTIFF (tiled + compressed) in ``dtype``.

    ``predictor`` overrides the dtype-based default (3 = floating-point,
    2 = integer horizontal differencing).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if predictor is None:
        # predictor 3 = floating-point, 2 = integer horizontal differencing
        predictor = 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2
    profile = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=predictor,
        bigtiff="IF_SAFER",   # large rasters can exceed 4 GB
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        if tags:
            dst.update_tags(**tags)
    logger.info("Wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Crop the conditioned mosaic back onto each input grid
# --------------------------------------------------------------------------- #
def crop_back(
    cond_tif: Path,
    dem_path: Path,
    out_dir: Path = OUT_DIR,
    nodata: float = NODATA,
    method: str = "fill",
    fill_limit: float = FILL_LIMIT,
) -> Path:
    """Write ``out_dir/<filled_|breached_><name>.tif`` on ``dem_path``'s grid.

    Reads the input's window out of the conditioned mosaic (same lattice
    by construction, so this is a pure crop), re-applies the input's
    nodata mask, and logs how much was filled or breached as a QC
    summary. The summary compares against the post-``resolve_flats``
    surface, so in breach mode its depths can differ from the exact
    breach-depth raster by up to ``MAX_FLAT_INFLATION``.
    """
    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        orig = src.read(1).astype("float64")
        if src.nodata is not None and src.nodata != nodata:
            orig[orig == src.nodata] = nodata
        transform, crs, bounds = src.transform, src.crs, src.bounds
        src_tags = src.tags()

    with rasterio.open(cond_tif) as src:
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        cond = src.read(
            1, window=window, boundless=True, fill_value=nodata
        ).astype("float64")
    if cond.shape != orig.shape:
        raise RuntimeError(
            f"{dem_path.name}: crop shape {cond.shape} != {orig.shape}"
        )

    valid = (orig != nodata) & np.isfinite(orig)
    cond[~valid] = nodata

    diff = cond[valid] - orig[valid]
    if method == "fill":
        if diff.size and diff.min() < -1e-3:
            raise RuntimeError(
                f"{dem_path.name}: fill lowered pixels by up to "
                f"{-diff.min():.3f} m"
            )
        raised = int(np.count_nonzero(diff > 0))
        logger.info(
            "%s: %d of %d valid pixels raised (%.2f %%), max fill depth %.3f m",
            dem_path.name, raised, int(valid.sum()),
            100.0 * raised / max(1, valid.sum()),
            float(diff.max()) if diff.size else 0.0,
        )
    else:
        allowed = fill_limit + MAX_FLAT_INFLATION + 1e-3
        if diff.size and diff.max() > allowed:
            raise RuntimeError(
                f"{dem_path.name}: breach raised pixels by up to "
                f"{diff.max():.3f} m (allowed {allowed:.3f} m = fill limit "
                f"+ flat inflation)"
            )
        lowered = diff < 0
        n_low = int(np.count_nonzero(lowered))
        logger.info(
            "%s: %d of %d valid pixels breached (%.2f %%), max breach depth "
            "%.3f m, mean %.3f m; %d pixels filled (shallow depressions)",
            dem_path.name, n_low, int(valid.sum()),
            100.0 * n_low / max(1, valid.sum()),
            float(-diff.min()) if diff.size else 0.0,
            float(-diff[lowered].mean()) if n_low else 0.0,
            int(np.count_nonzero(diff > MAX_FLAT_INFLATION)),
        )

    forwarded = {k: src_tags[k] for k in ("dem_source_tiles", "dem_carve")
                 if k in src_tags}
    if "dem_source_tiles" not in forwarded:
        logger.warning(
            "%s: input carries no provenance tags (pre-convention carved "
            "tile); recording presumed source", dem_path.name,
        )
    if method == "fill":
        prefix = "filled_"
        title = "Hydrologically conditioned (depression-filled) DEM"
        how = ("pysheds fill_depressions (priority-flood, Barnes "
               "et al. 2014) + resolve_flats (Barnes et al. 2014, "
               f"eps={FLAT_EPS:g}, max_iter={FLAT_MAX_ITER})")
        dem_fill = "pysheds_fill_depressions_resolve_flats"
    else:
        prefix = "breached_"
        title = "Hydrologically conditioned (depression-breached) DEM"
        shallow = (f"pysheds fill_depressions (Barnes et al. 2014) for "
                   f"depressions at most {fill_limit:g} m deep + "
                   if fill_limit > 0 else "")
        how = ("numba complete breaching (Lindsay, 2016; Soille, Vogt and "
               f"Colombo, 2003) + {shallow}pysheds resolve_flats (Barnes "
               f"et al. 2014, eps={FLAT_EPS:g}, max_iter={FLAT_MAX_ITER})")
        dem_fill = "numba_complete_breach+pysheds_resolve_flats"
    return write_dem(
        Path(out_dir) / f"{prefix}{dem_path.stem}.tif",
        cond, transform, crs, nodata, dtype="float64",
        tags=dict(
            title=title,
            fill_method=f"{how}, run on the virtual mosaic of all input "
                        "tiles and cropped back to this tile's grid; "
                        "float64 preserves the flat-resolution gradients",
            dem_fill=dem_fill,
            source_data_credit=(SOURCE_DATA_CREDIT
                                if "dem_source_tiles" in forwarded
                                else SOURCE_DATA_CREDIT_PRESUMED),
            generated_by="02_fill_dem.py",
            **forwarded,
        ),
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def fill_all(
    input_dir: Path = INPUT_DIR,
    out_dir: Path | None = None,
    work_dir: Path | None = None,
    keep_intermediate: bool = KEEP_INTERMEDIATE,
    method: str = "fill",
    fill_limit: float = FILL_LIMIT,
) -> list[Path]:
    """Condition every ``*.tif`` in ``input_dir``; return the written paths.

    ``out_dir`` defaults per method (``OUT_DIR`` / ``OUT_DIR_BREACH``) and
    ``work_dir`` to ``out_dir/_work``, so the two methods never overwrite
    each other's outputs.
    """
    if method not in ("fill", "breach"):
        raise ValueError(f"unknown method {method!r}")
    if out_dir is None:
        out_dir = OUT_DIR if method == "fill" else OUT_DIR_BREACH
    out_dir = Path(out_dir)
    if work_dir is None:
        work_dir = out_dir / "_work"
    work_dir = Path(work_dir)

    dems = sorted(Path(input_dir).glob("*.tif"))
    if not dems:
        raise FileNotFoundError(f"No .tif DEMs found in {input_dir}")
    logger.info("Found %d DEM(s) in %s", len(dems), input_dir)

    validate_inputs(dems)
    work_dir.mkdir(parents=True, exist_ok=True)

    vrt = build_vrt(dems, work_dir / "inputs_mosaic.vrt")
    mosaic_tif = materialize_mosaic(vrt, work_dir / "mosaic.tif")
    if method == "fill":
        cond_tif = fill(mosaic_tif, work_dir)
    else:
        cond_tif = breach(mosaic_tif, work_dir,
                          depth_out=out_dir / "depth" / "breach_depth.tif",
                          fill_limit=fill_limit)

    written = [crop_back(cond_tif, d, out_dir, method=method,
                         fill_limit=fill_limit) for d in dems]

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Removed intermediates in %s", work_dir)
    return written


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline stage 2: hydrologically condition the carved "
                    "DEMs by filling depressions (default) or by breaching "
                    "(carving) them."
    )
    parser.add_argument(
        "--method", choices=("fill", "breach"), default="fill",
        help="fill = pysheds priority-flood fill (default); breach = carve "
             "a drainage path through each depression's barrier, keeping "
             "basin floors at their true elevation (writes data/02_breached)"
    )
    parser.add_argument(
        "--fill-limit", type=float, default=FILL_LIMIT, metavar="M",
        help="breach mode only: depressions at most this deep (metres) are "
             "filled instead of breached; 0 breaches every depression, "
             "0.1-0.2 is a reasonable starting range (default %(default)g)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    written = fill_all(method=args.method, fill_limit=args.fill_limit)
    print("\nFilled DEM(s):" if args.method == "fill" else "\nBreached DEM(s):")
    for p in written:
        print(f"  {p}")
    if args.method == "breach":
        print(f"\nBreach depth raster:\n"
              f"  {OUT_DIR_BREACH / 'depth' / 'breach_depth.tif'}")
