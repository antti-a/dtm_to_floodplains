# dtm to floodplains — Geomorphic floodplains from digital terrain model

The pipeline produces rasters of geomorphic floodplains (GFPLAIN; Nardi et al., 2019) and height
above nearest drain (HAND; Nobre et al., 2016) directly from Finland's
national 2 m elevation model (KM2). This is terrain analysis only, no hydraulic
modelling is done. The pipeline is built for Finnish data provided by the National Land Survey (NLS) and the Environment Institute (SYKE).

DTM is first carved with SYKE's culvert-correction raster so that flow crosses
road embankments instead of ponding behind them. Carved DTM is then conditioned for hydrological calculations by filling depressions and pits to ensure that every pixel drains out of the modelled area. Flow routing and accumulation are then calculated to be used by HAND and floodplain calculations. The pipeline can be modified
to work in other areas by swapping or skipping the culvert-carving stage which at the moment is specific to data available for Finland.
The floodplain delineation (`h = a·A^b`) is the pipeline's only parametrized step. Suitable values of `a` and `b` depend on the intended use.

The six stage scripts are numbered in pipeline order (`01\_` … `06\_`) and
share one `data/` tree: each stage's output is already the next stage's
default input, and `00\_run\_pipeline.py` runs them in order. Each stage is
also a standalone command-line script, so any stage can be re-run alone
with different parameters. The three unnumbered files are companion modules
(`pipeline\_io.py`, `accumulation.py`, `mdinf.py`) imported by the stages.

## Quick start

```bash
git clone https://github.com/antti-a/dtm\_to\_floodplains.git
cd dtm\_to\_floodplains
conda env create -f environment.yml
conda activate water
# drop your DTM tiles (GeoTIFF) into data/00\_source\_dems/
python 00\_run\_pipeline.py
```

The result is `data/06\_floodplains/floodplains.tif` (1 = floodplain,
0 = upland) plus every intermediate product.
Stage 7 additionally writes `data/07\_classified/floodplains\_classified.tif`
(0 = dry land, 1 = potential floodplain/basin, 2 = blob, 3 = lake and
its shores) and `data/07\_classified/floodplains\_clean.tif` (class 1
alone as 1/0/−1: the floodplain minus lakes, shores and blobs).

## Running the pipeline

Full run: All six stages:

```bash
python 00\_run\_pipeline.py
```

Resume after a failure, or run a subset (earlier stages' outputs are
reused):

```bash
python 00\_run\_pipeline.py --from route
python 00\_run\_pipeline.py --only fill route
python 00\_run\_pipeline.py --skip hand
```

Adjust the floodplain parameters: The flood level `h = a·A^b` is the
only parametrized step. `a` sets the overall magnitude of `h`; `b` sets how fast `h` grows as drainage area grows. Suitable values depend on the intended use.



For example:

```bash
python 06\_floodplains.py --a 0.2 --b 0.3
```

Denser or sparser stream network for HAND and the floodplains: Lower
or raise the stream-initiation threshold (km² of upstream area):

```bash
python 05\_hand.py --upa-min 1.0
python 06\_floodplains.py --upa-min 1.0
```

Compare D8, MFD, Dinf and MDinf flow routing algorithms (not needed for
floodplains, but interesting and not widely available elsewhere): Route with all four
algorithms and get each one's stream network (GeoJSON), flow-direction
raster and a comparison table (stream pixels, Jaccard overlaps, drainage
density):

```bash
python 03\_flow\_router.py --fdir all
```

### Flag reference

|script|flag|meaning (default)|
|-|-|-|
|`00\_run\_pipeline.py`|`--from`, `--only`, `--skip`|which stages to run|
|`00\_run\_pipeline.py`|`--upa-min KM2`|minimum contributing area defining a stream in km² (2.0)|
|`03\_flow\_router.py`|`--upa-min KM2`|minimum contributing area defining a stream in km² (2.0)|
|`03\_flow\_router.py`|`--fdir d8 mfd dinf mdinf` / `all`|which routing algorithms to run (d8)|
|`04\_flow\_accumulation.py`|`--units m2/pixel`|accumulation in square metres or pixel counts (m2)|
|`05\_hand.py`, `06\_floodplains.py`|`--upa-min KM2`|minimum contributing area defining a stream in km² (2.0)|
|`06\_floodplains.py`|`--a`, `--b`|GFPLAIN power law `h = a·A^b` (0.1, 0.3)|
|`07\_classify.py`|`--radius PX`|opening disc radius in pixels; severs floodplain connections narrower than \~2·radius (3)|
|`07\_classify.py`|`--dmax M`|distance to the nearest stream pixel splitting class 1 from class 2, in metres (100)|
|`07\_classify.py`|`--lake-min-ha HA`|minimum area of a constant-elevation (hydro-flattened) water surface to classify as lake, in hectares; 0 disables (1)|

`python <script> --help` lists everything, including flags that repoint the
input and output locations. Stages 1–2 are configured by the constants at
the top of each script; stages 5–6 also have a `USER SETTINGS` block whose
values are simply the defaults a no-argument run uses.

## The stages

|#|script|reads|writes|
|-|-|-|-|
|1|`01\_carve\_dem.py`|`data/00\_source\_dems/`|`data/01\_carved/` (+ `data/culvert\_cache/`)|
|2|`02\_fill\_dem.py`|`data/01\_carved/`|`data/02\_filled/`|
|3|`03\_flow\_router.py`|`data/02\_filled/`|`data/03\_flows/`|
|4|`04\_flow\_accumulation.py`|`data/03\_flows/flow\_direction\_\*.tif`|`data/04\_accumulation/`|
|5|`05\_hand.py`|`data/02\_filled/` + `data/03\_flows/flow\_direction\_d8.tif` + `data/04\_accumulation/flow\_accumulation\_d8.tif`|`data/05\_hand/`|
|6|`06\_floodplains.py`|same as stage 5|`data/06\_floodplains/`|
|7|`07\_classify.py`|`data/06\_floodplains/floodplains.tif` + `data/04\_accumulation/flow\_accumulation\_d8.tif` + `data/03\_flows/flow\_direction\_d8.tif` + `data/01\_carved/`|`data/07\_classified/`|

1. **Carve** — lowers the DTM at culverts and road crossings with the SYKE
"Tierumpujen uomakorjaus" WCS layer so flow crosses embankments.
Downloads are windowed and cached; a re-run skips finished tiles.
2. **Fill** — pysheds `fill\_depressions` (priority-flood) and
`resolve\_flats` (both Barnes et al., 2014) on the mosaic of all tiles,
cropped back to each tile's grid. Outputs are float64 on purpose:
float32 collapses the flat-resolution gradients and silently
un-conditions the DEM (stage 3 verifies drainage and stops if so).
3. **Route** — mosaics the filled tiles and routes flow: D8 by default
(O'Callaghan and Mark, 1984) as that is the format every later stage consumes.
MFD, Dinf and MDinf available via `--fdir` for comparison, each
with its own network, comparison-table row and direction raster.
4. **Accumulate** — weighted flow accumulation (upstream contributing
area) for every flow-direction raster found.
5. **HAND** — height above nearest drain (Nobre et al., 2016): Each pixel's
elevation above the stream pixel it drains to along the D8 flow path,
with streams defined by the `--upa-min` threshold.
6. **Floodplains** — GFPLAIN (Nardi et al., 2019): Every stream pixel
carries a flood level `h = a·A^b` (h in m, A = upstream area in km²):
A ground pixel belongs to the floodplain of a stream pixel it drains to if it
rises no more than `h` metres above it.
7. **Classify** — cleans the floodplain raster into three classes: a
morphological opening (disc of `--radius` pixels) severs connections
narrower than \~2·radius; floodplain keeping stream contact, or within
`--dmax` metres of a stream pixel, is class 1 (potential
floodplain/basin) and farther floodplain is class 2 (blob); connected
regions of constant carved elevation of at least `--lake-min-ha`
hectares — KM2 hydro-flattens water surfaces, so nothing else is that
flat — are class 3 (lake), together with the shore floodplain whose
controlling stream pixel (first stream pixel downstream along D8) lies
inside a lake, so classes 1 and 2 describe river floodplain only. The
stream threshold is read from the stage-6 raster's tags. A second
raster, `floodplains\_clean.tif`, carries class 1 alone in the stage-6
binary encoding (1/0/−1).

## Outputs and metadata

All rasters are GeoTIFFs on the grid and coordinate reference system of the
input DTM tiles; nothing is reprojected. Stage 1 requires EPSG:3067
(ETRS89 / TM35FIN) because the SYKE culvert layer ships in it, so a full
run on Finnish KM2 data produces EPSG:3067 rasters.
If the culvert-carving stage is skipped for other areas, the later stages
accept any single projected, metre-based CRS shared by all tiles.

Every stage stamps its outputs with self-documenting GeoTIFF dataset tags, readable with `gdalinfo <file>` or `rasterio.open(...).tags()`.

## Credits

The beginning of the pipeline follows Rolim da Paz (2025): The condition-route-accumulate
workflow of stages 1–4, and then the pyflwdir library (Eilander et al., 2021;
https://github.com/Deltares/pyflwdir) in stages 5–6 creates HAND after Nobre et al. (2016), and GFPLAIN after
Nardi et al. (2019) with the coefficient `a` made an explicit parameter.

Stage 7's morphological
operators (binary opening, geodesic dilation, via SciPy) are standard
mathematical morphology (Soille, 2004).

Other essential tools for this project are: pysheds (D8/MFD/Dinf routing; stage 2 depression
filling and flat resolution after Barnes, Lehman and Mulla, 2014),
rasterio/GDAL, NumPy and Numba. The MDinf direction mathematics in
`mdinf.py` follow Seibert and McGlynn (2007), ported via WhiteboxTools'
MIT-licensed implementation (John Lindsay).

Source data: KM2 2 m DEM © National Land Survey of Finland (CC BY 4.0);
culvert corrections: SYKE "Tierumpujen uomakorjaus" WCS (CC BY 4.0).

## References

Barnes, R., Lehman, C. and Mulla, D. (2014) 'Priority-flood: an optimal
depression-filling and watershed-labeling algorithm for digital elevation
models', *Computers \& Geosciences*, 62, pp. 117–127. Available at:
https://doi.org/10.1016/j.cageo.2013.04.024

Barnes, R., Lehman, C. and Mulla, D. (2014) 'An efficient assignment of
drainage direction over flat surfaces in raster digital elevation models',
*Computers \& Geosciences*, 62, pp. 128–135. Available at:
https://doi.org/10.1016/j.cageo.2013.01.009

Eilander, D., van Verseveld, W., Yamazaki, D., Weerts, A., Winsemius, H.C.
and Ward, P.J. (2021) 'A hydrography upscaling method for scale-invariant
parametrization of distributed hydrological models', *Hydrology and Earth
System Sciences*, 25(9), pp. 5287–5313. Available at:
https://doi.org/10.5194/hess-25-5287-2021

Nardi, F., Annis, A., Di Baldassarre, G., Vivoni, E.R. and Grimaldi, S.
(2019) 'GFPLAIN250m, a global high-resolution dataset of Earth's
floodplains', *Scientific Data*, 6, 180309. Available at:
https://doi.org/10.1038/sdata.2018.309

Nobre, A.D., Cuartas, L.A., Momo, M.R., Severo, D.L., Pinheiro, A. and
Nobre, C.A. (2016) 'HAND contour: a new proxy predictor of inundation
extent', *Hydrological Processes*, 30(2), pp. 320–333. Available at:
https://doi.org/10.1002/hyp.10581

O'Callaghan, J.F. and Mark, D.M. (1984) 'The extraction of drainage
networks from digital elevation data', *Computer Vision, Graphics, and
Image Processing*, 28(3), pp. 323–344. Available at:
https://doi.org/10.1016/S0734-189X(84)80011-0

Rolim da Paz, A. (2025) *Digital elevation models for environmental studies*.
Cham: Springer. Available at: https://doi.org/10.1007/978-3-032-04523-2

Seibert, J. and McGlynn, B.L. (2007) 'A new triangular multiple flow
direction algorithm for computing upslope areas from gridded digital
elevation models', *Water Resources Research*, 43(4), W04501. Available at:
https://doi.org/10.1029/2006WR005128

Soille, P. (2004) *Morphological image analysis: principles and
applications*. 2nd edn. Berlin: Springer. Available at:
https://doi.org/10.1007/978-3-662-05088-0

