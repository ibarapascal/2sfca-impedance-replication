[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21626841.svg)](https://doi.org/10.5281/zenodo.21626841)

# Replication package: classification instability of the distance-decay choice in floating catchment accessibility

Code and derived results for a study of how the choice of distance-decay (impedance) function changes the
discrete classifications produced by two-step floating catchment area (2SFCA) accessibility measures, and of
what can be bounded about the defensible decay range from published origin-destination tables. Everything
needed to reproduce the reported diagnostics, the gravity inversion and every figure is included.

Author: Jiawei Jing (ORCID 0009-0000-8365-367X), Makuhari Development Corporation, Chiba, Japan.
The associated manuscript is under review; this repository is released so that the analysis can be
inspected and rerun independently of the review process.

## Contents

    scripts/
      pilot_flip_rate.py      Euclidean pilot: 6 specifications, one region (fast, ~90 s)
      full_network_flip.py    Main analysis: network 2SFCA x 6 specifications x 3 regions
      full_network_flip_v2.py Adds the two empirical reference specifications; retains distance pairs
      decay_inversion.py      Doubly constrained gravity inversion from person-trip OD tables
      patch_beta0006.py       Second fitting-protocol reference specification (phase-B only reuse)
      supplement_v3.py        Absolute-threshold rule, age stratification, perturbation and null floors,
                              boundary trimming, 500 m aggregation, capacity proxy, bootstrap CI,
                              destination-density stratified inversion
      verify_review.py        Verification suite: reference-implementation agreement, Furness convergence,
                              centroid-method and fitting-space sensitivity, quantile tie inspection
      make_figures.py         Figures 1-4
    results/
      *_results.json          Per-region diagnostics as reported in Tables 2 and 3
      decay_inversion.json    Band-level decay estimates and parametric fits, all purpose x mode series
      supplement_results.json Supplementary diagnostics (absolute rule, age strata, floors, bootstrap)

## Input data (all openly available, not redistributed here)

| Layer | Source |
|---|---|
| 125 m grid population, 2020 Population Census | e-Stat, Statistics Bureau of Japan |
| Medical facility points | Foursquare Open Source Places (Apache 2.0) |
| Pedestrian network | OpenStreetMap (ODbL) |
| Person-trip OD tables and planning-zone geometries | 6th Tokyo Metropolitan Area Person-Trip Survey (2018), distributed via e-Stat; zone shapefile from the survey council's data page |

## Running

Set two environment variables and run in order:

    export DATA_ROOT=/path/to/input/layers      # population grid, facilities, OSM network
    export WORK_ROOT=./work                     # outputs and cached distance pairs

    python3 full_network_flip.py all            # ~52 min, 3 regions, peak RSS ~6 GB
    python3 decay_inversion.py                  # seconds
    python3 full_network_flip_v2.py all         # ~52 min, adds empirical reference specifications
    python3 patch_beta0006.py                   # minutes, reuses cached distance pairs
    python3 supplement_v3.py                    # ~30 s, all supplementary diagnostics
    python3 make_figures.py                     # figures

`verify_review.py` is independent and may be run at any point after the first script.

Requirements: Python 3.11+, numpy, pandas, scipy, pyarrow, pyshp, matplotlib, and pyproj for the
destination-density stratification in `supplement_v3.py` (that block degrades gracefully if absent).

## Notes on reproduction

Distance computation dominates runtime. `full_network_flip.py` computes network distances once per
region by Dijkstra search from every facility node under a 5 km cut-off and shares them across all
specifications, so specification comparisons are exact rather than approximate. `full_network_flip_v2.py`
retains the cached `(facility node, cell, distance)` triples under `WORK_ROOT/full2/pairs_{region}/`,
which lets any additional specification be evaluated without repeating the graph search; both
`patch_beta0006.py` and the perturbation, catchment and capacity variants in `supplement_v3.py` use
that cache.

Random elements: the bootstrap in `supplement_v3.py` and the copula null both use a fixed seed
(20260726), so reported intervals reproduce exactly.

## Citing this package

Jing, Jiawei. 2026. "Replication Package: Classification Instability of the Distance-Decay Choice in
Floating Catchment Accessibility." Software, v1.0.1. Zenodo. https://doi.org/10.5281/zenodo.21626841

Please cite the archived release rather than the moving `main` branch, so that the exact code state is
identified.

## License

Code is released under the MIT License (see `LICENSE`). The input datasets are not redistributed here;
each is available from its own source under its own licence, as listed above.
