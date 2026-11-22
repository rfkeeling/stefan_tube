# stefan_tube: code and data for Figs. 2-3 and Appendix C

Code that reproduces the Stefan-tube calculations in Keeling et al., *Impact of
atmospheric water vapor on the exchange of gases, especially O2*: the
respiratory tube (Fig. 2), the photosynthesis variant (Fig. 3), and the tables
and numbers in Appendix C (numerical modelling details and validation).

The model is one-dimensional Stefan-Maxwell diffusion of O2, H2O, N2 and CO2 in
a tube open to ambient air at z = 0, with N2 stagnant and equimolar O2-CO2
exchange at the end wall (z = L), and a "pivot" beyond which the side wall is
moist (H2O held at saturation). It is integrated in time to steady state.
Fluxes are called F (mol m-2 s-1), as in the paper.

## Install and run

```bash
pip install -e .          # or: pip install -r requirements.txt  (scripts also run without installing)
pytest                    # about 15 s
python scripts/run_all.py # about 2 min; regenerates data/, figures/ and results/
```

Tested with Python 3.14, numpy 2.5, scipy 1.18, matplotlib 3.11. The package
limits BLAS to one thread on import; this is much faster for the small dense
systems solved here.

## What reproduces what

| Item in the paper | Script | Output |
|---|---|---|
| Table C1, binary diffusivities | `scripts/table_C1_diffusivities.py` | `results/table_C1_diffusivities.log` |
| Fig. 2, O2 flux vs. ambient humidity | `scripts/fig2_sweep.py`, `scripts/fig2_plot.py` | `data/fig2_o2_flux_vs_humidity.csv`, `figures/fig2_o2_flux_vs_humidity.png`; numbers quoted in Sect. 5.1 in `results/fig2_plot.log` |
| Fig. 3, photosynthesis pO2 at the end wall vs. the Severinghaus formula | `scripts/fig3_sweep.py`, `scripts/fig3_plot.py` | `data/fig3_photosynthesis_wall_o2.csv`, `data/fig3_pO2_vs_humidity.csv`, `figures/fig3_pO2_vs_humidity.png`; numbers quoted in Sect. 5.2 in `results/fig3_plot.log` |
| C3.5, cross-flux-drag check | `scripts/c3_5_cross_flux_drag_check.py` | `results/c3_5_cross_flux_drag_check.log` |
| C4, mass flux and pressure difference (isobaric approximation) | `scripts/c4_isobaric_estimate.py` | `results/c4_isobaric_estimate.log` |
| C5.1 (F_N2 = 0 at steady state), Tables C2 and C3, wall O2 (C2.3), flux uniformity (C3.4) | `scripts/c5_validation_tables.py` | `results/c5_validation_tables.log` |
| Scale independence (1/L flux scaling; Fig. 3 independent of L) | `scripts/length_scaling_check.py` | `results/length_scaling_check.log` |

`data/`, `figures/` and `results/` contain the output of `run_all.py` as
shipped; rerunning overwrites them.

## Package layout (`src/stefan_tube/`)

- `constants.py`: constants, Fuller diffusivities, the Severinghaus-corrected O2-H2O diffusivity, Bolton saturation vapor pressure.
- `solver.py`: `StefanTubeSolver`, the main solver, which integrates in time to steady state (Sect. C3).
- `photosynthesis.py`: `StefanTubeSolverPhotosynthesis`, the photosynthesis wall reaction.
- `bvp.py`: `StefanTubeBVP`, the steady-state boundary-value solver used for the cross-checks (C5.2-C5.3).
- `severinghaus.py`: the closed-form Severinghaus prediction used in Fig. 3 (Eq. 5).

## Conventions worth knowing

- Ambient mole fractions are on a dry-air basis (dry air includes its CO2) and diluted by humidity: x_O2 = chi_O2,d (1 - x_H2O) and x_CO2 = chi_CO2,d (1 - x_H2O), with chi_O2,d = 0.209339 (Aoki et al., 2019) and chi_CO2,d = 400 ppm.
- The saturation mole fraction is computed from the Bolton formula (x_sat = 0.031260 at 298.15 K, 101325 Pa).
- The humidity grids of Figs. 2 and 3 run from 0 to x_sat, in 15 equal steps.
- Cross-flux drag: `solve(..., no_cfd=True)` removes the term x_O2 F_H2O / (c_t D_O2-H2O) from the O2 Stefan-Maxwell relation.
- Sweeps are warm-started from the converged profile at the previous humidity (`warm_start=`), which needs only a few diffusive timescales of relaxation per point.

## Licenses and citation

Everything in this repository (code, data files, figures and logs) is
released under the BSD 3-Clause license (`LICENSE`). Citation information is in
`CITATION.cff`.
