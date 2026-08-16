# TTR Becker–Döring Reduction Cross-Check: Data and Code

Digitized ¹⁹F-NMR kinetic data and fitting code supporting:

Shoshitaishvili, A. *Testing a Two-Stage Aggregation Model with Dimensionality Reduction Against
Transthyretin Kinetic Data.* (bioRxiv preprint, in preparation.)

The paper tests whether the reduced (logistic) Becker–Döring dynamics of Shoshitaishvili &
Raibekas (2010) transfers to transthyretin (TTR) amyloidogenesis kinetics, using published
¹⁹F-NMR data from Sun, Dyson & Wright (2018), *Kinetic analysis of the multistep aggregation
pathway of human transthyretin*, PNAS 115, E6201–E6208.

## Files

- `Sun2018_TTR_panelB_298K_digitized.csv`
- `Sun2018_TTR_panelC_310K_digitized.csv`
- `Sun2018_TTR_panelD_277K_digitized.csv`

  Digitized time-course data for the three experimentally resolved populations — tetramer (T),
  intermediate (I), and aggregate (A) — at 298 K, 310 K, and 277 K, extracted point-by-point from
  Figure 2 of Sun, Dyson & Wright (2018). Columns: `species` (T/I/A), `time_hr`, `value`
  (population fraction, 0–1 scale).

- `digitize_sun2018_fig2.py`

  Script used to digitize the published figure into the CSV files above (pixel calibration,
  color-based point extraction, binning, outlier filtering — see the paper's Methods section for
  the full procedure description).

- `fit_final_mixture.py`

  Fitting script producing the paper's main results (Table 1, Figure 1): the two-stage reduced
  model fit against the digitized data at all three temperatures. Stage 1 is a two-component
  mixture, $x(t) = (1-a)\,x_{\rm slow}(t) + a\,x_{\rm add}(t)$, where $x_{\rm slow}$ is the
  closed-form reduced-manifold logistic and $x_{\rm add}$ is a disclosed, not mechanistically
  derived, added relaxation term; $T(0)$ is fixed at its exact physical value of 1 (not free or
  bounded) at all three temperatures. Stage 2 fits $m(t)$ (the flux into the aggregate) against
  this $x(t)$.

  This supersedes an earlier single-logistic version of Stage 1 (no longer included here), which
  required $T(0)$ to exceed 1 at two of the three temperatures to fit the data — the two-component
  mixture was introduced specifically to remove that infeasibility.

- `sdw_model_refit.py`

  Refits Sun, Dyson & Wright's own model (the plain reversible two-step chain T⇌I⇌A, rate
  constants $k_1,k_{-1},k_2,k_{-2}$) to the same digitized data used above, so that its RMSE can be
  compared directly against `fit_final_mixture.py`'s, on identical data and an identical metric.
  SDW's own paper reports fitted rate constants and van't Hoff free energies, not RMSE, so this
  comparison does not exist anywhere else; it supports the fit-quality and AIC comparison reported
  in the paper's "What this note adds beyond SDW's own model" section.

## Reproducing the results

Run `digitize_sun2018_fig2.py` to regenerate the CSVs from the source figure (or use the CSVs
directly), then run `fit_final_mixture.py` to reproduce the paper's Table 1 and main figure, and
`sdw_model_refit.py` to reproduce the SDW-model comparison.

Requires: `numpy`, `scipy`, `matplotlib`.
