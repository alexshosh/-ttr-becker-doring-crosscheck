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

- `fit_final_N1_2.py`

  Fitting script producing the paper's main results table and overlay figure: the two-stage
  reduced model fit (Stage 1 logistic tetramer depletion, Stage 2 flux-law drain into aggregate)
  against the digitized data at all three temperatures.

## Reproducing the results

Run `digitize_sun2018_fig2.py` to regenerate the CSVs from the source figure (or use the CSVs
directly), then run `fit_final_N1_2.py` to reproduce the paper's Table 1 and main figure.

Requires: `numpy`, `scipy`, `pandas`, `matplotlib`.
