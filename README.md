# Roblox Up-and-Coming Top-10 Forecaster

A program that forecasts the Roblox "Up-and-Coming" chart's top 10 over a
72-hour window, for the make.fun bounty `roblox-up-and-coming-forecast-01`.

## How it works

`main.py` (Python 3.13, standard library only) reads the chart history at
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `/output/forecast.json`
(or `$MAKEFUN_OUTPUT_DIR/forecast.json`).

Model: drift-diffusion over rank.

- Each game's current rank is its last-snapshot rank.
- Drift comes from **chart-age**: games that just entered the chart climb
  fast (mean +8.7 ranks over ~3 snapshots), while older games fall. The
  empirical age-drift table was measured on the kit's practice history.
- Individual rank velocity was tested and found to be pure noise
  (correlation ~0.02 with future climb), so it is not used.
- Playing-count surge was tested and added nothing beyond age, so it is
  not used.
- The predicted rank at time `t` is
  `r0 + drift * tau * (1 - exp(-t/tau))` (mean-reverting drift), with a
  Gaussian uncertainty growing as `sqrt(s0^2 + diff^2 * t)`. Probabilities
  per place come from the Gaussian density, time-averaged over the 72h
  window with weights favouring earlier (more predictable) hours.
- The unnamed share (0.25) covers games not named at a place, including
  brand-new games that enter during the window.

The program is fully deterministic: no wall clock, no randomness, no
network. Two runs on the same history write byte-identical output.
It runs in ~0.3s on the practice history.

## Practice evaluation

Scored on the real practice window (10 archive reads, 2026-11-08 to
2026-11-10), using a scorer that replicates the bounty log-score rule
exactly (validated: it reproduces the house baseline's published practice
score of -43.512039 to 6 decimal places).

| Forecast | Practice log score | Typical p/place | Top-10 hits |
|---|---:|---:|---:|
| House baseline (chart will not move) | -43.512 | 0.0129 | 6.4 |
| House model (AI forecast) | -37.2 | — | — |
| **This entry** | **-32.968** | **0.0370** | **7.0** |
| Best board entries | ~-30.2 | ~0.049 | — |

## Files

- `main.py` — the forecaster (stdlib only).
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry wallet.
