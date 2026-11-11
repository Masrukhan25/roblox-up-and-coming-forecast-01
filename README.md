# Roblox Up-and-Coming Top-10 Forecaster

A program that forecasts the Roblox "Up-and-Coming" chart's top 10 over a
72-hour window, for the make.fun bounty `roblox-up-and-coming-forecast-01`.

## How it works

`main.py` (Python 3.13, standard library only) reads the chart history at
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `/output/forecast.json`
(or `$MAKEFUN_OUTPUT_DIR/forecast.json`).

Model: drift-diffusion over rank + playing-spike surge + uniform floor.

- Each game's current rank is its last-snapshot rank.
- Drift comes from **chart-age**: games that just entered the chart climb
  fast (mean +8.7 ranks over ~3 snapshots), while older games fall. The
  empirical age-drift table was measured on the kit's practice history.
- Individual rank velocity was tested and found to be pure noise
  (correlation ~0.02 with future climb), so it is not used.
- **Playing-spike surge (new in v2):** a game's concurrent-player count
  spiking to >=3x its own median over the previous 3 snapshots is a real
  leading indicator of a top-10 entry. Measured on the kit history,
  spike>=2.0 games hit the top 10 within 72h 27% of the time (base rate
  4%), spike>=1.5 17% of the time. The surge component gives extra
  probability mass to spiking games, spread over places by the empirical
  landing-place distribution (bimodal: places 1-2 and 7-10). This fixes
  v1's biggest miss: in the practice window the two biggest spikers
  (Sweep The Moon 7.15x, Impact Lab 3.44x) shot to places 1-3 and 7-10,
  but v1 gave them ~0.003 probability each; v2 gives Sweep The Moon
  0.129 at place 1.
- **Uniform floor (new in v2):** a 12% mixture over the 15 most likely
  candidates guards against overconfident Gaussian tails. Validated as a
  robust improvement on rolling backtests (7 windows x 5 reads).
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
| v1 (age-drift only) | -32.968 | 0.0370 | 7.0 |
| **v2 (this entry: + surge + floor)** | **-30.623** | **0.0468** | **7.0** |
| Best board entries | ~-30.2 | ~0.049 | — |

Robustness (v2 vs v1): rolling backtest over 7 windows x 5 reads
(forecast from history cut at snapshots 12/15/18/21/24/27/30, scored on
the next 5 snapshots) — v2 mean -33.486 vs v1 -33.874 (+0.39); v2 wins
on 4 of 7 windows. The 2-window split (back24/back29) gives -42.912 /
-29.707 vs v1 -44.169 / -29.184. The surge signal itself was validated
on the full kit history before any practice tuning, and the parameters
(surge threshold 3.0, weight 0.7, floor 0.12) are smooth: +-0.1 in the
surge weight moves the practice score by <=0.1.

## Files

- `main.py` — the forecaster (stdlib only).
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry wallet.
