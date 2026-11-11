#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun program/v1).

Reads the chart history at $MAKEFUN_INPUT_DIR (default /input) and writes
/output/forecast.json (or $MAKEFUN_OUTPUT_DIR/forecast.json).

Model: drift-diffusion over rank.
- Each game's current rank is its last-snapshot rank.
- Drift comes from chart-age: games that just entered the chart climb fast
  (mean +8.7 ranks over ~3 snapshots), old games fall. The empirical
  age-drift table was measured on the kit's practice history.
- Rank velocity from individual trajectories was tested and found to be
  pure noise (corr ~0.02), so it is NOT used.
- Playing-count surge was tested and found to add nothing beyond age, so
  it is NOT used.
- The predicted rank at time t is r0 + drift * tau * (1 - exp(-t/tau))
  (mean-reverting drift), with Gaussian uncertainty growing as
  sqrt(s0^2 + diff^2 * t). Probabilities per place come from the Gaussian
  density, time-averaged over the 72h window with weights that favour
  earlier (more predictable) hours.
- The unnamed share (0.25) covers games not named at a place, including
  brand-new games that enter during the window.

Stdlib only. Fully deterministic: no wall clock, no randomness, no network.
"""
import json
import math
import os
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------- parameters
S0 = 1.5            # base rank std at t=0
DIFF = 0.7          # diffusion rate (std grows as DIFF * sqrt(t))
UNNAMED = 0.25      # unnamed share per place
MIN_P = 0.008       # minimum probability to include a row
MAX_RANK = 100      # candidate games: last-chart rank <= MAX_RANK
AGE_WEIGHT = 2.0    # multiplier on the empirical age drift
AGE_TAU = 12.0      # mean-reversion time constant for drift (hours)
TIME_DECAY = 48.0   # time-weight decay (hours); 0 = uniform
SAMPLE_TIMES = [0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]

# Empirical mean rank change (ranks/hour, negative = climbing) by chart-age
# in snapshots since first appearance on the chart. Measured on the kit's
# practice history (34 snapshots). Age 20+ is the censored/old bucket.
AGE_DRIFT = [
    (0, -0.46), (1, -0.29), (2, -0.19), (3, -0.11), (4, -0.13),
    (5, -0.086), (6, -0.063), (7, -0.043), (8, 0.028), (9, 0.067),
    (10, 0.087), (11, 0.13), (12, 0.20), (13, 0.13), (14, 0.22),
    (15, 0.15), (16, 0.15), (17, 0.17), (18, 0.18), (19, 0.17),
    (20, 0.24),
]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def gauss_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def age_drift_rate(age_snaps):
    if age_snaps <= 0:
        return AGE_DRIFT[0][1]
    if age_snaps >= 20:
        return AGE_DRIFT[-1][1]
    lo = AGE_DRIFT[int(age_snaps)]
    hi = AGE_DRIFT[int(age_snaps) + 1]
    f = age_snaps - int(age_snaps)
    return lo[1] * (1.0 - f) + hi[1] * f


def load_history(input_dir):
    path = os.path.join(input_dir, "history.ndjson")
    snaps = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                snaps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return snaps


def load_task(input_dir):
    path = os.path.join(input_dir, "task.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def build_forecast(snaps, task):
    places = int(task.get("places", 10))
    if not snaps:
        return {"probabilities": [], "top10": []}
    times = []
    for s in snaps:
        try:
            times.append(parse_time(s["at"]))
        except (KeyError, ValueError):
            times.append(None)
    # drop snapshots without a parseable time, keeping order
    paired = [(t, s) for t, s in zip(times, snaps) if t is not None]
    if not paired:
        return {"probabilities": [], "top10": []}
    paired.sort(key=lambda p: p[0])
    t_end = paired[-1][0]
    snaps = [s for _, s in paired]

    rank_series = defaultdict(list)
    first_appear = {}
    for i, s in enumerate(snaps):
        for r in s.get("up_and_coming") or []:
            try:
                uid = int(r["universe_id"])
                rk = int(r["rank"])
            except (KeyError, TypeError, ValueError):
                continue
            rank_series[uid].append(rk)
            if uid not in first_appear:
                first_appear[uid] = i
    last = snaps[-1]
    last_rank = {}
    for r in last.get("up_and_coming") or []:
        try:
            last_rank[int(r["universe_id"])] = int(r["rank"])
        except (KeyError, TypeError, ValueError):
            continue
    last_chart = sorted(last_rank, key=lambda u: last_rank[u])
    n_snaps = len(snaps)

    # time weights (favour earlier, more predictable hours)
    if TIME_DECAY and TIME_DECAY > 0:
        tw = [math.exp(-t / TIME_DECAY) for t in SAMPLE_TIMES]
    else:
        tw = [1.0] * len(SAMPLE_TIMES)
    tw_sum = sum(tw)

    candidates = set()
    for uid, rk in last_rank.items():
        if rk <= MAX_RANK:
            candidates.add(uid)
    # games recently in the top 15 may return to the top 10
    for s in snaps[-4:]:
        for r in s.get("up_and_coming") or []:
            try:
                if int(r["rank"]) <= 15:
                    candidates.add(int(r["universe_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    candidates = sorted(candidates)
    if not candidates:
        return {"probabilities": [], "top10": []}

    score = {uid: [0.0] * (places + 1) for uid in candidates}
    exp_rank = {}
    for uid in candidates:
        r0 = last_rank.get(uid)
        if r0 is None:
            series = rank_series.get(uid)
            r0 = series[-1] if series else 50
        fa = first_appear.get(uid, 0)
        age_snaps = (n_snaps - 1 - fa) if fa > 0 else 20
        drift = age_drift_rate(age_snaps) * AGE_WEIGHT
        tot = 0.0
        for ti, t in enumerate(SAMPLE_TIMES):
            mu = r0 + drift * AGE_TAU * (1.0 - math.exp(-t / AGE_TAU))
            mu = max(1.0, min(120.0, mu))
            tot += mu
            sig = math.sqrt(S0 * S0 + (DIFF * DIFF) * t)
            w = tw[ti] / tw_sum
            for k in range(1, places + 1):
                score[uid][k] += w * gauss_pdf(k, mu, sig)
        exp_rank[uid] = tot / len(SAMPLE_TIMES)

    probs = []
    for k in range(1, places + 1):
        tot = sum(score[uid][k] for uid in candidates)
        if tot <= 0:
            continue
        target = 1.0 - UNNAMED
        rows = []
        for uid in candidates:
            p = score[uid][k] / tot * target
            if p >= MIN_P:
                rows.append({"universe_id": uid, "place": k, "p": round(p, 6)})
        # at most 100 rows per place; keep the largest
        rows.sort(key=lambda r: -r["p"])
        probs.extend(rows[:100])

    ranked = sorted(candidates, key=lambda u: exp_rank[u])
    top10 = ranked[:places]
    if len(top10) < places:
        for uid in last_chart:
            if uid not in top10:
                top10.append(uid)
            if len(top10) >= places:
                break
    return {"probabilities": probs, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    snaps = load_history(input_dir)
    task = load_task(input_dir)
    forecast = build_forecast(snaps, task)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.json")
    # deterministic serialisation: sorted keys, fixed separators
    text = json.dumps(forecast, sort_keys=True, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
