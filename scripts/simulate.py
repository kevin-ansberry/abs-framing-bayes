"""Generate Statcast-like synthetic data with KNOWN parameters to test the model.

Truth: catcher effect SD is 0.30 in season A and 0.20 in season B (a 33% drop), umpire
SD 0.25 -> 0.20. If the fitted d_sigma_c posterior does not cover roughly -0.10, something
is wrong in the model or the pipeline. Run this before touching real data.

  python scripts/simulate.py --n 60000 --out data/processed/pitches_sim.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from absbayes.features import ZONE_HALF_WIDTH_FT, clean_pitches  # noqa: E402


def simulate(n=60_000, seed=1, n_ump=60, n_cat=70, n_pit=300, n_bat=400):
    rng = np.random.default_rng(seed)
    season = rng.choice([2025, 2026], size=n)
    month = rng.integers(4, 10, size=n)
    sc, su = {2025: 0.30, 2026: 0.20}, {2025: 0.25, 2026: 0.20}
    ump_eff = {s: rng.normal(0, su[s], n_ump) for s in sc}
    cat_eff = {s: rng.normal(0, sc[s], n_cat) for s in sc}
    pit_eff, bat_eff = rng.normal(0, 0.1, n_pit), rng.normal(0, 0.1, n_bat)
    ump = rng.integers(0, n_ump, n); cat = rng.integers(0, n_cat, n)
    pit = rng.integers(0, n_pit, n); bat = rng.integers(0, n_bat, n)

    sz_bot, sz_top = rng.normal(1.6, 0.1, n), rng.normal(3.5, 0.1, n)
    x_s = rng.normal(0, 0.9, n)
    z_s = rng.normal(0, 0.9, n)
    plate_x = x_s * ZONE_HALF_WIDTH_FT
    plate_z = (z_s + 1) / 2 * (sz_top - sz_bot) + sz_bot

    d = np.maximum(np.abs(x_s), np.abs(z_s)) - 1.0            # > 0 outside the zone
    slope = np.where(season == 2026, 8.0, 7.0)                # sharper edges in 2026
    eta = -slope * d + 0.2 * np.sin(2 * x_s) * (d > -0.5)
    eta += np.array([ump_eff[s][u] for s, u in zip(season, ump)])
    eta += np.array([cat_eff[s][c] for s, c in zip(season, cat)])
    eta += pit_eff[pit] + bat_eff[bat]
    y = rng.random(n) < 1 / (1 + np.exp(-eta))

    raw = pd.DataFrame({
        "game_pk": np.arange(n), "game_type": "R",
        "game_date": pd.to_datetime([f"{s}-{m:02d}-15" for s, m in zip(season, month)]),
        "description": np.where(y, "called_strike", "ball"),
        "plate_x": plate_x, "plate_z": plate_z, "sz_top": sz_top, "sz_bot": sz_bot,
        "fielder_2": cat + 1000, "pitcher": pit + 5000, "batter": bat + 9000})
    # One "game" per pitch keeps the true per-pitch umpire recoverable in the simulation.
    umpires = pd.DataFrame({"game_pk": np.arange(n), "umpire_id": ump})
    return raw, umpires


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--out", default="data/processed/pitches_sim.parquet")
    a = ap.parse_args()
    raw, umpires = simulate(a.n)
    df = clean_pitches(raw, umpires)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(df.groupby("season")["y"].agg(["size", "mean"]))
