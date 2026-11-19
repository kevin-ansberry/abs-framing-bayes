"""Fit stan/challenge.stan on ABS challenge outcomes.

Expects data/raw/abs_challenges.csv with one row per challenge and columns:
  game_pk, umpire_id, challenger_id, challenger_role (batter|catcher|pitcher),
  overturned (0/1), edge_dist (distance of the pitch from the zone edge, in zone
  half-width units, > 0).
Where this data comes from is up to you (Baseball Savant's ABS challenge pages or MLB's
data feeds); confirm what fields are public before relying on this step. Nothing here has
been run against real challenge data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

ROLES = {"batter": 1, "catcher": 2, "pitcher": 3}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/abs_challenges.csv")
    ap.add_argument("--out", default="outputs/challenge")
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df = df[df["edge_dist"] > 0].copy()
    df["role"] = df["challenger_role"].map(ROLES)
    df["ch_key"] = df["challenger_id"].astype(str) + "_" + df["challenger_role"]
    df["ch"] = pd.factorize(df["ch_key"])[0] + 1
    df["ump"] = pd.factorize(df["umpire_id"])[0] + 1
    ch_role = df.drop_duplicates("ch").sort_values("ch")["role"].to_numpy()
    data = {"N": len(df), "y": df["overturned"].to_numpy(), "role": df["role"].to_numpy(),
            "log_edge_dist": np.log(df["edge_dist"].to_numpy()),
            "J_ch": int(df["ch"].max()), "ch_role": ch_role, "ch": df["ch"].to_numpy(),
            "J_u": int(df["ump"].max()), "ump": df["ump"].to_numpy()}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fit = CmdStanModel(stan_file="stan/challenge.stan").sample(
        data=data, chains=a.chains, parallel_chains=a.chains, seed=2026, output_dir=a.out)
    print(fit.diagnose())
    print(fit.summary().loc[["b0[1]", "b0[2]", "b0[3]", "sigma_ch[1]", "sigma_ch[2]", "sigma_ch[3]", "sigma_u"]])
