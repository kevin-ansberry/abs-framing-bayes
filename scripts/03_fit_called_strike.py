"""Fit stan/called_strike.stan with CmdStanPy.

Start small (--n-max 30000, 2 chains) to confirm the model compiles and samples,
then scale up. A full-season fit on >100k pitches can take hours.
"""
import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from absbayes.features import build_stan_data  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/pitches.parquet")
    ap.add_argument("--n-max", type=int, default=100_000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--outcome", choices=["final", "original"], default="final",
                    help="final = call that stood; original = umpire's first call")
    ap.add_argument("--out", default=None, help="default: outputs/called_strike_<outcome>")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"outputs/called_strike_{a.outcome}"

    df = pd.read_parquet(a.data)
    if len(df) > a.n_max:
        df = df.sample(a.n_max, random_state=a.seed)
        # Re-index IDs after subsampling so every level appears in the data.
        for col, name in [("ump_idx", "ump_idx"), ("catcher_idx", "catcher_idx"),
                          ("pitcher_idx", "pitcher_idx"), ("batter_idx", "batter_idx"),
                          ("t_idx", "t_idx")]:
            df[name] = pd.factorize(df[col])[0] + 1
        df = df.reset_index(drop=True)
    data, (gx, gz) = build_stan_data(df, outcome=a.outcome)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out) / "grid.npz", gx=gx, gz=gz)
    df[["fielder_2", "catcher_idx"]].drop_duplicates().to_csv(
        Path(a.out) / "catcher_index.csv", index=False)
    df[["umpire_id", "ump_idx"]].drop_duplicates().to_csv(Path(a.out) / "umpire_index.csv", index=False)

    model = CmdStanModel(stan_file="stan/called_strike.stan")
    fit = model.sample(data=data, chains=a.chains, parallel_chains=a.chains,
                       iter_warmup=a.warmup, iter_sampling=a.samples,
                       adapt_delta=0.9, max_treedepth=8, inits=0.1,
                       seed=a.seed, output_dir=a.out)
    print(fit.diagnose())
    keep = ["a0", "beta_d", "beta_dd", "alpha_s", "rho_s", "alpha_d", "rho_d", "sigma_u", "sigma_c",
            "sigma_p", "sigma_b", "sigma_t", "d_sigma_c", "d_sigma_u"]
    print(fit.summary(sig_figs=3).loc[lambda s: s.index.str.split("[").str[0].isin(keep)])
    idata = az.from_cmdstanpy(fit, log_likelihood="log_lik", posterior_predictive="y_rep")
    print(az.loo(idata, var_name="log_lik"))
