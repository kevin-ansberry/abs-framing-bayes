"""Cleaning Statcast pitches and building the data dictionary the Stan models expect."""
import numpy as np
import pandas as pd

from .hsgp import bilinear_matrix, make_basis

# Pitches the batter does not swing at. "called_strike" is the outcome of interest.
TAKEN = {"called_strike", "ball", "blocked_ball"}
ZONE_HALF_WIDTH_FT = 17.0 / 24.0  # plate half-width (8.5 in)
REQUIRED = ["game_pk", "game_date", "game_type", "description", "plate_x", "plate_z",
            "sz_top", "sz_bot", "fielder_2", "pitcher", "batter"]


def normalize_location(plate_x, plate_z, sz_bot, sz_top):
    """Rescale so the nominal zone edge sits at +/-1 on both axes.

    x: horizontal, plate_x / half plate width.
    z: vertical, rescaled by each batter's own zone top and bottom.
    """
    x = np.asarray(plate_x, float) / ZONE_HALF_WIDTH_FT
    z = 2.0 * (np.asarray(plate_z, float) - np.asarray(sz_bot, float)) / (
        np.asarray(sz_top, float) - np.asarray(sz_bot, float)) - 1.0
    return x, z


def clean_pitches(raw, umpires, challenges=None, bound=2.5):
    """Filter to taken regular-season pitches and attach IDs for the model.

    raw: Statcast pitch-level frame (see REQUIRED columns).
    umpires: frame with columns game_pk, umpire_id (home-plate umpire).
    challenges: optional frame with game_pk, at_bat_number, pitch_number, overturned (0/1),
        one row per challenged pitch. Statcast stores the FINAL (post-challenge) call, so the
        umpire's ORIGINAL call is the final call, flipped on overturned pitches.

    Outcome columns: y = final call (1 = called strike); y_orig = original umpire call.
    """
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data is missing columns: {missing}")
    df = raw[(raw["game_type"] == "R") & raw["description"].isin(TAKEN)].copy()
    df = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2"])
    df = df[df["sz_top"] > df["sz_bot"]]
    df["y"] = (df["description"] == "called_strike").astype(int)
    df["overturned"] = 0
    if challenges is not None:
        keys = ["game_pk", "at_bat_number", "pitch_number"]
        if not {"at_bat_number", "pitch_number"} <= set(df.columns):
            raise ValueError("raw Statcast data needs at_bat_number and pitch_number to attach "
                             "challenge flags; re-run scripts/01_pull_statcast.py")
        ch = challenges[keys + ["overturned"]].dropna(subset=keys).copy()
        ch[keys] = ch[keys].astype(int)
        ch = ch.drop_duplicates(keys)
        df = df.drop(columns="overturned").merge(ch, on=keys, how="left")
        df["overturned"] = df["overturned"].fillna(0).astype(int)
    df["y_orig"] = np.where(df["overturned"] == 1, 1 - df["y"], df["y"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["game_date"].dt.year
    df = df.merge(umpires[["game_pk", "umpire_id"]], on="game_pk", how="inner")
    df["x_s"], df["z_s"] = normalize_location(
        df["plate_x"], df["plate_z"], df["sz_bot"], df["sz_top"])
    df = df[(df["x_s"].abs() <= bound) & (df["z_s"].abs() <= bound)]

    seasons = sorted(df["season"].unique())
    if len(seasons) != 2:
        raise ValueError(f"expected exactly 2 seasons, found {seasons}")
    df["season_idx"] = (df["season"] == seasons[1]).astype(int) + 1
    key = df["season"] * 100 + df["game_date"].dt.month
    df["t_idx"] = key.rank(method="dense").astype(int)
    for col, name in [("umpire_id", "ump_idx"), ("fielder_2", "catcher_idx"),
                      ("pitcher", "pitcher_idx"), ("batter", "batter_idx")]:
        df[name] = pd.factorize(df[col])[0] + 1
    return df.reset_index(drop=True)


def edge_distance(x_s, z_s):
    """Chebyshev distance outside the nominal zone (negative inside), in zone-edge units."""
    return np.maximum(np.abs(np.asarray(x_s, float)), np.abs(np.asarray(z_s, float))) - 1.0


def build_stan_data(df, L=4.0, m=14, n_ppc=5000, grid_lim=2.5, grid_n=51, seed=0,
                    outcome="final"):
    """Dictionary for stan/called_strike.stan plus the grid used for surface plots.

    The spatial surface is evaluated on a grid_n x grid_n grid (spacing 0.1 by default) and
    interpolated to each pitch with a sparse bilinear matrix. Pitches must lie within
    +/-grid_lim (clean_pitches enforces |x|, |z| <= 2.5).
    outcome: "final" models the call that stood (y); "original" models the umpire's first call.
    """
    if outcome not in ("final", "original"):
        raise ValueError("outcome must be 'final' or 'original'")
    rng = np.random.default_rng(seed)
    gx, gz = np.meshgrid(np.linspace(-grid_lim, grid_lim, grid_n),
                         np.linspace(-grid_lim, grid_lim, grid_n))
    Phi_grid, omega = make_basis(gx.ravel(), gz.ravel(), L, m)
    W = bilinear_matrix(df["x_s"].to_numpy(), df["z_s"].to_numpy(), grid_lim, grid_n)
    n_ppc = min(n_ppc, len(df))
    ppc_idx = np.sort(rng.choice(len(df), size=n_ppc, replace=False)) + 1  # Stan is 1-indexed
    data = {
        "N": len(df), "M": m * m, "omega": omega,
        "edge_d": edge_distance(df["x_s"], df["z_s"]),
        "y": df["y" if outcome == "final" else "y_orig"].to_numpy(), "season": df["season_idx"].to_numpy(),
        "T": int(df["t_idx"].max()), "tidx": df["t_idx"].to_numpy(),
        "J_u": int(df["ump_idx"].max()), "ump": df["ump_idx"].to_numpy(),
        "J_c": int(df["catcher_idx"].max()), "cat": df["catcher_idx"].to_numpy(),
        "J_p": int(df["pitcher_idx"].max()), "pit": df["pitcher_idx"].to_numpy(),
        "J_b": int(df["batter_idx"].max()), "bat": df["batter_idx"].to_numpy(),
        "N_ppc": n_ppc, "ppc_idx": ppc_idx,
        "G": gx.size, "Phi_grid": Phi_grid,
        "edge_grid": edge_distance(gx.ravel(), gz.ravel()),
        # Stan CSR arrays are 1-indexed
        "nnz": int(W.nnz), "W_w": W.data, "W_v": W.indices.astype(int) + 1,
        "W_u": W.indptr.astype(int) + 1,
    }
    if data["T"] < 2:
        raise ValueError("need at least two season-months for the time trend")
    return data, (gx, gz)
