import sys
from pathlib import Path

import numpy as np

from absbayes.features import (ZONE_HALF_WIDTH_FT, build_stan_data, clean_pitches, edge_distance,
                               normalize_location)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simulate import simulate  # noqa: E402


def test_zone_edges_map_to_plus_minus_one():
    x, z = normalize_location([ZONE_HALF_WIDTH_FT, -ZONE_HALF_WIDTH_FT, 0, 0],
                              [2.5, 2.5, 1.6, 3.5], [1.6] * 4, [3.5] * 4)
    assert np.allclose(x[:2], [1, -1]) and np.allclose(z[2:], [-1, 1])


def test_pipeline_on_synthetic_data():
    raw, umpires = simulate(n=3000, seed=3, n_ump=10, n_cat=12, n_pit=40, n_bat=50)
    df = clean_pitches(raw, umpires)
    assert set(df["season_idx"]) == {1, 2} and df["y"].isin([0, 1]).all()
    data, (gx, gz) = build_stan_data(df, m=6, n_ppc=200)
    assert data["Phi_grid"].shape == (data["G"], 36) and data["G"] == gx.size
    assert data["W_u"].shape == (data["N"] + 1,) and data["nnz"] == 4 * data["N"]
    assert data["W_v"].min() >= 1 and data["W_v"].max() <= data["G"]
    for idx, J in [("ump", "J_u"), ("cat", "J_c"), ("pit", "J_p"), ("bat", "J_b")]:
        assert data[idx].min() == 1 and data[idx].max() == data[J]
    assert data["ppc_idx"].min() >= 1 and data["ppc_idx"].max() <= data["N"]


def test_called_strike_rate_rises_toward_zone_center():
    raw, umpires = simulate(n=6000, seed=4, n_ump=10, n_cat=12, n_pit=40, n_bat=50)
    df = clean_pitches(raw, umpires)
    inside = df[(df.x_s.abs() < 0.5) & (df.z_s.abs() < 0.5)]["y"].mean()
    outside = df[(df.x_s.abs() > 1.5) | (df.z_s.abs() > 1.5)]["y"].mean()
    assert inside > 0.9 and outside < 0.1


def test_edge_distance_is_zero_on_zone_edge_and_negative_inside():
    d = edge_distance([0.0, 1.0, 0.5, 2.0], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(d, [-1.0, 0.0, 0.0, 1.0])


def test_stan_data_has_edge_terms():
    raw, umpires = simulate(n=3000, seed=5, n_ump=10, n_cat=12, n_pit=40, n_bat=50)
    df = clean_pitches(raw, umpires)
    data, _ = build_stan_data(df, m=6, n_ppc=100)
    assert data["edge_d"].shape == (data["N"],) and data["edge_grid"].shape == (data["G"],)


def test_original_call_flips_only_overturned_pitches():
    import pandas as pd
    raw, umpires = simulate(n=3000, seed=6, n_ump=10, n_cat=12, n_pit=40, n_bat=50)
    raw["at_bat_number"], raw["pitch_number"] = 1, 1  # one "game" per pitch in the simulation
    flip = raw["game_pk"].to_numpy()[:200:4]           # 50 pitches overturned
    ch = pd.DataFrame({"game_pk": flip, "at_bat_number": 1, "pitch_number": 1, "overturned": 1})
    df = clean_pitches(raw, umpires, ch)
    base = clean_pitches(raw.drop(columns=["at_bat_number", "pitch_number"]), umpires)
    assert len(df) == len(base) and df["overturned"].sum() > 0
    assert (df.loc[df.overturned == 1, "y_orig"] == 1 - df.loc[df.overturned == 1, "y"]).all()
    assert (df.loc[df.overturned == 0, "y_orig"] == df.loc[df.overturned == 0, "y"]).all()


def test_no_challenges_means_original_equals_final():
    raw, umpires = simulate(n=2000, seed=7, n_ump=10, n_cat=12, n_pit=40, n_bat=50)
    df = clean_pitches(raw, umpires)
    assert (df["y"] == df["y_orig"]).all() and df["overturned"].sum() == 0


def test_stan_data_outcome_switch():
    import pandas as pd
    raw, umpires = simulate(n=2000, seed=8, n_ump=10, n_cat=12, n_pit=40, n_bat=50)
    raw["at_bat_number"], raw["pitch_number"] = 1, 1
    ch = pd.DataFrame({"game_pk": raw["game_pk"].to_numpy()[:100], "at_bat_number": 1,
                       "pitch_number": 1, "overturned": 1})
    df = clean_pitches(raw, umpires, ch)
    d_final, _ = build_stan_data(df, m=6, n_ppc=50, outcome="final")
    d_orig, _ = build_stan_data(df, m=6, n_ppc=50, outcome="original")
    assert (d_final["y"] != d_orig["y"]).sum() == df["overturned"].sum()
