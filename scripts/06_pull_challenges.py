"""Pull pitch-level challenge events from MLB's game feed and build the challenge-model input.

Needs internet. Reads data/raw/statcast_2026.parquet (for game_pk) and data/raw/umpires.csv.
Writes:
  data/raw/challenges_raw.csv  every pitch-level review found (cached; rerun with --refresh)
  data/raw/abs_challenges.csv  input for scripts/04_fit_challenge.py

Then prints diagnostics. READ THEM before trusting anything: review types, match rate to
Statcast, coordinate agreement, whether catchers are who we think, and whether Statcast
stores the original or the corrected call on overturned pitches.
Statcast must include at_bat_number and pitch_number (re-pull if the parquet lacks them).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from absbayes.challenges import COLUMNS, add_edge_dist, parse_feed  # noqa: E402
from absbayes.features import normalize_location  # noqa: E402

URL = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"


def pull(game_pks, sleep=0.1):
    frames, failed = [], []
    for i, gp in enumerate(game_pks, 1):
        try:
            r = requests.get(URL.format(int(gp)), timeout=60)
            r.raise_for_status()
            frames.append(parse_feed(r.json(), int(gp)))
        except Exception as exc:
            failed.append(int(gp))
            print(f"game {gp}: {exc}")
        if i % 25 == 0:
            print(f"{i}/{len(game_pks)} games")
        time.sleep(sleep)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return out, failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--statcast", default="data/raw/statcast_2026.parquet")
    ap.add_argument("--umpires", default="data/raw/umpires.csv")
    ap.add_argument("--raw-out", default="data/raw/challenges_raw.csv")
    ap.add_argument("--out", default="data/raw/abs_challenges.csv")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    sc = pd.read_parquet(a.statcast)
    need = {"at_bat_number", "pitch_number"}
    if not need <= set(sc.columns):
        sys.exit(f"Statcast file lacks {need - set(sc.columns)}; re-run scripts/01_pull_statcast.py "
                 "after deleting data/raw/statcast_*.parquet")

    raw_path = Path(a.raw_out)
    if raw_path.exists() and not a.refresh:
        raw = pd.read_csv(raw_path)
        print(f"using cached {raw_path} ({len(raw)} rows); --refresh to re-pull")
    else:
        raw, failed = pull(sc["game_pk"].unique())
        raw.to_csv(raw_path, index=False)
        print(f"pulled {len(raw)} pitch-level reviews; {len(failed)} games failed")

    print("\n== review types (all pitch-level reviews) ==")
    print(raw["review_type"].value_counts(dropna=False).to_string())
    print("\n== challenger role / overturn rate ==")
    print(raw.groupby("challenger_role", dropna=False)["overturned"].agg(["size", "mean"]).to_string())

    m = raw.merge(sc[["game_pk", "at_bat_number", "pitch_number", "description", "plate_x",
                      "plate_z", "sz_top", "sz_bot", "fielder_2"]],
                  on=["game_pk", "at_bat_number", "pitch_number"], how="left",
                  suffixes=("", "_sc"))
    matched = m["description"].notna()
    print(f"\n== merge to Statcast: {matched.mean():.1%} of {len(m)} reviews matched ==")
    if matched.any():
        dx = (m.loc[matched, "pX"].astype(float).abs() - m.loc[matched, "plate_x"].astype(float).abs()).abs()
        dz = (m.loc[matched, "pZ"].astype(float) - m.loc[matched, "plate_z"].astype(float)).abs()
        print(f"median |abs(pX) - abs(plate_x)| = {dx.median():.4f} ft, median |pZ - plate_z| = {dz.median():.4f} ft "
              "(near 0 means the join is correct)")
        cat = m[matched & (m["challenger_role"] == "catcher")]
        if len(cat):
            print(f"catcher-role challengers equal to Statcast fielder_2: "
                  f"{(cat['challenger_id'] == cat['fielder_2']).mean():.1%}")
        ov = m[matched & (m["overturned"] == 1)].copy()
        if len(ov):
            x, z = normalize_location(ov["plate_x"].astype(float), ov["plate_z"].astype(float),
                                      ov["sz_bot"].astype(float), ov["sz_top"].astype(float))
            ov["inside_zone"] = np.maximum(np.abs(x), np.abs(z)) < 1
            print("\n== overturned pitches: Statcast description vs. pitch inside the zone? ==")
            print(pd.crosstab(ov["inside_zone"], ov["description"]).to_string())
            print("(If Statcast keeps the CORRECTED call, inside_zone=True rows are called_strike and "
                  "inside_zone=False rows are ball. If it keeps the ORIGINAL call, the opposite.)")

    umps = pd.read_csv(a.umpires)[["game_pk", "umpire_id"]].drop_duplicates("game_pk")
    out = add_edge_dist(raw).merge(umps, on="game_pk", how="inner")
    out = out[(out["in_progress"] == 0) & out["challenger_role"].notna() & (out["edge_dist"] > 0)]
    out = out[["game_pk", "umpire_id", "challenger_id", "challenger_role", "overturned", "edge_dist"]]
    out.to_csv(a.out, index=False)
    print(f"\nwrote {len(out)} rows to {a.out}")
