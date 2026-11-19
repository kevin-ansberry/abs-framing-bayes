"""Pull pitch-level Statcast data and home-plate umpires for two seasons.

Usage:
  python scripts/01_pull_statcast.py --start-2025 2025-03-27 --end-2025 2025-09-28 \
                                     --start-2026 2026-03-25 --end-2026 2026-09-27
CHECK the regular-season start/end dates for each year before running; the defaults
above are placeholders. Requires internet access. Raw data is cached under data/raw/.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from pybaseball import cache, statcast

RAW = Path("data/raw")
KEEP = ["game_pk", "game_date", "game_type", "description", "plate_x", "plate_z",
        "sz_top", "sz_bot", "fielder_2", "pitcher", "batter", "stand", "p_throws",
        "pitch_type", "balls", "strikes", "inning", "home_team", "away_team",
        "at_bat_number", "pitch_number"]


def pull_season(start, end, tag):
    out = RAW / f"statcast_{tag}.parquet"
    if out.exists():
        print(f"{out} exists, skipping")
        return pd.read_parquet(out)
    # statcast() chunks by day internally; large ranges can be slow, so go month by month.
    edges = pd.date_range(start, end, freq="MS").union([pd.Timestamp(start)]).sort_values()
    frames = []
    for i, a in enumerate(edges):
        b = (edges[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(edges) else pd.Timestamp(end)
        b = min(b, pd.Timestamp(end))
        print(f"pulling {a.date()} to {b.date()}")
        frames.append(statcast(start_dt=str(a.date()), end_dt=str(b.date())))
    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in KEEP if c in df.columns]]
    df.to_parquet(out)
    return df


def fetch_home_plate_umpires(game_pks, sleep=0.1):
    """Home-plate umpire per game from the MLB Stats API boxscore ('officials').

    NOTE: written against the public statsapi.mlb.com schema but not run by the author;
    inspect one response if this returns empty results.
    """
    rows = []
    for gp in sorted(set(game_pks)):
        url = f"https://statsapi.mlb.com/api/v1/game/{gp}/boxscore"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            for off in r.json().get("officials", []):
                if off.get("officialType") == "Home Plate":
                    rows.append({"game_pk": gp, "umpire_id": off["official"]["id"],
                                 "umpire_name": off["official"]["fullName"]})
        except Exception as exc:  # keep going; report at the end
            print(f"game {gp}: {exc}")
        time.sleep(sleep)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-2025", default="2025-03-27")
    ap.add_argument("--end-2025", default="2025-09-28")
    ap.add_argument("--start-2026", default="2026-03-25")
    ap.add_argument("--end-2026", default="2026-09-27")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    cache.enable()
    d25 = pull_season(a.start_2025, a.end_2025, "2025")
    d26 = pull_season(a.start_2026, a.end_2026, "2026")
    games = pd.concat([d25, d26])["game_pk"].unique()
    ump_path = RAW / "umpires.csv"
    if ump_path.exists():
        print("umpires.csv exists, skipping")
    else:
        fetch_home_plate_umpires(games).to_csv(ump_path, index=False)
