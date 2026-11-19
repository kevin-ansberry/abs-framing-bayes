"""Clean raw Statcast pitches, attach umpire IDs and challenge flags.

Writes data/processed/pitches.parquet with both outcomes:
  y       the final call that stood (1 = called strike)
  y_orig  the umpire's original call (final call, flipped on overturned challenges)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from absbayes.features import clean_pitches  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--challenges", default="data/raw/challenges_raw.csv")
    ap.add_argument("--out", default="data/processed/pitches.parquet")
    a = ap.parse_args()
    raw = pd.concat([pd.read_parquet(p) for p in sorted(Path(a.raw_dir).glob("statcast_*.parquet"))],
                    ignore_index=True)
    umpires = pd.read_csv(Path(a.raw_dir) / "umpires.csv")
    ch = None
    if Path(a.challenges).exists():
        ch = pd.read_csv(a.challenges)
        # keep ABS-type reviews only: finished, with a named challenger
        ch = ch[(ch["in_progress"] == 0) & ch["challenger_role"].notna()]
        print(f"using {len(ch)} challenge rows from {a.challenges}")
    else:
        print("no challenge file found; y_orig will equal y")
    df = clean_pitches(raw, umpires, ch)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(df.groupby("season").agg(pitches=("y", "size"), strike_rate_final=("y", "mean"),
                                   strike_rate_original=("y_orig", "mean"),
                                   overturned=("overturned", "sum")))
