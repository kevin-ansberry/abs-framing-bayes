"""Make figures and headline numbers from a fitted called-strike model.

  python scripts/05_report.py --fit-dir outputs/called_strike_original
  python scripts/05_report.py --fit-dir outputs/called_strike_final

Uses the newest run in the folder, so interrupted earlier runs do not break it.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import from_csv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", default="outputs/called_strike_original")
    a = ap.parse_args()
    d = Path(a.fit_dir)
    files = sorted(d.glob("called_strike-*.csv"))
    stamp = files[-1].name.rsplit("_", 1)[0]  # timestamp of the newest run
    fit = from_csv(path=[str(p) for p in files if p.name.startswith(stamp)])
    label = d.name.replace("called_strike_", "")
    g = np.load(d / "grid.npz")
    gx, gz = g["gx"], g["gz"]
    s25 = fit.stan_variable("surf_2025").mean(axis=0).reshape(gx.shape)
    s26 = fit.stan_variable("surf_2026").mean(axis=0).reshape(gx.shape)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for a_, s, t in [(ax[0], s25, f"2025 called-strike probability ({label} call)"),
                     (ax[1], s26, f"2026 called-strike probability ({label} call)")]:
        c = a_.contourf(gx, gz, s, levels=np.linspace(0, 1, 11), cmap="viridis")
        a_.set_title(t, fontsize=10)
        fig.colorbar(c, ax=a_)
    c = ax[2].contourf(gx, gz, s26 - s25, levels=21, cmap="RdBu_r")
    ax[2].set_title("Change, 2026 minus 2025", fontsize=10)
    fig.colorbar(c, ax=ax[2])
    for a_ in ax:
        a_.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "w--", lw=1)
        a_.set_xlabel("horizontal (zone half-widths)")
        a_.set_ylabel("vertical (rescaled zone)")
    fig.savefig(d / "zone_surface.png", dpi=150)

    # Did the spread of catcher and umpire effects change? Posterior of the difference.
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for a_, name, t in [(ax2[0], "d_sigma_c", "Catcher effect scale: 2026 - 2025"),
                        (ax2[1], "d_sigma_u", "Umpire effect scale: 2026 - 2025")]:
        draws = fit.stan_variable(name)
        a_.hist(draws, bins=40, color="#1F3A5F")
        a_.axvline(0, color="k", lw=1)
        lo, hi = np.percentile(draws, [5, 95])
        p_neg = float((draws < 0).mean())
        a_.set_title(f"{t} ({label} call)\n90% interval [{lo:.3f}, {hi:.3f}]   P(<0) = {p_neg:.2f}",
                     fontsize=9)
        print(f"{label:9s} {name:10s} mean {draws.mean():+.3f}  90% [{lo:+.3f}, {hi:+.3f}]  P(<0) = {p_neg:.3f}")
    fig2.savefig(d / "effect_scale_change.png", dpi=150)
    print("saved figures to", d)
