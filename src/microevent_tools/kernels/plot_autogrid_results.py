#!/usr/bin/env python3
"""Plot the best LM solution from every completed autogrid SUMMARY."""
from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"

import argparse, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SUMMARY = re.compile(r"SUMMARY .*?s=([0-9.eE+-]+)\s+q=([0-9.eE+-]+)\s+rho=([0-9.eE+-]+)")
SOLUTION = re.compile(r"^\s*\d+\s+([0-9.eE+-]+)\s+")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = []
    for path in sorted(args.result_dir.glob("result_*.dat")):
        text = path.read_text(errors="replace")
        m = SUMMARY.search(text)
        if not m or "SUMMARY (" not in text:
            continue
        block = text[text.find(m.group(0)):]
        chis = []
        for line in block.splitlines():
            x = SOLUTION.match(line)
            if x:
                chis.append(float(x.group(1)))
        if not chis:
            continue
        rows.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), min(chis)])
    if not rows:
        raise SystemExit("no completed SUMMARY files")
    a = np.asarray(rows)
    best = np.nanmin(a[:, 3]); delta = a[:, 3] - best
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # Match grid_result_v3.ipynb: large points and reversed jet map.  The
    # color is the absolute chi2/dof (not delta chi2/dof); low values are red.
    chi = a[:, 3]
    sc = ax.scatter(a[:, 0], a[:, 1], c=chi, s=20, cmap="jet_r",
                    vmin=np.nanmin(chi), vmax=np.nanpercentile(chi, 99))
    # Mark the five best grid points by absolute chi2/dof rank.
    best_indices = np.argsort(chi)[:5]
    ax.scatter(a[best_indices, 0], a[best_indices, 1], s=95, marker="*",
               facecolors="none", edgecolors="black", linewidths=1.2,
               zorder=4)
    for rank, index in enumerate(best_indices, start=1):
        ax.annotate(f"{rank}", (a[index, 0], a[index, 1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=10,
                    fontweight="bold", color="black", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="s", ylabel="q", title=f"{len(a):,} completed grid points")
    ax.grid(alpha=.2, which="both")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.minorticks_on()
    cbar.ax.set_ylabel(r"$\chi^2/\mathrm{dof}$")
    cbar.ax.invert_yaxis()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"plotted {len(a)} grid points; best chi2/dof={best:.8g}; output={args.output}")

if __name__ == "__main__":
    main()
