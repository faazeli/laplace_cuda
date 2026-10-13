"""Plot solver output (Python alternative to visualize.m).

    python visualize.py                 # contour plot of the latest solution_iter_*.txt -> solution.png
    python visualize.py --gif           # animate all dumps -> solution.gif (run the solver with --dump-every N)
"""

import argparse
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def dumps():
    files = glob.glob("solution_iter_*.txt")
    return sorted(files, key=lambda f: int(re.search(r"(\d+)", f).group(1)))


def draw(ax, path):
    u = np.loadtxt(path)
    ax.clear()
    cs = ax.contourf(u, levels=20, cmap="viridis")
    ax.contour(u, levels=10, colors="white", linewidths=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"Laplace solution, iteration {re.search(r'(\d+)', path).group(1)}")
    return cs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    files = dumps()
    if not files:
        raise SystemExit("No solution_iter_*.txt files found. Run the solver first.")
    rows, cols = np.loadtxt(files[-1]).shape
    fig, ax = plt.subplots(figsize=(8, 8 * rows / cols + 0.8), dpi=120)
    if args.gif:
        FuncAnimation(fig, lambda i: draw(ax, files[i]), frames=len(files)).save("solution.gif", writer=PillowWriter(fps=6))
        print("Wrote solution.gif")
    else:
        fig.colorbar(draw(ax, files[-1]), ax=ax, label="u", fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig("solution.png")
        print(f"Wrote solution.png from {files[-1]}")
