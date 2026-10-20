"""
Sublevel-set filtrations and Euler characteristic (EC) curves on a grid.

Works with any performance function g (see Performance_Functions.py), including a
Gaussian process emulator:
    values = grid_values(g, lower, upper, shape)   # g on a regular grid (any dimension)
    chis = ec_curve(values, thresholds)            # chi of {values <= t} for each t
Use superlevel=True for {values >= t}, i.e. the SuS failure domains {g > Y_L}.

HOW TO RUN: set the parameters below, then run this file (optionally: python Euler_Characteristic.py SEED).
"""

import sys
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from Performance_Functions import random_bumps

INK = "#2b2b2b"
NEW = "#d9534f"

try:
    SEED = int(sys.argv[1])                                  # Generation of a random landscape
except (IndexError, ValueError):
    SEED = np.random.randint(0, 100_000)

N_BUMPS = 100                                                # Parameter for landscape
ROWS, COLS = 10, 10                                          # Change number of rows/columns
N_LEVELS = 5                                                 # Change number of different heights in filtration
PER_ROW = 5                                                  # Change formatting
MAX_CELL_DIM = 1                                             # 1 = graph (V - E), 2 = also fill squares (V - E + F)


# EC computation (any dimension)

def grid_values(g, lower, upper, shape):
    """g evaluated on a regular grid over [lower, upper]^d; values[i1, ..., id] = g(x1, ..., xd)."""
    axes = [np.linspace(lower, upper, k) for k in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return g(np.column_stack([m.ravel() for m in mesh])).reshape(shape)


def quantise(values, n_levels):
    """Bin values into heights 1..n_levels with (roughly) equal numbers of grid points each."""
    return 1 + np.digitize(values, np.quantile(values, np.linspace(0, 1, n_levels + 1)[1:-1]))


def euler_characteristic(on, max_cell_dim=1):
    """
    chi of the cubical complex on the grid points where `on` is True, using cells up to
    dimension max_cell_dim: 1 gives the graph (vertices - edges between grid neighbours),
    2 also counts filled squares, ..., on.ndim gives the full complex (components - holes + ...).
    """
    chi = 0
    for k in range(min(max_cell_dim, on.ndim) + 1):
        for axes in combinations(range(on.ndim), k):           # k-cells spanning these axes
            cells = on
            for a in axes:                                     # cell is on if both ends are on
                lo = [slice(None)] * on.ndim; lo[a] = slice(None, -1)
                hi = [slice(None)] * on.ndim; hi[a] = slice(1, None)
                cells = cells[tuple(lo)] & cells[tuple(hi)]
            chi += (-1) ** k * int(cells.sum())
    return chi


def ec_curve(values, thresholds, superlevel=False, max_cell_dim=1):
    """chi of {values <= t} (or {values >= t} if superlevel) for each threshold t."""
    return np.array([euler_characteristic(values >= t if superlevel else values <= t, max_cell_dim)
                     for t in thresholds])


# Plot 1: the 3D map

def plot_surface(g, levels, lower=0, upper=1, title=""):
    img = np.flipud(levels.T)                                  # rows = x2 from top to bottom
    rows, cols = img.shape
    n_levels = img.max()
    x, y = np.meshgrid(np.linspace(lower, upper, 60), np.linspace(lower, upper, 60))
    fig = plt.figure(figsize=(11, 4.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(x, y, g(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape),
                    cmap="viridis", linewidth=0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("height")
    ax.set_title("(a) Height map $z = f(x, y)$")

    ax = fig.add_subplot(1, 2, 2)
    im = ax.imshow(img, cmap="Greys", vmin=0, vmax=n_levels + 1)
    cell_pt = 4.5 * 72 / max(rows, cols)
    if 0.45 * cell_pt >= 4.5:
        for (i, j), v in np.ndenumerate(img):
            ax.text(j, i, v, ha="center", va="center",
                    fontsize=min(12, 0.45 * cell_pt) * (0.75 if v >= 10 else 1),
                    color="white" if v > n_levels * 0.6 else INK)
    else:
        fig.colorbar(im, ax=ax, ticks=np.arange(1, n_levels + 1), fraction=0.046, pad=0.04,
                     label="height level")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(b) Sampled on a {rows}×{cols} grid, heights 1–{n_levels}")

    fig.suptitle(title)
    fig.subplots_adjust(top=0.84, bottom=0.08)
    plt.show()


# Plot 2: Filtration (2D only)

def graph_at(img, level):
    on = img <= level
    h = on[:, :-1] & on[:, 1:]
    v = on[:-1, :] & on[1:, :]
    return on, h, v


def draw_pixels(ax, img, level, cell_pt):
    on = img <= level
    num_size = min(10, 0.45 * cell_pt)
    ax.imshow(np.where(on, 0.55, 1.0), cmap="gray", vmin=0, vmax=1)
    if num_size >= 6:
        for (i, j), v in np.ndenumerate(img):
            ax.text(j, i, v, ha="center", va="center", color=INK,
                    fontsize=num_size * (0.75 if v >= 10 else 1))
    if cell_pt >= 4:
        ax.set_xticks(np.arange(-0.5, img.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, img.shape[0]), minor=True)
        ax.grid(which="minor", color="white", lw=min(1.5, 0.12 * cell_pt))
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(False)


def edge_segments(h, v):
    hi, hj = np.nonzero(h)
    vi, vj = np.nonzero(v)
    starts = np.r_[np.c_[hj, -hi], np.c_[vj, -vi]]
    ends = np.r_[np.c_[hj + 1, -hi], np.c_[vj, -vi - 1]]
    return np.stack([starts, ends], axis=1).astype(float)


def draw_graph(ax, img, level, cell_pt, max_cell_dim):
    on, h, v = graph_at(img, level)
    on0, h0, v0 = graph_at(img, level - 1)
    line_w = float(np.clip(0.1 * cell_pt, 0.3, 1.5))
    ax.add_collection(LineCollection(edge_segments(h & h0, v & v0), colors=INK, linewidths=line_w))
    ax.add_collection(LineCollection(edge_segments(h & ~h0, v & ~v0), colors=NEW, linewidths=line_w))
    if cell_pt >= 6:
        for mask, colour in [(on & on0, "#9a9a9a"), (on & ~on0, NEW)]:
            i, j = np.nonzero(mask)
            ax.scatter(j, -i, s=(0.4 * cell_pt) ** 2, color=colour, edgecolor=INK,
                       linewidth=min(0.8, 0.08 * cell_pt), zorder=2)
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(-img.shape[0] + 0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, -0.06, rf"$\ell={level},\ \chi={euler_characteristic(on, max_cell_dim)}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=11)


def plot_filtration(levels, per_row=5, max_cell_dim=1, title=""):
    img = np.flipud(levels.T)                                  # rows = x2 from top to bottom
    rows, cols = img.shape
    n_levels = img.max()
    fig_w, gap, left, right = 13, 0.3, 0.03, 0.97              # Sizes that scale with the grid
    n_cols = min(n_levels, per_row)
    n_blocks = -(-n_levels // per_row)
    panel_w = fig_w * (right - left) / (n_cols + (n_cols - 1) * gap)
    panel_h = min(panel_w * rows / cols, 3.5)
    cell_pt = 72 * min(panel_w / cols, panel_h / rows)

    pix_gap, label_gap, top_m, bottom_m = 0.3, 0.9, 0.6, 0.7   # spacing in inches
    block_h = 2 * panel_h + pix_gap
    fig_h = top_m + n_blocks * block_h + (n_blocks - 1) * label_gap + bottom_m
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(n_blocks, 1, left=left, right=right,
                             top=1 - top_m / fig_h, bottom=bottom_m / fig_h,
                             hspace=label_gap / block_h)

    for b in range(n_blocks):
        block = outer[b].subgridspec(2, n_cols, wspace=gap, hspace=pix_gap / panel_h)
        for c, l in enumerate(range(b * per_row + 1, min((b + 1) * per_row, n_levels) + 1)):
            draw_pixels(fig.add_subplot(block[0, c]), img, l, cell_pt)
            draw_graph(fig.add_subplot(block[1, c]), img, l, cell_pt, max_cell_dim)
            if l < n_levels and c + 1 < n_cols:                # "⊆" between consecutive steps
                for row in (0, 1):
                    here, nxt = block[row, c].get_position(fig), block[row, c + 1].get_position(fig)
                    fig.text((here.x1 + nxt.x0) / 2, (here.y0 + here.y1) / 2, "⊆", ha="center",
                             va="center", fontsize=16, family="DejaVu Sans")

    fig.suptitle(f"Filtration  →        {title}", fontsize=12, y=1 - 0.2 / fig_h, va="top")
    plt.show()


# Plot 3: EC curve

def plot_ec_curve(thresholds, chis, xlabel=r"$\ell$", title="", ax=None, label=None, color=INK):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(thresholds, chis, where="post", color=color, lw=1.5, label=label)
    ax.scatter(thresholds, chis, s=40, color="#9a9a9a", edgecolor=INK, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Euler Characteristic ($\chi_\ell$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax


if __name__ == "__main__":
    g = random_bumps(N_BUMPS, d=2, seed=SEED)
    levels = quantise(grid_values(g, 0, 1, (COLS, ROWS)), N_LEVELS)
    ls = np.arange(0, N_LEVELS + 1)
    chis = ec_curve(levels, ls, max_cell_dim=MAX_CELL_DIM)
    about = f"(seed {SEED}, {ROWS}×{COLS} grid)"

    print("seed:", SEED)
    print("heights:\n", np.flipud(levels.T))
    for l, chi in zip(ls[1:], chis[1:]):
        print(f"l = {l}:  V = {np.sum(levels <= l):4d}  chi = {chi}")
    plot_surface(g, levels, title=f"Random height map (seed {SEED})")
    plot_filtration(levels, PER_ROW, MAX_CELL_DIM, about)
    ax = plot_ec_curve(ls, chis, title=f"EC curve of the filtration {about}")
    ax.set_xticks(ls if N_LEVELS <= 20 else np.linspace(0, N_LEVELS, 11).round())
    plt.show()
