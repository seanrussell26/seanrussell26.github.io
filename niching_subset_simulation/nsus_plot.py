"""
Plots for runs of nsus.py (2D problems), in the style of Subset_Simulation.py.

    plot_levels_2d(ss, threshold)          samples coloured by branch, the
                                           failure boundary and the partition
                                           boundaries found by the classifier
    plot_hill_valley_graph(ss, level)      the hill valley graph built on one
                                           level (cf. Figs. 5-6 of the paper)
    plot_sus_vs_nsus(sus, nss, threshold)  the two side by side

None of this is in Kinnear's package (his plot.py only scatters the levels);
it only reads the finished tree, so it never changes a run.
"""

import numpy as np
import matplotlib.pyplot as plt

# Fixed categorical order: branch k always gets colour k.
BRANCH_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRUNK_COLOUR = "#8a8a85"          # levels shared by every branch


def branch_colour_index(ss):
    """Colour group of each level: a level inherits its parent's group unless
    the parent was split, in which case each child starts a new group.
    Returns {level: group}, group -1 being the unsplit trunk."""
    groups, next_group = {}, 0
    for level in ss.initial_level.preorder():
        parent = level.parent
        if parent is None:
            groups[level] = -1
        elif len(parent.children) > 1:
            groups[level] = next_group
            next_group += 1
        else:
            groups[level] = groups[parent]
    return groups


def _colour(group):
    return TRUNK_COLOUR if group < 0 else BRANCH_COLOURS[group % len(BRANCH_COLOURS)]


def _grid(ss, lim, res):
    """Evaluate the raw performance function on a res x res grid. The raw
    (uncached) function is used, so plotting never changes eval counts."""
    g = ss.performance_function.non_cache_performance_function
    t = np.linspace(lim[0], lim[1], res)
    X1, X2 = np.meshgrid(t, t)
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    G = np.array([g(x) for x in pts]).reshape(X1.shape)
    return X1, X2, pts, G


def _auto_lim(ss):
    allx = np.array([s.array for s in ss.all_samples])
    return (np.floor(allx.min() - 0.5), np.ceil(allx.max() + 0.5))


def draw_partition_boundaries(ss, ax, X1, X2, pts):
    """Black lines where a classifier splits a level. Each boundary is only
    drawn inside the region that level itself lives in (its parent sets)."""
    for level in ss.all_levels:
        if len(level._buds) < 2 or 'classifier' not in level.extra_info:
            continue
        labels = level.extra_info['classifier'].predict(pts).astype(float)
        inside = np.array([level.indicator.partition_indicator(x) for x in pts])
        labels[inside == 0] = np.nan
        k = len(level.extra_info['partition'])
        ax.contour(X1, X2, labels.reshape(X1.shape), levels=np.arange(k - 1) + 0.5,
                   colors="k", linewidths=1.2, zorder=4)


def plot_levels_2d(ss, threshold, lim=None, res=150, ax=None, title=None,
                   every=1, boundaries=True, legend=True, filename=None):
    """Scatter every level, coloured by branch.

    ss        : a finished SubsetSimulation / HillValleyGraphNSuS
    threshold : failure threshold (drawn as the solid contour g = threshold)
    every     : plot only every `every`-th sample (the paper uses 10)
    res       : grid resolution for the contours (lower it for slow g)
    """
    if ss.dimension != 2:
        raise ValueError("plot_levels_2d needs dimension = 2")
    from nsus import estimate_leaf_probability

    lim = _auto_lim(ss) if lim is None else lim
    X1, X2, pts, G = _grid(ss, lim, res)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7))

    groups = branch_colour_index(ss)
    handles = {}
    for level in ss.all_levels:
        group = groups[level]
        arr = np.array([s.array for s in level.sample_list])[::every]
        handles[group] = ax.scatter(arr[:, 0], arr[:, 1], s=6, color=_colour(group),
                                    linewidths=0, zorder=2, alpha=0.8)
    # One legend entry per colour group, in group order
    for group in sorted(handles):
        if group < 0:
            label = "Trunk (before any split)" if len(handles) > 1 else "All levels (no split)"
        else:
            leaves = [lf for lf in ss.leaves if groups[lf] == group]
            if leaves:
                pf = sum(estimate_leaf_probability(threshold, lf) for lf in leaves)
                label = f"Branch {group + 1}  (pF ≈ {pf:.2e})"
            else:
                label = f"Branch {group + 1}  (split again)"
        handles[group].set_label(label)
    entries = [handles[group] for group in sorted(handles)]

    # Failure boundary g = threshold (failure side is g >= threshold)
    ax.contour(X1, X2, G, levels=[threshold], colors="k", linewidths=1.5,
               linestyles="--", zorder=3)
    entries += ax.plot([], [], "k--", lw=1.5, label="Limit state g = threshold")
    if boundaries:
        draw_partition_boundaries(ss, ax, X1, X2, pts)
        if any(len(lv._buds) > 1 for lv in ss.all_levels):
            entries += ax.plot([], [], "k-", lw=1.2, label="Partition boundary")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    if title is None:
        from nsus import exceedance_probability
        title = (f"{type(ss).__name__}: {len(ss.all_levels)} levels, "
                 f"{len(ss.leaves)} branch(es), "
                 f"$p_F$ = {exceedance_probability(ss, threshold):.3e}")
    ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(handles=entries, loc="best", fontsize=8, markerscale=2.5, framealpha=0.9)
    if own_fig:
        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150)
            print(f"Saved plot to {filename}")
    return ax


def plot_hill_valley_graph(ss, level_name, ax=None, filename=None):
    """The hill valley graph built on a level, drawn in the input space.
    Nodes are the graph samples, coloured by the community (niche) that was
    chosen; edges join samples with no valley found between them."""
    level = ss.find(level_name)
    info = level.extra_info
    if 'hvg' not in info:
        raise ValueError(f"No hill valley graph was built on level {level_name} "
                         "(it was stopped, or max_branches was already reached).")
    graph = info['hvg'].graph
    X = np.array([s.array for s in info['graph_samples']])
    if X.shape[1] != 2:
        raise ValueError("plot_hill_valley_graph needs dimension = 2")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7))
    allx = np.array([s.array for s in level.sample_list])
    ax.scatter(allx[:, 0], allx[:, 1], s=4, color="#c3c2b7", linewidths=0,
               label="Level samples", zorder=1)
    for a, b in graph.edges():
        ax.plot(X[[a, b], 0], X[[a, b], 1], color="#555555", lw=0.7, zorder=2)
    ax.plot([], [], color="#555555", lw=0.7, label="Graph edges")

    labels = np.zeros(len(X), dtype=int)
    for k, community in enumerate(info.get('partition', [set(range(len(X)))])):
        for node in community:
            labels[node] = k
    for k in np.unique(labels):
        name = f"Niche {k}" if 'partition' in info else "Graph samples (no split)"
        ax.scatter(X[labels == k, 0], X[labels == k, 1], s=60, zorder=3,
                   color=BRANCH_COLOURS[k % len(BRANCH_COLOURS)],
                   edgecolors="white", linewidths=1.5, label=name)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Hill valley graph on level {level_name} "
                 f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)",
                 fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    if own_fig:
        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150)
            print(f"Saved plot to {filename}")
    return ax


def plot_sus_vs_nsus(sus, nss, threshold, lim=None, res=150, every=1, filename=None):
    if lim is None:
        a, b = _auto_lim(sus), _auto_lim(nss)
        lim = (min(a[0], b[0]), max(a[1], b[1]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    plot_levels_2d(sus, threshold, lim=lim, res=res, ax=axes[0], every=every)
    plot_levels_2d(nss, threshold, lim=lim, res=res, ax=axes[1], every=every)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
    return fig, axes
