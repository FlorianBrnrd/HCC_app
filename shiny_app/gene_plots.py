"""
Core analysis + plotting functions for the gene-expression explorer.

This module is a direct port of the logic in
`flared_gene_expression_standalone.ipynb`, plus a couple of small additions
needed to go from "pick a cluster and plot it" (the notebook's workflow) to
"a user types a gene name and gets the relevant plots" (the app's workflow):

  - find_gene_locations(): where does this gene show up in the template
    (as a flared marker, and/or inside metagene groups)?
  - find_best_cluster_for_gene(): pick the single cluster whose cells show
    the strongest expression of the gene, restricted to clusters that have
    a human-readable name (cluster_node_names_tea.pkl) so we don't land on
    a tiny/uninteresting internal tree node.
  - plot_gene_across_all_cells(): the "context" plot -- gene expression
    across every cell in raw matrix order (not clustered / reordered).

Everything else (get_cluster_node_cell_ids, prepare_background,
plot_cell_content, etc.) is unchanged from the notebook.
"""

import re
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import colorcet as cc

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist


# ---------------------------------------------------------------------------
# Tree helper functions (Source: clustering_v1.py)
# ---------------------------------------------------------------------------

def get_cluster_node_cell_ids(tree, node):
    """Return the list of cells contained in a given node ID."""
    return tree.nodes[node].get("cells", [])


def get_cluster_node_descendants(node, tree):
    """Return a list of all descendant nodes (including the node itself)."""
    if node not in tree.nodes:
        raise ValueError(f"Node {node} not found in tree.")
    return [node] + list(nx.descendants(tree, node))


# ---------------------------------------------------------------------------
# Cell-content preparation functions (Source: cell_content.py)
# ---------------------------------------------------------------------------

def load_cluster_color_table(path):
    """
    Load the precomputed gene/cluster color table (one row per gene, with
    columns including cluster_node and cluster_node_color) and collapse it
    to a single cluster_node -> RGBA tuple dict. Colors are stored as
    string tuples in the TSV (e.g. "(0.0, 0.0, 0.0, 1.0)") and need
    literal_eval to become real tuples.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    df["cluster_node_color"] = df["cluster_node_color"].apply(ast.literal_eval)
    return df.groupby("cluster_node")["cluster_node_color"].first().to_dict()


def default_color():
    return (0.827, 0.827, 0.827, 1.0)


def _resolve_cells(node, tree, gene_matrix):
    cells = get_cluster_node_cell_ids(tree=tree, node=node)
    return [c for c in cells if c in gene_matrix.index]


def _expr_pct(cells, gene_matrix):
    expr = gene_matrix.loc[cells]
    return expr.div(expr.sum(axis=1), axis=0) * 100


def get_subtree_linkage(cluster_node, cluster_tree, gene_matrix):
    subset_cells = get_cluster_node_cell_ids(tree=cluster_tree, node=cluster_node)
    subset_expr = gene_matrix.loc[subset_cells]
    Z_sub = linkage(pdist(subset_expr), method='ward', metric='euclidean')
    return Z_sub, subset_cells


def prepare_background(node, gene_matrix, tree, template):
    cells = _resolve_cells(node, tree, gene_matrix)
    if not cells:
        return None, None, None, None

    expr_pct = _expr_pct(cells, gene_matrix)

    series_list, color_list, cluster_list = [], [], []
    for entry in template:
        genes = [g for g in entry["genes"] if g in expr_pct.columns]
        if genes:
            s = expr_pct[genes].sum(axis=1).rename(entry["col"])
        else:
            s = pd.Series(0.0, index=cells, name=entry["col"])
        series_list.append(s)
        color_list.append(entry["color"])
        cluster_list.append(entry["cluster"])

    cell_df = pd.concat(series_list, axis=1)
    return cell_df, list(cell_df.columns), color_list, cluster_list


def prepare_overlay_gene_list(node, gene_matrix, tree, template, genes_to_show, gene_color_map):
    cells = _resolve_cells(node, tree, gene_matrix)
    if not cells:
        return None, None, None, None, None, None

    expr_pct = _expr_pct(cells, gene_matrix)

    fg_series, fg_colors_list, fg_to_gene = [], [], []
    for entry in template:
        if entry["category"] == "flared":
            gene = entry["genes"][0]
            if gene in expr_pct.columns:
                fg_series.append(expr_pct[gene].rename(entry["col"]))
            else:
                fg_series.append(pd.Series(0.0, index=cells, name=entry["col"]))
            fg_colors_list.append(gene_color_map.get(gene, default_color()))
            fg_to_gene.append(gene)
        else:
            genes_present = [g for g in entry["genes"] if g in expr_pct.columns]
            for gene in genes_present:
                fg_series.append(expr_pct[gene].rename(gene))
                fg_colors_list.append(gene_color_map.get(gene, default_color()))
                fg_to_gene.append(gene)

    fg_df = pd.concat(fg_series, axis=1)
    fg_cats = list(fg_df.columns)

    genes_to_show = set(genes_to_show)
    fg_mask = pd.DataFrame(False, index=cells, columns=fg_cats)
    for cat, gene in zip(fg_cats, fg_to_gene):
        if gene is not None and gene in genes_to_show:
            fg_mask[cat] = True

    return cells, fg_df, fg_cats, fg_colors_list, fg_to_gene, fg_mask


# ---------------------------------------------------------------------------
# Rendering functions (Source: plots.py)
# ---------------------------------------------------------------------------

def _merge_contiguous_clusters(cell_df, categories, colors, clusters):
    units = []
    i = 0
    while i < len(categories):
        cluster = clusters[i]
        color = colors[i]
        source = [categories[i]]
        j = i + 1
        while j < len(categories) and clusters[j] == cluster:
            source.append(categories[j])
            j += 1
        units.append((f"cluster_{cluster}", color, source))
        i = j
    return units


def _style_cell_axis(ax, cell_df, title, show_cell_labels=False, orientation=None):
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(cell_df) - 0.5)
    ax.set_xlabel("Expression Contribution (%)", weight="bold")
    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(left=False)
    if show_cell_labels:
        ax.set_yticks(range(len(cell_df)))
        ax.set_yticklabels(cell_df.index, fontsize=3)
        ax.yaxis.tick_right()
    else:
        ax.set_yticks([])
    sns.despine(ax=ax, top=True, left=True)


def render_background(cell_df, categories, colors, clusters, ax=None, merge_by_cluster=False,
                       min_block=0.0, alpha=1.0, subtract_df=None, show_outline=True, title=None,
                       orientation="vertical"):
    if ax is None:
        _, ax = plt.subplots()

    n_cells = len(cell_df)
    bar_fn = ax.bar if orientation == "horizontal" else ax.barh
    span_kw = "bottom" if orientation == "horizontal" else "left"
    size_kw = "width" if orientation == "horizontal" else "height"

    if merge_by_cluster:
        units = _merge_contiguous_clusters(cell_df, categories, colors, clusters)
    else:
        units = [(cat, color, [cat]) for cat, color in zip(categories, colors)]

    for row_i, (cell, row) in enumerate(cell_df.iterrows()):
        pos = 0.0
        for label, color, source_cats in units:
            val = sum(row[c] for c in source_cats)
            if val == 0:
                continue
            gate_val = val
            if subtract_df is not None and label in subtract_df.columns:
                gate_val = max(0.0, val - subtract_df.loc[cell, label])
            if gate_val >= min_block:
                bar_fn(row_i, val, **{span_kw: pos, size_kw: 1.0}, color=color, alpha=alpha,
                       edgecolor=None, lw=0)
            pos += val

    if show_outline:
        bar_fn(range(n_cells), 100, **{span_kw: 0, size_kw: 1.0}, color="none",
               edgecolor="black", lw=0.3)

    _style_cell_axis(ax, cell_df, title, orientation=orientation)
    return ax


def render_foreground_opaque(cell_df, categories, colors, ax, threshold=1.0, fg_mask=None,
                              orientation="vertical"):
    color_of = dict(zip(categories, colors))
    bar_fn = ax.bar if orientation == "horizontal" else ax.barh
    span_kw = "bottom" if orientation == "horizontal" else "left"
    size_kw = "width" if orientation == "horizontal" else "height"

    for row_i, (cell, row) in enumerate(cell_df.iterrows()):
        pos = 0.0
        for cat in categories:
            val = row[cat]
            if val == 0:
                continue
            if color_of[cat] == default_color():
                pos += val
                continue
            visible = val > threshold
            if fg_mask is not None:
                visible = visible and fg_mask.loc[cell, cat]
            if visible:
                bar_fn(row_i, val, **{span_kw: pos, size_kw: 1.0}, color=color_of[cat],
                       alpha=1.0, edgecolor="white", lw=0.3)
            pos += val
    return ax


def _build_foreground_legend(fg_df, fg_cats, fg_colors, threshold=1.0, fg_mask=None):
    handles, labels = [], []
    for cat, color in zip(fg_cats, fg_colors):
        if color == default_color():
            continue
        visible = fg_df[cat] > threshold
        if fg_mask is not None:
            visible = visible & fg_mask[cat]
        if visible.any():
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", lw=0.3))
            labels.append(cat)
    return handles, labels


def plot_subtree(Z_sub, subset_cells, ax=None, linewidth=0.5):
    if ax is None:
        fig, ax = plt.subplots()
    with plt.rc_context({'lines.linewidth': linewidth}):
        dendrogram(Z_sub, labels=subset_cells, orientation='left', leaf_rotation=0,
                   ax=ax, color_threshold=0, no_labels=True, above_threshold_color='black')
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# ---------------------------------------------------------------------------
# plot_cell_content (Source: plots.py) -- overlay_gene_list strategy only
# ---------------------------------------------------------------------------

def plot_cell_content(node, gene_matrix, tree, template, cluster_names, strategy,
                       threshold=1.0, min_remainder=1.0, min_block=None, merge_by_cluster=None,
                       bg_alpha=None, per_cell_top=None, gene_color_map=None, genes_to_show=None,
                       legend_genes=None, seed=42, ax=None):
    is_overlay = strategy.startswith("overlay_")
    if merge_by_cluster is None:
        merge_by_cluster = True if is_overlay else False
    if bg_alpha is None:
        bg_alpha = 0.35 if is_overlay else 1.0
    if min_block is None:
        min_block = min_remainder if is_overlay else 0.0

    legend_fg = None
    legend_fg_mask = None

    if strategy == "transcriptrum":
        cell_df, categories, colors, clusters = prepare_background(node, gene_matrix, tree, template)
        if cell_df is None:
            return None

        def render(ax):
            render_background(cell_df, categories, colors, clusters, ax=ax,
                               merge_by_cluster=merge_by_cluster, min_block=min_block, alpha=bg_alpha)

    elif strategy == "overlay_gene_list":
        if gene_color_map is None or genes_to_show is None:
            raise ValueError("overlay_gene_list requires gene_color_map and genes_to_show")
        bg_df, bg_cats, bg_colors, bg_clusters = prepare_background(node, gene_matrix, tree, template)
        if bg_df is None:
            return None

        result = prepare_overlay_gene_list(node, gene_matrix, tree, template,
                                            genes_to_show=genes_to_show, gene_color_map=gene_color_map)
        if result[0] is None:
            return None
        cells, fg_df, fg_cats, fg_colors_list, fg_to_gene, fg_mask = result
        cell_df = bg_df

        def render(ax):
            render_background(bg_df, bg_cats, bg_colors, bg_clusters, ax=ax,
                               merge_by_cluster=merge_by_cluster, min_block=min_block, alpha=bg_alpha)
            render_foreground_opaque(fg_df, fg_cats, fg_colors_list, ax=ax, threshold=threshold,
                                      fg_mask=fg_mask)

        legend_fg = (fg_df, fg_cats, fg_colors_list)
        legend_fg_mask = fg_mask

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    subtree, subset_cells = get_subtree_linkage(node, tree, gene_matrix)

    BAR_HEIGHT_INCHES = 0.05
    plot_height = max(3, len(cell_df) * BAR_HEIGHT_INCHES)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, plot_height), dpi=150,
                              gridspec_kw={"width_ratios": [0.2, 0.8]})
    ax_tree, ax_cells = axes[0], axes[1]

    plot_subtree(subtree, subset_cells, ax=ax_tree, linewidth=0.5)
    render(ax_cells)

    if legend_fg is not None:
        fg_df_leg, fg_cats_leg, fg_colors_leg = legend_fg
        handles, labels = _build_foreground_legend(fg_df_leg, fg_cats_leg, fg_colors_leg,
                                                     threshold=threshold, fg_mask=legend_fg_mask)
        if legend_genes is not None and handles:
            legend_filter = set(legend_genes)
            filtered = [(h, lbl) for h, lbl in zip(handles, labels) if lbl in legend_filter]
            if filtered:
                handles, labels = zip(*filtered)
            else:
                handles, labels = [], []
        if handles:
            fig.legend(handles, labels, loc="center left", ncol=1, fontsize=6, frameon=False,
                       bbox_to_anchor=(1.0, 0.5))

    name = cluster_names.get(node, f"cluster_{node}")
    fig.suptitle(f"{name}", y=1.02, fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Notebook-local analysis functions
# ---------------------------------------------------------------------------

def gene_color_from_name(gene, palette=None):
    """
    Deterministic color for a gene, based on the gene's own name -- not its
    position within whatever list happens to be passed in. Guarantees the
    same gene always gets the same color everywhere it's plotted (a
    different cluster, a different query, a different session), regardless
    of what other genes are in the list or what order they're in.

    Uses a stable string hash (not Python's built-in hash(), which is
    salted per-process/per-run unless PYTHONHASHSEED is fixed) to pick an
    index into the glasbey palette.
    """
    if palette is None:
        palette = cc.glasbey
    digest = hashlib.md5(gene.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(palette)
    return (*matplotlib.colors.to_rgb(palette[index]), 1.0)


def compute_high_expression_genes_in_cluster(gene_matrix, tree, cluster_node=1, threshold_pct=1.0,
                                              min_cells=1, seed=42):
    cluster_cells = get_cluster_node_cell_ids(tree=tree, node=cluster_node)
    cluster_cells = [c for c in cluster_cells if c in gene_matrix.index]
    if not cluster_cells:
        raise ValueError(f"no cells found under node {cluster_node}")

    sub_matrix = gene_matrix.loc[cluster_cells]
    n_cluster_cells = len(sub_matrix)

    pct = sub_matrix.div(sub_matrix.sum(axis=1), axis=0) * 100
    above = pct > threshold_pct
    n_above = above.sum(axis=0)
    keep = n_above >= min_cells
    qualifying_genes = n_above[keep].index.tolist()

    qualifying = pd.DataFrame({
        'n_cells_above': n_above[keep],
        'frac_cluster_above': n_above[keep] / n_cluster_cells,
        'max_pct': pct[qualifying_genes].max(axis=0),
        'mean_pct': pct[qualifying_genes].mean(axis=0),
    }).sort_values('n_cells_above', ascending=False)

    # color map -- deterministic, based on gene identity (see
    # gene_color_from_name above), not list order/length.
    gene_colors = {gene: gene_color_from_name(gene) for gene in qualifying_genes}
    return qualifying, gene_colors


def top_expressed_per_cell(gene_matrix, tree, cluster_node, top_n=3):
    cells = get_cluster_node_cell_ids(tree=tree, node=cluster_node)
    cells = [c for c in cells if c in gene_matrix.index]
    if not cells:
        return set()
    sub = gene_matrix.loc[cells]
    top_genes = set()
    for cell in sub.index:
        cell_expr = sub.loc[cell]
        cell_expr = cell_expr[cell_expr > 0]
        if len(cell_expr) == 0:
            continue
        top_genes.update(cell_expr.nlargest(top_n).index.tolist())
    return top_genes


# ---------------------------------------------------------------------------
# NEW: gene -> cluster resolution for the web app
# ---------------------------------------------------------------------------

def find_flared_cluster_for_gene(gene, template):
    """
    If `gene` is a designated 'flared' marker gene in the template, return
    its cluster node id. Each flared gene maps to exactly one cluster in
    this dataset. Returns None if the gene isn't a flared marker.
    """
    for entry in template:
        if entry["category"] == "flared" and gene in entry["genes"]:
            return entry["cluster"]
    return None


def get_flared_genes_for_cluster(node, template):
    """Return the list of curated flared marker genes for a cluster node."""
    return [e["genes"][0] for e in template if e["category"] == "flared" and e["cluster"] == node]


def association_quality(query_gene, node, source, annotation_index=None):
    """
    Look up curated PCC / prior-annotation info for (query_gene, node) from
    the precomputed per-cluster annotation tables (see
    load_node_annotation_tables / build_annotation_index below).

    This does not estimate anything live: if an annotation file has been
    supplied for every tissue, every gene/cluster pair either has curated
    data or it doesn't -- and "doesn't" is reported plainly as "not_found"
    rather than guessed at.

    Returns a dict: {status, pcc, precomputed, mean_expression,
    node_annotation, gene_other_annotations}, where status is one of:
      - "established": the query gene IS this cluster's flared marker gene
        (from the older notebook template). Kept as a distinct,
        unambiguous status; if a precomputed row also exists for this
        pair, its PCC is shown instead of leaving it blank.
      - "curated": a precomputed row exists and `used_for_annotation` is
        True -- the gene was actually used to annotate this cluster/tissue.
      - "predicted": a precomputed row exists and `used_for_annotation` is
        False -- correlated with the tissue but not part of how it was
        annotated (guilt-by-association).
      - "not_found": no precomputed annotation row exists for this
        gene/cluster pair.
    """
    if annotation_index is not None:
        node_rows = annotation_index.get(query_gene)
        if node_rows and node in node_rows:
            row = node_rows[node]
            status = "established" if source == "flared" else (
                "curated" if row["used_for_annotation"] else "predicted"
            )
            return {
                "status": status,
                "pcc": row["PCC"],
                "precomputed": True,
                "mean_expression": row.get("mean_expression"),
                "node_annotation": row.get("node_annotation"),
                "gene_other_annotations": row.get("gene_other_annotations"),
            }

    if source == "flared":
        return {"status": "established", "pcc": None, "precomputed": False,
                "mean_expression": None, "node_annotation": None, "gene_other_annotations": None}

    return {"status": "not_found", "pcc": None, "precomputed": False,
            "mean_expression": None, "node_annotation": None, "gene_other_annotations": None}




# ---------------------------------------------------------------------------
# NEW: precomputed per-cluster PCC / prior-annotation tables
# ---------------------------------------------------------------------------
#
# One file per cluster (e.g. "node_6_tail_hypodermis.numbers"), each with
# columns: node, WBgeneID, gene_name, mean_expression, PCC,
# used_for_annotation, node_annotation, gene_other_annotations.
# `used_for_annotation` is the curated ground truth for whether a gene was
# actually part of how that cluster/tissue was annotated ("established"/
# "curated"), vs. merely correlated with it ("predicted" / guilt-by-
# association). Where available, this supersedes the live PCC heuristic
# above.

_ANNOTATION_EXPECTED_COLS = {
    "node", "gene_name", "mean_expression", "PCC", "used_for_annotation", "node_annotation",
}


def load_node_annotation_table(path):
    """
    Load one precomputed per-cluster annotation file into a standardized
    DataFrame. Supports .tsv only.
    """
    path = Path(path)

    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported annotation file type: {path.suffix}")

    missing = _ANNOTATION_EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")

    df["node"] = df["node"].astype(int)
    df["used_for_annotation"] = df["used_for_annotation"].astype(bool)
    df["PCC"] = df["PCC"].astype(float)
    df["mean_expression"] = df["mean_expression"].astype(float)
    return df


def load_node_annotation_tables(dir_path):
    """
    Load every per-cluster annotation file in a directory (one file per
    node) and combine into a single DataFrame. Malformed/unreadable files
    are skipped (with a printed warning) rather than crashing the app, so
    one bad file doesn't take down the whole tool. Returns an empty
    DataFrame with the expected columns if the directory doesn't exist or
    has no matching files.
    """
    dir_path = Path(dir_path)
    cols = ["node", "WBgeneID", "gene_name", "mean_expression", "PCC",
            "used_for_annotation", "node_annotation", "gene_other_annotations"]
    if not dir_path.exists():
        return pd.DataFrame(columns=cols)

    frames = []
    for path in sorted(dir_path.iterdir()):
        if path.suffix.lower() == ".tsv":
            try:
                frames.append(load_node_annotation_table(path))
            except Exception as e:
                print(f"Warning: couldn't load annotation file {path.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def reference_gene_for_node(node, annotation_df):
    """
    Reference gene for a cluster: the gene with the highest precomputed
    mean_expression among that cluster's curated annotation rows (from
    load_node_annotation_tables() / load_annotation_data(), loaded once and
    cached -- not recomputed per query).

    Returns (gene_name, mean_expression), or (None, None) if no annotation
    rows exist for this node.
    """

    sub = annotation_df[annotation_df["node"] == node]
    if sub.empty:
        return None, None
    row = sub.loc[sub["mean_expression"].idxmax()]
    return row["gene_name"], float(row["mean_expression"])


def build_annotation_index(annotation_df):
    """
    Build gene_name -> {node: row_dict} for O(1) lookup of precomputed
    PCC / used_for_annotation / node_annotation / etc. per (gene, node).
    """
    index = {}
    for _, row in annotation_df.iterrows():
        index.setdefault(row["gene_name"], {})[int(row["node"])] = row.to_dict()
    return index


def node_annotation_label(annotation_df, node):
    """Return the curated tissue label (node_annotation) for a node, if available."""
    sub = annotation_df[annotation_df["node"] == node]
    if sub.empty:
        return None
    return sub["node_annotation"].iloc[0]


def build_gene_cluster_index(template):
    """
    Build gene -> list of clusters the gene appears in, from the template's
    'specific' entries (per-cluster metagene groups). This is a one-time,
    cheap-to-build lookup (the template has ~2k entries, independent of
    matrix size) that tells us every cluster a gene is curated to belong to,
    without ever touching the expression matrix.
    """
    index = {}
    for entry in template:
        if entry["category"] != "specific":
            continue
        cluster = entry["cluster"]
        for gene in entry["genes"]:
            index.setdefault(gene, set()).add(cluster)
    return index


def compute_cell_totals(gene_matrix):
    """
    Precompute each cell's total expression (row sum) once. This is the
    expensive part of turning raw counts into per-cell percentages, and it
    doesn't depend on which cluster/node we're looking at -- so computing it
    once up front (and reusing it for every node) turns an O(n_nodes *
    n_genes) scan into an O(n_genes) one-time cost plus a cheap O(n_nodes)
    lookup per gene query.
    """
    return gene_matrix.sum(axis=1)


def _expression_summary_for_candidates(gene, gene_matrix, tree, candidate_nodes, cell_totals):
    """
    Among a (small) set of candidate cluster nodes, pick the one with the
    highest mean per-cell expression of `gene`. Returns (best_node, info) or
    (None, None) if none of the candidates have usable cells.
    """
    gene_col = gene_matrix[gene]
    best_node, best_mean_pct, best_max_pct, best_n_cells = None, -1.0, -1.0, 0

    for node in candidate_nodes:
        if node not in tree.nodes:
            continue
        cells = _resolve_cells(node, tree, gene_matrix)
        if not cells:
            continue
        totals = cell_totals.loc[cells]
        pct = gene_col.loc[cells].div(totals) * 100
        mean_pct = pct.mean()
        if mean_pct > best_mean_pct:
            best_node = node
            best_mean_pct = mean_pct
            best_max_pct = pct.max()
            best_n_cells = len(cells)

    if best_node is None:
        return None, None
    return best_node, {
        "mean_pct": best_mean_pct,
        "max_pct": best_max_pct,
        "n_cells": best_n_cells,
    }


def find_best_cluster_for_gene(gene, gene_matrix, tree, cluster_names, gene_cluster_index,
                                cell_totals=None, allow_expression_fallback=True):
    """
    Resolve `gene` to the single best cluster to display, without relying on
    a full expression scan (which gets slower as the matrix grows).

    Strategy, in order:
      1. Flared marker: handled separately by find_flared_cluster_for_gene
         before this is even called (kept out of here so the caller can
         label that case distinctly).
      2. Template lookup: if the gene appears in the curated 'specific'
         metagene groups (gene_cluster_index, built once from the template
         -- independent of matrix size), use that. If it appears in exactly
         one cluster, use it directly with no expression computation at
         all. If it appears in a handful of clusters, break the tie by
         picking whichever of *those* clusters has the highest mean
         expression of the gene (cheap: only a few candidates, not a full
         scan).
      3. Last resort: if the gene isn't in the template at all, optionally
         fall back to scanning every named cluster by mean expression. This
         is the slow path and its cost grows with the matrix, so it's
         skipped by default once `allow_expression_fallback=False`.

    Returns (node_id, info_dict) where info_dict has a "source" key
    ("template" or "expression_scan") plus expression stats when available,
    or (None, None) if the gene can't be resolved.
    """
    if gene not in gene_matrix.columns:
        return None, None

    if cell_totals is None:
        cell_totals = compute_cell_totals(gene_matrix)

    template_clusters = gene_cluster_index.get(gene)

    if template_clusters:
        template_clusters = sorted(template_clusters)
        if len(template_clusters) == 1:
            node = template_clusters[0]
            cells = _resolve_cells(node, tree, gene_matrix)
            info = {"source": "template", "n_clusters_in_template": 1, "n_cells": len(cells)}
            if cells:
                pct = gene_matrix[gene].loc[cells].div(cell_totals.loc[cells]) * 100
                info["mean_pct"] = pct.mean()
                info["max_pct"] = pct.max()
            return node, info

        node, expr_info = _expression_summary_for_candidates(
            gene, gene_matrix, tree, template_clusters, cell_totals
        )
        if node is not None:
            expr_info["source"] = "template"
            expr_info["n_clusters_in_template"] = len(template_clusters)
            return node, expr_info
        # candidates existed but had no usable cells -- fall through

    if not allow_expression_fallback:
        return None, None

    # Gene isn't in the curated template at all -- fall back to scanning
    # every named cluster by mean expression (slow; scales with matrix size).
    candidate_nodes = list(cluster_names.keys())
    node, expr_info = _expression_summary_for_candidates(
        gene, gene_matrix, tree, candidate_nodes, cell_totals
    )
    if node is None:
        return None, None
    expr_info["source"] = "expression_scan"
    return node, expr_info


# ---------------------------------------------------------------------------
# NEW: tissue name -> cluster/flared-gene resolution
# ---------------------------------------------------------------------------

def parse_cluster_tissue(name):
    """
    Parse the tissue/cell-type label out of a cluster name string like
    '[2] cluster nspc-20 – excretory gland cell (34/36)' -> 'excretory gland cell'.
    Returns None if the string doesn't match the expected pattern.

    Note: only the en-dash (\u2013, "–") is treated as the separator, not a
    plain ASCII hyphen -- gene names like "nspc-20" contain hyphens too, and
    matching on those would grab the wrong substring.
    """
    m = re.search(r"\u2013\s*(.+?)\s*\(\d+/\d+\)\s*$", name)
    if m:
        return m.group(1).strip()
    return None


def build_tissue_index(cluster_names, template):
    """
    Build tissue name (lowercased) -> list of (cluster_node, flared_genes)
    for every cluster that has a parseable tissue label, regardless of
    whether it has a designated flared marker gene. `flared_genes` is an
    empty tuple for clusters with none -- callers should fall back to that
    cluster's most-expressed gene as the representative (see
    most_expressed_gene_in_cluster) rather than treating "no flared gene"
    as "tissue not found".
    """
    flared_by_cluster = {}
    for entry in template:
        if entry["category"] == "flared":
            flared_by_cluster.setdefault(entry["cluster"], []).append(entry["genes"][0])

    index = {}
    for node, name in cluster_names.items():
        tissue = parse_cluster_tissue(name)
        if tissue is None:
            continue
        flared_genes = tuple(sorted(flared_by_cluster.get(node, [])))
        index.setdefault(tissue, []).append((node, flared_genes))
    return index


def all_tissue_names(tissue_index):
    """
    Sorted list of every distinct tissue/cell-type name known to the
    dataset (the tissue_index's keys, already lowercased by
    build_tissue_index). Used to populate a dropdown so users can only
    select tissue names that actually exist, rather than free-typing a
    possibly-misspelled one.
    """
    return sorted(tissue_index.keys())


def find_tissue_matches(query, tissue_index):
    """
    Case-insensitive match of `query` against tissue names: exact matches
    first, then substring matches (e.g. "gland" matches "excretory gland
    cell"). Returns a list of (tissue_label, cluster_node, flared_genes)
    triples, one per matching cluster -- `flared_genes` may be an empty
    tuple, in which case the caller should resolve a representative gene
    via most_expressed_gene_in_cluster instead.
    """
    query_l = query.strip()
    if not query_l:
        return []

    exact, partial = [], []
    for tissue, entries in tissue_index.items():
        if tissue == query_l:
            bucket = exact
        elif query_l in tissue:
            bucket = partial
        else:
            continue
        for node, flared_genes in entries:
            bucket.append((tissue, node, flared_genes))

    exact.sort(key=lambda t: (t[0], t[1]))
    partial.sort(key=lambda t: (t[0], t[1]))
    return exact + partial


def most_expressed_gene_in_cluster(node, gene_matrix, tree):
    """
    The single most highly expressed gene (by mean expression across the
    cluster's cells) -- used as the representative gene for a tissue when
    its cluster has no designated flared marker gene.

    Returns (gene_name, mean_expression) or (None, None) if the cluster has
    no usable cells.
    """
    cells = _resolve_cells(node, tree, gene_matrix)
    if not cells:
        return None, None
    means = gene_matrix.loc[cells].mean(axis=0)
    gene = means.idxmax()
    return gene, float(means[gene])


# ---------------------------------------------------------------------------
# NEW: gene expression across ALL cells, in raw matrix order (context plot)
# ---------------------------------------------------------------------------

def cluster_color_for_node(node, colormap="tab20"):
    """
    Deterministic color for a given cluster node, independent of any
    particular plot's node ordering -- same node always maps to the same
    color across pages/reruns (unlike tree_figure's node_colors, which is
    assigned by enumeration order over whatever subset of nodes is being
    drawn that time).
    """
    cmap = matplotlib.colormaps[colormap] if hasattr(matplotlib, "colormaps") else plt.get_cmap(colormap)
    return cmap(node % cmap.N)


def plot_gene_across_all_cells(gene, gene_matrix, highlight_cells=None, highlight_color=None,
                                figsize=(15, 3)):
    """
    Line + fill plot of a gene's expression (RPM) across every cell in the
    matrix, in the matrix's native (reindexed) row order -- gives context
    for how the highlighted cluster's expression compares to the whole
    dataset.

    The full profile is drawn in grey; the cells belonging to the
    highlighted cluster are redrawn on top in `highlight_color` (e.g. the
    cluster's own color), so the region of the profile corresponding to
    that cluster is immediately visible against the grey background.

    highlight_cells: optional iterable of cell IDs to highlight (e.g. the
    cells in the cluster shown in the companion plot).
    highlight_color: color for the highlighted segment. Defaults to crimson
    if highlight_cells is given but no color is specified.
    """
    values = gene_matrix[gene].values
    n_cells = len(values)
    x = np.arange(n_cells)

    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize, dpi=300)

        # full profile, grey
        ax.plot(x, values, linewidth=1, marker=None, alpha=0.5, color="#888888", zorder=1)
        ax.fill_between(x, values, color="#888888", alpha=0.1, zorder=1)

        if highlight_cells is not None:
            highlight_set = set(highlight_cells)
            mask = gene_matrix.index.isin(highlight_set)
            if mask.any():
                color = highlight_color if highlight_color is not None else "crimson"
                # NaN outside the highlighted cells so the overlay only
                # draws (and fills) over that segment of the x-axis.
                highlighted_values = np.where(mask, values, np.nan)
                ax.plot(x, highlighted_values, linewidth=1, marker=None, alpha=0.9,
                        color=color, zorder=10)
                ax.fill_between(x, highlighted_values, color=color, alpha=0.25, zorder=10)

        ax.set_xlim(-1, n_cells + 1)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(bottom=-5)
        ax.set_ylabel(f"{gene} (RPM)", weight="bold")
        sns.despine(ax=ax)
        # Fixed, constant margins (not auto-fit to label width) -- this is
        # what keeps the plotted x-axis region pixel-aligned across genes
        # when several of these figures are stacked vertically. Auto/tight
        # layout would let the left margin shift with each gene's label
        # width and y-tick digit count.
        fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.12)

    return fig

def wormbase_anatomy_link(annotation_text):
    """
    Parse a "{name} WBbt:XXXXXXX" style annotation string (as stored in the
    node_annotation / gene_other_annotations columns) into a (label, url)
    pair linking to that term's WormBase anatomy page. Returns (None, None)
    if no WBbt ID is found (e.g. empty/missing annotation).
    """
    if not annotation_text:
        return None, None
    match = re.search(r"WBbt:\d+", annotation_text)
    if not match:
        return None, None
    wbbt_id = match.group(0)
    label = annotation_text.replace(wbbt_id, "").strip()
    url = f"https://wormbase.org/species/all/anatomy_term/{wbbt_id}"
    return label, url