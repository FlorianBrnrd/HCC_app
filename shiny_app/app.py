"""
HCC Explorer -- Shiny for Python prototype

Two-page navbar (spectrum + context), plotly-based interactive spectrum
figure with dendrogram, and hierarchy navigation (parent / children).
"""

import re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from plotly.subplots import make_subplots
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget
from matplotlib.ticker import FuncFormatter
from ipyaggrid import Grid

import data_loaders as dl
import gene_plots as gp

MAX_EXTRA_GENES = 200  # pre-registered output slots (effectively unbounded)
MAX_NAV_CELLS = 1000  # nav buttons for clusters larger than this are disabled


# ---------------------------------------------------------------------------
# Process-scoped data
# ---------------------------------------------------------------------------

if not dl.matrix_exists():
    raise SystemExit(
        f"Gene expression matrix not found at {dl.MATRIX_REINDEXED_PATH}.\n"
        "Place HCC_gene_count_matrix_rpm_reindexed_unsupervised.tsv in the "
        "data/ folder (or set GENE_APP_DATA_DIR to point at one) and retry."
    )

tree = dl.load_tree()
template = dl.load_spectrum()
cluster_names = dl.load_cluster_names()
gene_matrix = dl.load_gene_matrix()
cell_totals = dl.load_cell_totals()
gene_cluster_index = dl.load_gene_cluster_index()
tissue_index = dl.load_tissue_index()
annotation_df, annotation_index = dl.load_annotation_data()
cluster_colors = dl.load_cluster_colors()

tissue_options = gp.all_tissue_names(tissue_index)
cell_annotation_colors = dl.load_cell_annotation_colors()


def _cluster_size(nid):
    """Number of cells in a cluster node."""
    return len(gp.get_cluster_node_cell_ids(tree=tree, node=nid))





def _compact_number(x, pos=None):
    """Format axis ticks compactly: 12000 → '12k', 1_500_000 → '1.5M'."""
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:g}G"
    if ax >= 1e6:
        return f"{x/1e6:g}M"
    if ax >= 1e3:
        return f"{x/1e3:g}k"
    if ax == int(ax):
        return f"{int(x)}"
    return f"{x:g}"



# ---------------------------------------------------------------------------
# Interactive plotly spectrum figure
# ---------------------------------------------------------------------------
def plot_annotation_strip_matplotlib(gene_matrix, cell_colors, offset=0):
    """Horizontal color strip: one colored bar per cell showing its main-cluster
    assignment, with thin vertical dividers at cluster boundaries and cell-index
    labels above. Uses the SAME axes rectangle (via set_position) as the gene
    plots so the first cell lands at the same figure-x pixel in every plot.

    `offset` is added to the drawn cell indices when computing text labels so
    that a custom range (say 4000..4200) shows those absolute positions rather
    than 0..200 local to the slice.
    """
    if not cell_colors:
        return None
    cells = list(gene_matrix.index)
    n = len(cells)
    if n == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.3), dpi=150)
    default_color = "#eeeeee"

    # Draw color bars and detect where the annotation color changes
    boundaries = [0]
    prev_color = None
    for i, cell in enumerate(cells):
        color = cell_colors.get(cell, default_color)
        ax.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.9, lw=0)
        if i > 0 and color != prev_color:
            boundaries.append(i)
        prev_color = color

    # Vertical dividers at cluster boundaries (skip 0 since it's the left edge)
    for b in boundaries[1:]:
        ax.axvline(x=b - 0.5, color="black", linewidth=0.4, alpha=0.7)

    # Cell-index labels above the strip; thin out if there are too many
    label_positions = list(boundaries)
    if (n - 1) not in label_positions:
        label_positions.append(n - 1)
    max_labels = 20
    if len(label_positions) > max_labels:
        step = max(1, len(label_positions) // max_labels)
        thinned = label_positions[::step]
        if (n - 1) not in thinned:
            thinned.append(n - 1)
        label_positions = thinned
    for pos in label_positions:
        ax.text(
            pos, 1.05, str(pos + offset),
            ha="left", va="bottom", fontsize=5, color="#555",
            rotation=45,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    # Exact axes rectangle: identical [left, width] as the gene plot below.
    ax.set_position([0.10, 0.10, 0.88, 0.55])
    return fig



def plot_gene_across_all_cells_matplotlib(gene, gene_matrix, highlight_cells,
                                          highlight_color):
    """Matplotlib version of gene expression across cells.

    Full-width pale-gray line for all cells; the cluster's cells drawn as a
    thicker colored line on top over the same x-range. Gene name as the plot
    title (top-left), no x-ticks, y-axis kept for RPM readout. X-limits are
    locked to the full column range so plots stack with consistent widths.
    """
    if gene not in gene_matrix.columns:
        return None
    values = gene_matrix[gene].values
    cells = list(gene_matrix.index)
    n = len(cells)
    x = np.arange(n)

    highlight_set = set(highlight_cells)
    highlight_idx = np.array([c in highlight_set for c in cells])
    # Same values, NaN outside the cluster so the colored line breaks
    highlighted_values = np.where(highlight_idx, values, np.nan)

    if isinstance(highlight_color, tuple):
        color_arg = highlight_color[:3]
    else:
        color_arg = highlight_color

    fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
    ax.plot(x, values, color="#cccccc", linewidth=1)
    ax.plot(x, highlighted_values, color=color_arg, linewidth=2.2)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_title(gene, loc="left", fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_number))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    # Exact axes rectangle: identical [left, width] as the annotation strip.
    # Bottom+height differ (strip has less vertical room; gene plot leaves
    # room at top for the title). set_position pins the rectangle so wide
    # y-tick labels can't shift the y-axis around.
    ax.set_position([0.10, 0.10, 0.88, 0.72])
    return fig


def plot_cell_content_plotly(node, gene_matrix, tree, template, cluster_names,
                             gene_color_map, genes_to_show):
    """Interactive plotly version of gp.plot_cell_content.

    Dendrogram on the left, colored gene contributions per cell on the right.
    Highlighted genes overlay a single gray "other genes" background at their
    true cumulative x-positions (matches matplotlib's `pos += val` layout).
    Trace count = n_highlighted_genes + 2.
    """
    result = gp.prepare_overlay_gene_list(
        node, gene_matrix, tree, template,
        genes_to_show=genes_to_show, gene_color_map=gene_color_map,
    )
    if result[0] is None:
        return None
    cells, fg_df, fg_cats, fg_colors_list, fg_to_gene, fg_mask = result

    # Reorder cells to match subtree dendrogram leaf order
    subtree, subset_cells = gp.get_subtree_linkage(node, tree, gene_matrix)
    dendro = sch.dendrogram(subtree, no_plot=True)
    ordered_cells = [subset_cells[i] for i in dendro["leaves"]]
    fg_df = fg_df.loc[ordered_cells]
    fg_mask = fg_mask.loc[ordered_cells]
    n_cells = len(ordered_cells)

    # True x-start of each gene in each cell (mirrors matplotlib's pos accumulator)
    starts = fg_df.cumsum(axis=1).shift(axis=1, fill_value=0.0)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.15, 0.85],
        shared_yaxes=True, horizontal_spacing=0.005,
    )

    # --- Dendrogram (left) ---
    for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"]):
        fig.add_trace(go.Scatter(
            x=[-d for d in dcoord],
            y=[(y - 5) / 10 for y in icoord],
            mode="lines", line=dict(color="#333", width=1),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # --- Right panel ---
    y_pos = list(range(n_cells))
    threshold_pct = 0.1

    # Gray background (0..100 per cell)
    fig.add_trace(go.Bar(
        x=[100] * n_cells, y=y_pos, orientation="h",
        marker=dict(color="#ffffff", line=dict(width=0)),
        name="other genes",
        hoverinfo="skip",
        showlegend=True,
    ), row=1, col=2)

    def to_rgb(c):
        if isinstance(c, tuple):
            return f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        return c

    # Highlighted-gene overlays
    for cat, color, gene in zip(fg_cats, fg_colors_list, fg_to_gene):
        vals = fg_df[cat].values
        mask = fg_mask[cat].values
        visible_per_cell = mask & (vals > threshold_pct)
        if not visible_per_cell.any():
            continue
        color_str = to_rgb(color)
        per_bar_colors = [
            color_str if v else "rgba(0,0,0,0)" for v in visible_per_cell
        ]
        hover_text = [
            f"<b>{gene}</b><br>Contribution: {v:.2f}%"
            if vis else None
            for c, v, vis in zip(ordered_cells, vals, visible_per_cell)
        ]
        fig.add_trace(go.Bar(
            x=vals,
            base=starts[cat].values,
            y=y_pos,
            orientation="h",
            marker=dict(color=per_bar_colors, line=dict(width=0)),
            name=gene,
            text=hover_text,
            textposition="none",
            hoverinfo="text",
        ), row=1, col=2)

    # Black outline per cell (drawn last so it sits on top)
    edge_w = max(0.2, min(0.6, 100 / max(n_cells, 1)))
    fig.add_trace(go.Bar(
        x=[100] * n_cells, y=y_pos, orientation="h",
        marker=dict(color="rgba(0,0,0,0)",
                    line=dict(color="black", width=edge_w)),
        hoverinfo="skip",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        barmode="overlay",
        height=max(500, n_cells * 8),
        margin=dict(l=10, r=10, t=50, b=40),
        legend=dict(orientation="v", x=1.02, y=1),
        dragmode="pan",
        bargap=0,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                    row=1, col=1)
    fig.update_xaxes(title="Expression Contribution (%)",
                    range=[0, 100], row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

explorer_sidebar = ui.sidebar(
    ui.input_radio_buttons(
        "search_mode", "Search by:",
        choices=["Gene", "Tissue"], inline=True,
    ),
    ui.panel_conditional(
        "input.search_mode == 'Gene'",
        ui.input_text(
            "gene_query", "Gene name:",
            placeholder="e.g. nspc-20 or F40F8.4",
        ),
    ),
    ui.panel_conditional(
        "input.search_mode == 'Tissue'",
        ui.input_selectize(
            "tissue_query", "Tissue / cell-type:",
            choices=[""] + list(tissue_options),
            selected="",
        ),
    ),
    ui.output_ui("cluster_picker_ui"),
    ui.panel_conditional(
        "input.page === 'Cells associated with this gene'",
        ui.input_slider(
            "threshold", "Threshold (%)",
            min=0.1, max=2.0, value=2.0, step=0.1,
        ),
    ),
    ui.hr(),                        # ← sidebar-level, NOT inside the panel_conditional above
    ui.output_ui("nav_header"),     # ← same
    width=340,
)

spectrum_page = ui.nav_panel(
    "Cells associated with this gene",
    ui.output_ui("spectrum_section"),
)

context_page = ui.nav_panel(
    "Gene expression across all cells",
    ui.output_ui("context_section"),
)

app_ui = ui.page_navbar(
    spectrum_page,
    context_page,
    sidebar=explorer_sidebar,
    header=ui.TagList(
        ui.card(ui.output_ui("results_header"), class_="mb-4"),
        # Server → client sync for Plot-button state. The grid reads
        # window.HCC_plotted_genes when drawing its Plot column cells, so
        # any server-side change (add-gene input, clear button, new query)
        # needs to update that set and repaint the grid.
        ui.tags.script("""
            document.addEventListener('DOMContentLoaded', function() {
                if (!window.Shiny) return;
                Shiny.addCustomMessageHandler('plotted_genes_updated', function(msg) {
                    window.HCC_plotted_genes = new Set(msg.genes || []);
                    if (window.HCC_grid_api) {
                        // refreshCells re-invokes the button cellRenderer;
                        // redrawRows re-invokes getRowStyle for backgrounds.
                        if (window.HCC_grid_api.refreshCells) {
                            window.HCC_grid_api.refreshCells({force: true});
                        }
                        if (window.HCC_grid_api.redrawRows) {
                            window.HCC_grid_api.redrawRows();
                        }
                    }
                });
            });
        """),
    ),
    title="HCC Explorer",
    id="page",
    padding="1rem",
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):

    # ----- Navigation state --------------------------------------------------
    navigation_override = reactive.value(None)
    plotted_genes = reactive.value([])                              # gene table "Plot" button state
    add_gene_error_msg = reactive.value(None)                       # inline error under add-gene input
    table_collapsed = reactive.value(False)                         # left-column fold state

    @reactive.effect
    @reactive.event(input.toggle_table)
    def _on_toggle_table():
        table_collapsed.set(not table_collapsed.get())

    @reactive.effect
    def _reset_override_on_query():
        input.search_mode()
        input.gene_query()
        input.tissue_query()
        try:
            input.cluster_choice()
        except Exception:
            pass
        navigation_override.set(None)
        plotted_genes.set([])                                        # clear on new query too
        add_gene_error_msg.set(None)

    @reactive.effect
    @reactive.event(input.plot_gene_click, ignore_none=True)
    def _on_plot_click():
        payload = input.plot_gene_click()
        gene = (payload or {}).get("gene")
        if not gene:
            return
        current = plotted_genes.get()
        if gene in current:
            plotted_genes.set([g for g in current if g != gene])   # toggle off
        else:
            plotted_genes.set(current + [gene])

    @reactive.effect
    @reactive.event(input.add_gene_btn)
    def _on_add_gene():
        gene = (input.add_gene_query() or "").strip()
        if not gene:
            add_gene_error_msg.set(None)
            return
        # Case-insensitive lookup against the expression matrix
        matches = [c for c in gene_matrix.columns if c.lower() == gene.lower()]
        if not matches:
            add_gene_error_msg.set(f"'{gene}' not found in the expression matrix.")
            return
        canonical = matches[0]
        current = plotted_genes.get()
        if canonical in current:
            # Toggle off if already plotted
            plotted_genes.set([g for g in current if g != canonical])
            add_gene_error_msg.set(f"Removed '{canonical}'.")
        else:
            plotted_genes.set(current + [canonical])
            add_gene_error_msg.set(None)

    @reactive.effect
    @reactive.event(input.clear_extra_plots)
    def _on_clear_plots():
        plotted_genes.set([])
        add_gene_error_msg.set(None)

    @reactive.effect
    def _sync_plotted_genes_to_client():
        """Push the current plotted_genes list to the browser so the Ag-Grid
        button cells reflect any server-side change (add-gene input, clear
        button, new-query reset). No-op on the roundtrip when the client
        already made the change itself -- the message just confirms.
        """
        genes = list(plotted_genes.get())
        session.send_custom_message("plotted_genes_updated", {"genes": genes})

    @reactive.effect
    @reactive.event(input.range_reset)
    def _on_range_reset():
        n = len(gene_matrix.index)
        session.send_input_message("range_start", {"value": 0})
        session.send_input_message("range_end", {"value": n - 1})

    @reactive.effect
    @reactive.event(input.zoom_to_cluster_btn)
    def _on_zoom_to_cluster():
        ctx = cluster_ctx()
        if ctx is None:
            return
        # Map the current cluster's cell IDs back to positions in the full
        # gene matrix, then set the range inputs to span that block.
        pos_by_cell = {cell: i for i, cell in enumerate(gene_matrix.index)}
        positions = [pos_by_cell[c] for c in ctx["cluster_cells"] if c in pos_by_cell]
        if not positions:
            return
        session.send_input_message("range_start", {"value": min(positions)})
        session.send_input_message("range_end", {"value": max(positions)})

    @reactive.effect
    @reactive.event(input.nav_parent, ignore_none=True)
    def _on_nav_parent():
        ctx = cluster_ctx()
        if ctx is None:
            return
        parents = list(tree.predecessors(ctx["node"]))
        if parents and _cluster_size(parents[0]) <= MAX_NAV_CELLS:
            navigation_override.set(parents[0])

    @reactive.effect
    @reactive.event(input.nav_child_0, ignore_none=True)
    def _on_nav_child_0():
        ctx = cluster_ctx()
        if ctx is None:
            return
        children = list(tree.successors(ctx["node"]))
        if children and _cluster_size(children[0]) <= MAX_NAV_CELLS:
            navigation_override.set(children[0])

    @reactive.effect
    @reactive.event(input.nav_child_1, ignore_none=True)
    def _on_nav_child_1():
        ctx = cluster_ctx()
        if ctx is None:
            return
        children = list(tree.successors(ctx["node"]))
        if len(children) > 1 and _cluster_size(children[1]) <= MAX_NAV_CELLS:
            navigation_override.set(children[1])

    # ----- Query resolution --------------------------------------------------

    @reactive.calc
    def tissue_entries():
        if input.search_mode() != "Tissue":
            return []
        query = (input.tissue_query() or "").strip()
        if not query:
            return []
        # Exact-name match only (case-insensitive). Prevents "AB" from also
        # returning ABppapparpr, etc.
        matches = [
            m for m in gp.find_tissue_matches(query, tissue_index)
            if m[0].lower() == query.lower()
        ]
        entries, seen = [], set()
        for tissue, node, _flared in matches:
            if node in seen:
                continue
            ref_gene, _ = gp.reference_gene_for_node(node, annotation_df)
            if ref_gene is None:
                continue
            seen.add(node)
            cluster_label = cluster_names.get(node, f"cluster_{node}")
            parts = cluster_label.split("–", 1)
            suffix = parts[1].strip() if len(parts) == 2 else cluster_label
            label = f"[{node}] cluster {ref_gene} – {suffix}"
            entries.append((label, node, tissue, ref_gene))
        entries.sort(key=lambda e: (
            query.lower() != e[2],
            -_cluster_size(e[1]),
            e[1],
        ))
        return entries

    @render.ui
    def cluster_picker_ui():
        entries = tissue_entries()
        if len(entries) <= 1:
            return None
        return ui.input_select(
            "cluster_choice", "Choose a cluster:",
            choices={e[0]: e[0] for e in entries},
        )

    @reactive.calc
    def resolved():
        if input.search_mode() == "Gene":
            query = (input.gene_query() or "").strip()
            if not query:
                return None
            if query not in gene_matrix.columns:
                return {"error": f"'{query}' wasn't found as a gene in the expression matrix."}
            flared = gp.find_flared_cluster_for_gene(query, template)
            if flared is not None:
                node = flared
                source = "flared"
                match_info = "most expressed gene specific to this cell group"
            else:
                node, info = gp.find_best_cluster_for_gene(
                    query, gene_matrix, tree, cluster_names, gene_cluster_index,
                    cell_totals=cell_totals,
                )
                if node is None:
                    return {"error": f"Couldn't find any cluster associated with '{query}'."}
                source = info["source"]
                if source == "template":
                    n = info["n_clusters_in_template"]
                    match_info = (
                        "the group of cells that this gene is most specific to"
                        if n == 1 else
                        f"appears in {n} clusters in the reference groupings; "
                        f"shown here is the one with highest expression "
                        f"(mean {info['mean_pct']:.2f}% of {info['n_cells']} cells)"
                    )
                else:
                    match_info = (
                        "gene not in the reference groupings -- fell back to scanning "
                        "all named clusters by expression "
                        f"(mean {info['mean_pct']:.2f}% / max {info['max_pct']:.2f}% "
                        f"of {info['n_cells']} cells)"
                    )
            assoc = gp.association_quality(
                query, node, source, annotation_index=annotation_index,
            )
            return {
                "gene": query, "node": node,
                "match_info": match_info, "association": assoc,
            }

        # Tissue mode
        query = (input.tissue_query() or "").strip()
        if not query:
            return None
        entries = tissue_entries()
        if not entries:
            return {"error": f"No clusters matched tissue '{query}'."}
        if len(entries) == 1:
            _, node, tissue, gene = entries[0]
        else:
            try:
                choice = input.cluster_choice()
            except Exception:
                choice = entries[0][0]
            _, node, tissue, gene = next(
                (e for e in entries if e[0] == choice), entries[0]
            )
        flared_genes = gp.get_flared_genes_for_cluster(node, template)
        assoc_source = "flared" if gene in flared_genes else "curated_table"
        assoc = gp.association_quality(
            gene, node, assoc_source, annotation_index=annotation_index,
        )
        return {
            "gene": gene, "node": node,
            "match_info": f"tissue match for '{query}' → {tissue}",
            "association": assoc,
        }

    @reactive.calc
    def cluster_ctx():
        r = resolved()
        if r is None or "error" in r:
            return None
        override = navigation_override.get()
        node = override if override is not None else r["node"]
        cluster_cells = gp.get_cluster_node_cell_ids(tree=tree, node=node)
        cluster_color = cluster_colors.get(node, gp.cluster_color_for_node(node))
        ref_gene, _ = gp.reference_gene_for_node(node, annotation_df)
        return {
            **r,
            "node": node,              # override wins over r["node"]
            "cluster_cells": cluster_cells,
            "cluster_color": cluster_color,
            "ref_gene": ref_gene,
        }

    @reactive.calc
    def context_matrix():
        ctx = cluster_ctx()
        if ctx is None:
            return gene_matrix
        # Range inputs drive everything now — zoom-to-cluster is a shortcut
        # button that sets these values, not a separate state.
        n = len(gene_matrix.index)
        try:
            start = int(input.range_start() or 0)
        except Exception:
            start = 0
        try:
            end = int(input.range_end() or n - 1)
        except Exception:
            end = n - 1
        start = max(0, start)
        end = min(n - 1, end)
        if end < start:
            start, end = 0, n - 1
        if start == 0 and end == n - 1:
            return gene_matrix
        return gene_matrix.iloc[start:end + 1]

    # ----- Results card ------------------------------------------------------

    def _results_content():
        r = resolved()
        if r is None:
            return ui.p(
                "Type a gene name or select a tissue to get started.",
                class_="text-muted mb-0",
            )
        if "error" in r:
            return ui.p(r["error"], class_="text-warning mb-0")

        ctx = cluster_ctx()
        cluster_label = cluster_names.get(ctx["node"], f"cluster_{ctx['node']}")
        ref_gene = ctx["ref_gene"]

        # (n/m) markers count from cluster label suffix e.g. "(33/35)"
        m = re.search(r"\((\d+/\d+)\)\s*$", cluster_label)
        markers = m.group(1) if m else None

        # Predicted annotation (WBbt link)
        annotation_label, annotation_url = None, None
        own = annotation_df[annotation_df["node"] == ctx["node"]]
        if not own.empty:
            annotation_label, annotation_url = gp.wormbase_anatomy_link(
                own["node_annotation"].iloc[0]
            )

        # Reference gene's mean expression + PCC in this cluster
        ref_mean, ref_pcc = None, None
        if ref_gene is not None and not own.empty:
            ref_row = own[own["gene_name"] == ref_gene]
            if not ref_row.empty:
                ref_mean = float(ref_row["mean_expression"].iloc[0])
                ref_pcc = float(ref_row["PCC"].iloc[0])

        def wb_button(url):
            return ui.tags.a(
                "View on WormBase ↗",
                href=url, target="_blank",
                class_="btn btn-sm btn-outline-primary ms-2",
            )

        def gene_wb_url(g):
            return f"https://wormbase.org/species/c_elegans/gene/{g}"

        def dash():
            return ui.span("—", class_="text-muted")

        def row(label, value):
            return ui.div(
                ui.span(f"{label}: ", class_="text-muted"),
                value,
                class_="mb-1 d-flex align-items-center",
            )

        # Top header (mode-dependent)
        if input.search_mode() == "Gene":
            # Always show the queried gene at the top. The current cluster's
            # ref gene is shown separately in the Cluster annotation section
            # ("gene cluster"), so we don't lose it.
            top = ui.div(
                ui.h4(ctx["gene"], class_="fw-bold mb-0 d-inline"),
                wb_button(gene_wb_url(ctx["gene"])),
                class_="d-flex align-items-center mb-3",
            )
        else:
            # In tissue mode, show the resolved cluster's predicted annotation
            # (which changes as you navigate), falling back to the search query.
            heading = annotation_label or (input.tissue_query() or "").strip()
            top_children = [ui.h4(heading, class_="fw-bold mb-0 d-inline")]
            if annotation_url:
                top_children.append(wb_button(annotation_url))
            top = ui.div(*top_children, class_="d-flex align-items-center mb-3")

        # Cluster annotation section
        gene_cluster_value = dash() if ref_gene is None else ui.TagList(
            ui.tags.code(ref_gene),
            wb_button(gene_wb_url(ref_gene)),
        )
        if annotation_url:
            pred_value = ui.TagList(ui.span(annotation_label), wb_button(annotation_url))
        elif annotation_label:
            pred_value = ui.span(annotation_label)
        else:
            pred_value = dash()

        cluster_section = ui.div(
            ui.h6("Cluster annotation",
                  class_="text-muted text-uppercase small mb-2"),
            row("gene cluster", gene_cluster_value),
            row("gene cluster expression",
                f"{ref_mean:.1f} RPM" if ref_mean is not None else dash()),
            row("gene cluster PCC",
                f"{ref_pcc:.2f}" if ref_pcc is not None else dash()),
            row("predicted annotation", pred_value),
            row("Tissue gene markers", f"({markers})" if markers else dash()),
            row("p-value", dash()),
            class_="mb-3",
        )

        return ui.TagList(top, cluster_section)


    @render.ui
    def results_header():
        return _results_content()


    def _nav_content():
        ctx = cluster_ctx()
        if ctx is None:
            return ui.p("(no cluster selected)", class_="text-muted mb-0")

        def short_label(nid):
            label = cluster_names.get(nid, f"cluster_{nid}")
            match = re.match(r"\[(\d+)\]\s*cluster\s+(\S+)", label)
            return f"[{match.group(1)}] {match.group(2)}" if match else f"[{nid}]"

        def dash():
            return ui.span("(none)", class_="text-muted")

        def nav_btn(button_id, arrow, node):
            """Active nav button, or a disabled span if the target is too large."""
            n = _cluster_size(node)
            label = f"{arrow} {short_label(node)} ({n} cells)"
            if n > MAX_NAV_CELLS:
                return ui.tags.span(
                    label,
                    class_="btn btn-sm btn-outline-secondary disabled",
                    title=f"Too large to render (limit: {MAX_NAV_CELLS} cells)",
                )
            return ui.input_action_button(
                button_id, label,
                class_="btn btn-sm btn-outline-primary",
            )

        current_node = ctx["node"]
        parents = list(tree.predecessors(current_node))
        children = list(tree.successors(current_node))

        parent_widgets = (
            [nav_btn("nav_parent", "⬆", parents[0])] if parents else [dash()]
        )
        current_widget = ui.tags.span(
            f"{short_label(current_node)} ({_cluster_size(current_node)} cells)",
            class_="btn btn-sm btn-primary",
            style="pointer-events: none;",
        )
        children_widgets = (
            [nav_btn(f"nav_child_{i}", "⬇", c) for i, c in enumerate(children[:2])]
            if children else [dash()]
        )

        def row(label, widgets):
            return ui.div(
                ui.div(f"{label}:", class_="text-muted mb-1"),
                ui.div(*widgets, class_="d-flex flex-wrap gap-2"),
                class_="mb-3",
            )

        return ui.TagList(
            ui.h3("Cluster navigation",
                  class_="text-primary text-uppercase fw-bold small mb-2"),
            row("Parental cluster", parent_widgets),
            row("Current cluster", [current_widget]),
            row("Children clusters", children_widgets),
        )

    @render.ui
    def nav_header():
        return _nav_content()

    # ----- Spectrum tab ------------------------------------------------------

    @reactive.calc
    def spectrum_inputs():
        ctx = cluster_ctx()
        if ctx is None:
            return None
        qualifying, gene_colors = gp.compute_high_expression_genes_in_cluster(
            gene_matrix=gene_matrix, tree=tree, cluster_node=ctx["node"],
            threshold_pct=input.threshold(), min_cells=1,
        )
        gene_color_map = dict(gene_colors)
        if ctx["gene"] not in gene_color_map:
            gene_color_map[ctx["gene"]] = (0.85, 0.1, 0.1, 1.0)
        genes_to_show = list(set(qualifying.index.tolist()) | {ctx["gene"]})
        return {
            "ctx": ctx,
            "gene_color_map": gene_color_map,
            "genes_to_show": genes_to_show,
        }

    @render.ui
    def spectrum_section():
        if cluster_ctx() is None:
            return None
        return output_widget("spectrum_plot")

    @render_widget
    def spectrum_plot():
        d = spectrum_inputs()
        if d is None:
            return None
        return plot_cell_content_plotly(
            node=d["ctx"]["node"], gene_matrix=gene_matrix, tree=tree,
            template=template, cluster_names=cluster_names,
            gene_color_map=d["gene_color_map"],
            genes_to_show=d["genes_to_show"],
        )

    # ----- Context tab: ref/query plots --------------------------------------

    @render.ui
    def context_section():
        ctx = cluster_ctx()
        if ctx is None:
            return None
        n_cells = len(ctx["cluster_cells"])
        n_total = len(gene_matrix.index)
        extra_slots = [
            ui.output_ui(f"extra_slot_{i}") for i in range(MAX_EXTRA_GENES)
        ]

        left_column = ui.div(
            ui.h4("Gene annotation table"),
            ui.input_switch(
                "include_subclusters",
                "Include sub-cluster annotations",
                value=False,
            ),
            ui.p(
                "Click 📈 Plot on a row to add that gene's expression plot "
                "on the right (click again to remove).",
                class_="text-muted small",
            ),
            # Free-text "add another gene" input for genes not in the table
            ui.div(
                ui.input_text(
                    "add_gene_query", label=None,
                    placeholder="Plot another gene by name (toggle if already plotted)...",
                    width="100%",
                ),
                ui.input_action_button(
                    "add_gene_btn", "Add",
                    class_="btn btn-sm btn-outline-primary",
                ),
                class_="d-flex gap-2 align-items-start mb-2",
            ),
            ui.output_ui("add_gene_error"),
            output_widget("gene_table", height="900px"),
        )

        right_column = ui.div(
            ui.h4("Expression across all cells"),
            # Single-line control: numeric range OR zoom-to-cluster shortcut.
            # A separate reset button below returns to the full range.
            ui.div(
                ui.span("Plot from cell", class_="text-muted me-2"),
                ui.input_numeric(
                    "range_start", label=None, value=0,
                    min=0, max=n_total - 1, step=1, width="110px",
                ),
                ui.span("to cell", class_="mx-2 text-muted"),
                ui.input_numeric(
                    "range_end", label=None, value=n_total - 1,
                    min=0, max=n_total - 1, step=1, width="110px",
                ),
                ui.span("or", class_="mx-2 text-muted"),
                ui.input_action_button(
                    "zoom_to_cluster_btn",
                    f"Zoom on cluster cells ({n_cells})",
                    class_="btn btn-sm btn-outline-primary",
                ),
                class_="d-flex align-items-center flex-wrap mb-2",
            ),
            ui.input_action_button(
                "range_reset", "Reset current view",
                class_="btn btn-sm btn-outline-secondary mb-2",
            ),
            ui.output_plot("annotation_strip_plot", height="120px"),
            ui.output_ui("ref_gene_section"),
            ui.output_ui("query_gene_label"),
            ui.output_plot("query_gene_plot", height="200px"),
            ui.output_ui("additional_genes_header"),
            *extra_slots,
        )

        collapsed = table_collapsed.get()
        toggle_label = "▶ Show table" if collapsed else "◀ Hide table"
        toggle_btn = ui.input_action_button(
            "toggle_table",
            toggle_label,
            class_="btn btn-sm btn-outline-secondary mb-2",
        )
        right_wrapped = ui.div(toggle_btn, right_column)

        if collapsed:
            # Full-width right column; Ag-Grid state persists in window globals
            # across the re-render on unfold.
            return right_wrapped

        return ui.layout_columns(
            left_column,
            right_wrapped,
            col_widths=(5, 7),
            gap="1rem",
        )

    @render.ui
    def ref_gene_label():
        ctx = cluster_ctx()
        if ctx is None or ctx["ref_gene"] is None:
            return None
        if ctx["ref_gene"] == ctx["gene"]:
            return ui.p(
                "Selected gene is the reference gene",
                class_="text-muted small mb-1 mt-2",
            )
        return ui.h6(
            "Reference gene",
            class_="text-muted text-uppercase small mb-1 mt-2",
        )


    @render.plot
    def annotation_strip_plot():
        if cluster_ctx() is None or not cell_annotation_colors:
            return None
        ctx_matrix = context_matrix()
        # Compute the offset = position of the first cell in the sliced
        # context_matrix within the full gene_matrix. That way the strip
        # labels show absolute cell indices (4000, 4128, 4200) even when
        # a custom range is applied.
        cells = ctx_matrix.index
        offset = 0
        if len(cells) > 0:
            first = cells[0]
            positions = gene_matrix.index.get_indexer([first])
            if len(positions) > 0 and positions[0] >= 0:
                offset = int(positions[0])
        return plot_annotation_strip_matplotlib(
            ctx_matrix, cell_annotation_colors, offset=offset,
        )

    @render.ui
    def add_gene_error():
        msg = add_gene_error_msg.get()
        if not msg:
            return None
        # Green tone if it's a friendly "removed X" confirmation
        cls = "text-success" if msg.startswith("Removed") else "text-danger"
        return ui.p(msg, class_=f"{cls} small mb-0 mt-1")


    @render.ui
    def ref_gene_section():
        ctx = cluster_ctx()
        if ctx is None or ctx["ref_gene"] is None:
            return None
        # When the selected gene IS the reference gene, only show the caption
        # (the plot is redundant with query_gene_plot below).
        if ctx["ref_gene"] == ctx["gene"]:
            return ui.p(
                "Selected gene is the reference gene",
                class_="text-muted small mb-1 mt-2",
            )
        return ui.TagList(
            ui.h6("Reference gene",
                  class_="text-muted text-uppercase small mb-1 mt-2"),
            ui.output_plot("ref_gene_pane", height="200px"),
        )

    @render.plot
    def ref_gene_pane():
        ctx = cluster_ctx()
        if (ctx is None or ctx["ref_gene"] is None
                or ctx["ref_gene"] == ctx["gene"]):
            return None
        return plot_gene_across_all_cells_matplotlib(
            ctx["ref_gene"], context_matrix(),
            highlight_cells=ctx["cluster_cells"],
            highlight_color=ctx["cluster_color"],
        )





    @render.plot
    def ref_gene_plot():
        ctx = cluster_ctx()
        if (ctx is None or ctx["ref_gene"] is None
                or ctx["ref_gene"] == ctx["gene"]):
            return None
        return plot_gene_across_all_cells_matplotlib(
            ctx["ref_gene"], context_matrix(),
            highlight_cells=ctx["cluster_cells"],
            highlight_color=ctx["cluster_color"],
        )

    @render.ui
    def query_gene_label():
        ctx = cluster_ctx()
        if ctx is None or ctx["ref_gene"] == ctx["gene"]:
            return None
        return ui.h6(
            "Selected gene",
            class_="text-muted text-uppercase small mb-1 mt-2",
        )

    @render.plot
    def query_gene_plot():
        ctx = cluster_ctx()
        if ctx is None:
            return None
        return plot_gene_across_all_cells_matplotlib(
            ctx["gene"], context_matrix(),
            highlight_cells=ctx["cluster_cells"],
            highlight_color=ctx["cluster_color"],
        )

    @render.ui
    def additional_genes_header():
        if not selected_extra_genes():
            return None
        return ui.div(
            ui.h6(
                "Additional gene(s)",
                class_="text-muted text-uppercase small mb-0 d-inline me-3",
            ),
            ui.input_action_button(
                "clear_extra_plots", "Clear all",
                class_="btn btn-sm btn-outline-danger",
            ),
            class_="mt-3 mb-2 d-flex align-items-center",
        )

    # ----- Context tab: annotation table + extra-gene plots ------------------

    @reactive.calc
    def gene_table_df():
        ctx = cluster_ctx()
        if ctx is None:
            return None
        node = ctx["node"]
        if input.include_subclusters():
            relevant = nx.descendants(tree, node) | {node}
            include_cluster_col = True
        else:
            relevant = {node}
            include_cluster_col = False

        # Column order specified by user:
        # gene / mean expression / PCC / known marker / annotated gene / previous annotations
        cols = ["gene_name", "mean_expression", "PCC",
                "used_for_annotation", "node_annotation",
                "gene_other_annotations"]
        labels = ["gene", "mean expression (RPM)", "PCC score",
                  "Known marker", "Predicted cell annotation",
                  "Prior gene annotations"]
        if include_cluster_col:
            cols.append("node")
            labels.append("Cluster")

        # Sort: cluster ascending (when present), then PCC descending.
        # Keeps rows for the same cluster grouped together in sub-cluster mode.
        raw = annotation_df[annotation_df["node"].isin(relevant)][cols]
        if include_cluster_col:
            raw = raw.sort_values(["node", "PCC"], ascending=[True, False])
        else:
            raw = raw.sort_values("PCC", ascending=False)
        rows = raw.reset_index(drop=True)
        rows.columns = labels
        # NOTE: no _plotted column here. The Plot-button state is tracked
        # client-side in window.HCC_plotted_genes so the grid doesn't have
        # to re-render (and lose sort/filter/column state) on every click.
        return rows

    @render_widget
    def gene_table():
        df = gene_table_df()
        if df is None or df.empty:
            return None

        n = len(df)
        paginate = n > 100

        plot_button_fn = (
            "function(params) {"
            "  if (!window.HCC_plotted_genes) window.HCC_plotted_genes = new Set();"
            "  var active = window.HCC_plotted_genes.has(params.data.gene);"
            "  var btn = document.createElement('button');"
            "  btn.innerText = active ? '✓ Plotted' : '📈 Plot';"
            "  btn.className = active ? 'btn btn-sm btn-primary' "
            "                         : 'btn btn-sm btn-outline-primary';"
            "  btn.style.padding = '0 8px';"
            "  btn.onclick = function(e) {"
            "    e.stopPropagation();"
            "    var g = params.data.gene;"
            "    if (window.HCC_plotted_genes.has(g)) {"
            "      window.HCC_plotted_genes.delete(g);"
            "    } else {"
            "      window.HCC_plotted_genes.add(g);"
            "    }"
            "    if (params.api.redrawRows) {"
            "      params.api.redrawRows({rowNodes: [params.node]});"
            "    }"
            "    var s = window.Shiny || (window.parent && window.parent.Shiny);"
            "    if (s && s.setInputValue) {"
            "      s.setInputValue('plot_gene_click',"
            "        {gene: g, nonce: Math.random()},"
            "        {priority: 'event'});"
            "    }"
            "  };"
            "  return btn;"
            "}"
        )

        row_style_fn = (
            "function(params) {"
            "  if (params.data && window.HCC_plotted_genes && "
            "      window.HCC_plotted_genes.has(params.data.gene)) {"
            "    return {backgroundColor: '#e7f1ff'};"
            "  }"
            "  return null;"
            "}"
        )

        # ---- Client-side grid state preservation --------------------------
        # Because gene_table_df changes every time plotted_genes changes (the
        # _plotted column recomputes), @render_widget destroys and recreates
        # the whole Ag-Grid widget on every Plot click. Without this, the
        # user's sort / filter / column-order / current-page state gets wiped
        # on every click.
        #
        # We stash the grid's state in a `window` global on every state-change
        # event, then restore it in `onFirstDataRendered` on the freshly-built
        # grid. State survives across re-renders in the same browser session.
        # Ag-Grid ignores unknown column IDs in applyColumnState, so toggling
        # sub-clusters (which changes the column set) degrades gracefully.

        save_state_js = (
            "function(params) {"
            "  window.HCC_grid_state = {"
            "    columnState: params.api.getColumnState(),"
            "    filterModel: params.api.getFilterModel(),"
            "    currentPage: params.api.paginationGetCurrentPage ? "
            "                 params.api.paginationGetCurrentPage() : null"
            "  };"
            "}"
        )

        restore_state_js = (
            "function(params) {"
            "  window.HCC_grid_api = params.api;"
            "  var s = window.HCC_grid_state;"
            "  if (!s) return;"
            "  if (s.columnState) {"
            "    params.api.applyColumnState({"
            "      state: s.columnState, applyOrder: true"
            "    });"
            "  }"
            "  if (s.filterModel) {"
            "    params.api.setFilterModel(s.filterModel);"
            "  }"
            "  if (typeof s.currentPage === 'number' && "
            "      params.api.paginationGoToPage) {"
            "    params.api.paginationGoToPage(s.currentPage);"
            "  }"
            "}"
        )

        column_defs = [
            {
                "headerName": "Plot",
                "cellRenderer": plot_button_fn,
                "width": 90,
                "minWidth": 90,
                "maxWidth": 90,
                "suppressSizeToFit": True,
                "sortable": False,
                "filter": False,
                "suppressMovable": True,
                "pinned": "left",
            },
            {
                "field": "gene", "width": 110,
                "minWidth": 110, "suppressSizeToFit": True,
                "pinned": "left",
                "suppressMovable": True,
            },
            {
                "field": "mean expression (RPM)",
                "width": 160,
                "minWidth": 160, "suppressSizeToFit": True,
                "valueFormatter":
                    "function(p){return p.value == null ? '' : "
                    "Number(p.value).toFixed(1);}",
            },
            {
                "field": "PCC score",
                "width": 110,
                "minWidth": 110, "suppressSizeToFit": True,
                "valueFormatter":
                    "function(p){return p.value == null ? '' : "
                    "Number(p.value).toFixed(2);}",
            },
            {
                "field": "Known marker",
                "width": 130,
                "minWidth": 130, "suppressSizeToFit": True,
                "valueFormatter":
                    "function(p){var v=p.value; "
                    "return (v===true||v==='True'||v==='true') ? '✓' : '—';}",
                "cellStyle":
                    "function(p){var v=p.value; "
                    "if(v===true||v==='True'||v==='true') "
                    "return {backgroundColor:'#d4edda', color:'#155724', "
                    "fontWeight:'bold', textAlign:'center'}; "
                    "return {textAlign:'center', color:'#999'};}",
            },
            {"field": "Predicted cell annotation", "flex": 1, "minWidth": 180},
            {"field": "Prior gene annotations", "flex": 1, "minWidth": 180},
        ]
        if "Cluster" in df.columns:
            column_defs.append({
                "field": "Cluster", "width": 100,
                "minWidth": 100, "suppressSizeToFit": True,
            })

        grid_options = {
            "columnDefs": column_defs,
            "defaultColDef": {
                "sortable": True,
                "filter": True,
                "resizable": True,
                # column reordering is enabled by default; Plot column has
                # suppressMovable=True so users can't drag it away from the left
            },
            "animateRows": True,
            "pagination": paginate,
            "paginationPageSize": 100 if paginate else None,
            "rowHeight": 34,
            "getRowStyle": row_style_fn,
            # State preservation across re-renders
            "onFirstDataRendered": restore_state_js,
            "onSortChanged": save_state_js,
            "onFilterChanged": save_state_js,
            "onColumnMoved": save_state_js,
            "onColumnResized": save_state_js,
            "onColumnPinned": save_state_js,
            "onColumnVisible": save_state_js,
            "onPaginationChanged": save_state_js,
        }

        return Grid(
            grid_data=df,
            grid_options=grid_options,
            theme="ag-theme-balham",
            columns_fit="size_to_fit",
            index=False,
            quick_filter=False,
            export_csv=False,
            export_excel=False,
            menu={"buttons": []},
            height=900,
        )

    @reactive.calc
    def selected_extra_genes():
        """Genes currently toggled on via the table's Plot buttons.
        Excludes the queried gene and the ref gene since those have their
        own dedicated plots above.
        """
        ctx = cluster_ctx()
        if ctx is None:
            return []
        cols = context_matrix().columns
        return [g for g in plotted_genes.get()
                if g not in (ctx["gene"], ctx["ref_gene"]) and g in cols]

    def _register_extra_slot(i: int):
        @output(id=f"extra_slot_{i}")
        @render.ui
        def _slot_ui():
            genes = selected_extra_genes()
            if i >= len(genes):
                return None
            return ui.output_plot(f"extra_plot_{i}", height="200px")

        @output(id=f"extra_plot_{i}")
        @render.plot
        def _slot_plot():
            ctx = cluster_ctx()
            genes = selected_extra_genes()
            if ctx is None or i >= len(genes):
                return None
            return plot_gene_across_all_cells_matplotlib(
                genes[i], context_matrix(),
                highlight_cells=ctx["cluster_cells"],
                highlight_color=ctx["cluster_color"],
            )

    for i in range(MAX_EXTRA_GENES):
        _register_extra_slot(i)


app = App(app_ui, server)