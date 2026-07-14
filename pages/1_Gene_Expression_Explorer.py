"""
Gene Expression Explorer
------------------------
A public-facing search tool. Type either:
  - a gene of interest, or
  - a tissue / cell-type name (e.g. "excretory gland cell", "coelomic system")

...and get back:
  1. The per-cell expression composition plot for the relevant cluster, and
  2. The gene's expression across every cell in the dataset, in raw matrix
     order, for context.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import gene_plots as gp
import data_loaders as dl

st.set_page_config(page_title="Gene Expression Explorer", layout="wide")


# ---------------------------------------------------------------------------
# Gene table + additional-gene plots, scoped as its own fragment: toggling a
# checkbox here reruns only this section, not the whole page -- so the
# (expensive) spectrum and context plots above aren't rebuilt from scratch
# every time someone selects an extra gene.
# ---------------------------------------------------------------------------

@st.fragment
def render_gene_table_and_extra_plots(node, gene_to_display, ref_gene, tree, annotation_df,
                                       context_matrix, cluster_cells, cluster_color):

    # Reserve a spot for the additional-gene plots FIRST -- its position
    # (right under the main reference/queried gene plots above this
    # fragment) is fixed here, even though nothing gets filled into it
    # until after the toggle/table below are processed. Created fresh on
    # every fragment rerun (not shared across reruns), so there's no risk
    # of stale/duplicate plots accumulating across toggles.
    extra_plots_slot = st.container()

    st.markdown(''' #### :material/select_check_box: Add more genes from this cluster's annotation table:''')

    include_subclusters = st.toggle(
        "Include sub-cluster annotations",
        value=False,
        help=(
            "Also show curated genes from this cluster's descendant "
            "sub-clusters, not just this cluster's own annotation rows. "
            "Any of them can be plotted the same way."
        ),
    )

    if include_subclusters:
        relevant_nodes = nx.descendants(tree, node) | {node}
        columns = ["gene_name", "node", "node_annotation", "gene_other_annotations",
                   "mean_expression", "PCC", "used_for_annotation"]
        column_labels = ["gene", "Cluster", "Predicted cell annotation", "Prior gene annotations",
                          "mean expression (RPM)", "PCC score", "Known marker"]
    else:
        relevant_nodes = {node}
        columns = ["gene_name", "node_annotation", "gene_other_annotations",
                   "mean_expression", "PCC", "used_for_annotation"]
        column_labels = ["gene", "Predicted cell annotation", "Prior gene annotations",
                          "mean expression (RPM)", "PCC score", "Known marker"]

    cluster_rows = (
        annotation_df[annotation_df["node"].isin(relevant_nodes)]
        [columns]
        .sort_values("mean_expression", ascending=False)
        .reset_index(drop=True)
    )
    cluster_rows.columns = column_labels

    if cluster_rows.empty:
        st.info("No curated genes found for this cluster (or its sub-clusters).")
        return

    st.markdown('''Select additional genes of interest to output their expression across cells''')

    picker_df = cluster_rows.copy()
    picker_df.insert(0, "Plot", False)

    edited = st.data_editor(
        picker_df,
        hide_index=True,
        disabled=[c for c in picker_df.columns if c != "Plot"],
        key=f"annotation_picker_{node}_{include_subclusters}",
        width='stretch',
    )

    extra_genes = [
        g for g in edited.loc[edited["Plot"], "gene"].tolist()
        if g not in (gene_to_display, ref_gene) and g in context_matrix.columns
    ]

    with extra_plots_slot:
        for extra_gene in extra_genes:
            st.markdown(f'''➕ **Additional gene: :grey-background[{extra_gene}]**''')

            fig_extra = gp.plot_gene_across_all_cells(
                extra_gene, context_matrix,
                highlight_cells=cluster_cells, highlight_color=cluster_color,
            )
            st.pyplot(fig_extra, width='stretch', dpi=300, bbox_inches=None)
            plt.close(fig_extra)


# ---------------------------------------------------------------------------
# Rendering: shared by both the gene-search and tissue-search paths
# ---------------------------------------------------------------------------

def render_results(gene_to_display, node, match_label, gene_matrix_reindexed, tree, template, cluster_names, threshold, association=None):

    cluster_label = cluster_names.get(node, f"cluster_{node}")

    st.success(f"Showing **{gene_to_display}** in **{cluster_label}** ({match_label})", width=750)

    if association is not None:
        status = association["status"]
        precomputed = association.get("precomputed", False)

        if status == "established":
            pass
        elif status == "not_found":
            st.caption(
                f"ℹ️ No curated PCC / prior-annotation data found for **{gene_to_display}** "
                f"in this cluster yet."
            )
        else:
            pass

    cluster_cells = gp.get_cluster_node_cell_ids(tree=tree, node=node)
    cluster_color = cluster_colors.get(node, gp.cluster_color_for_node(node))
    ref_gene, ref_mean = gp.reference_gene_for_node(node, annotation_df)

    # This cluster's own predicted cell annotation is the same for every
    # gene row in the table (it's a property of the node, not the gene) --
    # surface it once here as a clickable WormBase link, independent of
    # whatever the gene table's sub-cluster toggle is currently showing.
    own_node_rows = annotation_df[annotation_df["node"] == node]
    if not own_node_rows.empty:
        node_annotation_raw = own_node_rows["node_annotation"].iloc[0]
        label, url = gp.wormbase_anatomy_link(node_annotation_raw)
        if url:
            st.markdown(f"🔬 **Predicted cell annotation:** [{label} ↗]({url})")

    # --- Section 1: cell composition spectrum, full width ---
    st.markdown(''' #### :material/immunology: Cells associated with this gene''')

    with st.spinner("Rendering cluster plot..."):
        qualifying, gene_colors = gp.compute_high_expression_genes_in_cluster(
            gene_matrix=gene_matrix_reindexed, tree=tree, cluster_node=node,
            threshold_pct=threshold, min_cells=1,
        )
        gene_color_map = dict(gene_colors)
        if gene_to_display not in gene_color_map:
            gene_color_map[gene_to_display] = (0.85, 0.1, 0.1, 1.0)  # force a visible color

        genes_to_show = list(set(qualifying.index.tolist()) | {gene_to_display})

        # Full legend: every gene actually drawn on the plot, not just
        # each cell's top-3-expressed + the queried gene.
        legend_genes = genes_to_show

        fig = gp.plot_cell_content(
            node=node, gene_matrix=gene_matrix_reindexed, tree=tree, template=template,
            cluster_names=cluster_names, strategy="overlay_gene_list",
            threshold=0.1, gene_color_map=gene_color_map,
            genes_to_show=genes_to_show, legend_genes=legend_genes, bg_alpha=0,
        )
        if fig is not None:
            st.pyplot(fig, width='stretch', dpi=500)
            plt.close(fig)
        else:
            st.info("No cell data available for this cluster.")

    # --- Section 2: gene expression context, full width ---
    with st.spinner("Rendering context plot..."):

        st.markdown(''' #### :material/genetics: Gene expression across all cells (context)''')

        zoom_to_cluster = st.toggle(
            f"🔍 Zoom to this cluster's {len(cluster_cells)} cells only",
            value=False,
            help=(
                "Off: plot spans every cell in the dataset, with this cluster's "
                "cells highlighted as a segment. On: plot spans only this "
                "cluster's own cells, for a closer look at variation within it."
            ),
        )
        context_matrix = gene_matrix_reindexed.loc[cluster_cells] if zoom_to_cluster else gene_matrix_reindexed

        # --- Reference gene ---
        if ref_gene is not None and ref_gene != gene_to_display:

            st.markdown(f'''
                📌 **Reference gene: :red-background[{ref_gene}]**
                        ''')

            fig_ref = gp.plot_gene_across_all_cells(
                ref_gene, context_matrix,
                highlight_cells=cluster_cells,
                highlight_color=cluster_color,
            )

            st.pyplot(fig_ref, width='stretch', dpi=300, bbox_inches=None)
            plt.close(fig_ref)

        elif ref_gene == gene_to_display:
            st.markdown(f'''📌 **Selected gene is the reference gene: :red-background[{ref_gene}]**''')

        # --- Queried gene label (independent of the branch above --
        # needs to run whenever the queried gene isn't the reference gene,
        # not just in the "no annotation rows" leftover case an elif would
        # have restricted it to) ---
        if ref_gene != gene_to_display:
            st.markdown(f'''🔬 **Selected gene: :blue-background[{gene_to_display}]**''')

        # --- Queried gene ---
        fig2 = gp.plot_gene_across_all_cells(
            gene_to_display, context_matrix,
            highlight_cells=cluster_cells, highlight_color=cluster_color,
        )
        st.pyplot(fig2, width='stretch', dpi=300, bbox_inches=None)
        plt.close(fig2)

    # --- Section 3: gene table + additional gene plots, full width,
    # scoped as its own fragment (see function above) ---
    render_gene_table_and_extra_plots(
        node, gene_to_display, ref_gene, tree, annotation_df,
        context_matrix, cluster_cells, cluster_color,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title(":material/biotech: Gene Expression Explorer")
st.write(
    "Search for a **gene** or pick a **tissue / cell-type** to see the cells "
    "it's associated with, and how expression compares across the full dataset."
)

if not dl.matrix_exists():
    st.error(
        f"Gene expression matrix not found at `{dl.MATRIX_REINDEXED_PATH}`.\n\n"
        "Place your `HCC_gene_count_matrix_rpm_reindexed_unsupervised.tsv` file in the "
        "`data/` folder next to this app (or set the `GENE_APP_DATA_DIR` environment "
        "variable to point at a folder containing it) and reload."
    )
    st.stop()

tree = dl.load_tree()
template = dl.load_spectrum()
cluster_names = dl.load_cluster_names()
gene_matrix_reindexed = dl.load_gene_matrix()
cell_totals = dl.load_cell_totals(gene_matrix_reindexed)
gene_cluster_index = dl.load_gene_cluster_index(template)
tissue_index = dl.load_tissue_index(cluster_names, template)
annotation_df, annotation_index = dl.load_annotation_data()
cluster_colors = dl.load_cluster_colors()


search_mode = st.radio("**Search by:**", ["Gene", "Tissue"], horizontal=True)
threshold = st.slider(
    "Select threshold", 0.1, 2.0, 2.0, step=0.1, width=200,
    help=(
        "Minimum percentage of a cell's total expression a gene must represent "
        "to be highlighted (colored) in the plot on the left. Lower = more genes "
        "shown per cell; higher = only the most dominant gene(s) per cell. "
        "See the About & Glossary page for more detail."
    ),
)


if search_mode == "Gene":
    query = st.text_input("**Gene name:**", placeholder="e.g. nspc-20 or F40F8.4",width=500).strip()
else:
    tissue_options = gp.all_tissue_names(tissue_index)
    if not tissue_options:
        st.warning("No tissue/cell-type names were found in this dataset.")
        query = None
    else:
        query = st.selectbox(
            "**Tissue / cell-type:**", tissue_options,
            index=None, placeholder="Select a tissue…",width=750
        )

if search_mode == "Gene" and query:
    if query in gene_matrix_reindexed.columns:
        with st.spinner(f"Finding the cluster where {query} appears..."):
            flared_node = gp.find_flared_cluster_for_gene(query, template)
            if flared_node is not None:
                node = flared_node
                source = "flared"
                match_info = "most expressed gene specific to this cell group"
            else:
                node, info = gp.find_best_cluster_for_gene(
                    query, gene_matrix_reindexed, tree, cluster_names, gene_cluster_index,
                    cell_totals=cell_totals,
                )
                if node is None:
                    st.warning(f"Couldn't find any cluster associated with '{query}'.")
                    st.stop()
                source = info["source"]  # "template" or "expression_scan"
                if source == "template":
                    n = info["n_clusters_in_template"]
                    if n == 1:
                        match_info = "the group of cells that this gene is most specific to"
                    else:
                        match_info = (
                            f"appears in {n} clusters in the reference groupings; "
                            f"shown here is the one with highest expression "
                            f"(mean {info['mean_pct']:.2f}% of {info['n_cells']} cells)"
                        )
                else:
                    match_info = (
                        f"gene not in the reference groupings -- fell back to scanning "
                        f"all named clusters by expression "
                        f"(mean {info['mean_pct']:.2f}% / max {info['max_pct']:.2f}% "
                        f"of {info['n_cells']} cells)"
                    )

            association = gp.association_quality(query, node, source,
                                                    annotation_index=annotation_index)

        render_results(query, node, match_info, gene_matrix_reindexed, tree, template, cluster_names,
                       threshold, association=association)
        st.page_link("pages/2_About.py", label="What do these terms mean? See the About & Glossary page.",
                     icon=":material/menu_book:")
    else:
        st.warning(f"'{query}' wasn't found as a gene in the expression matrix.")

elif search_mode == "Tissue" and query:
        matches = gp.find_tissue_matches(query, tissue_index)

        if not matches:
            st.warning(f"No clusters matched tissue '{query}'.")
            st.stop()

        def cluster_size(node):
            return len(gp.get_cluster_node_cell_ids(tree=tree, node=node))

        # One entry per matching cluster (deduplicated by node), labeled
        # with the annotation table's highest-expressed gene for that node
        # -- not whatever gene the cluster happens to be historically named
        # after in cluster_names.pkl (those can go stale, see cpz-1/ttr-50).
        # Clusters with no annotation-table rows are skipped entirely.
        seen_nodes = set()
        cluster_entries = []  # (label, node, tissue, gene)
        for tissue, node, flared_genes in matches:
            if node in seen_nodes:
                continue
            ref_gene, ref_mean = gp.reference_gene_for_node(node, annotation_df)
            if ref_gene is None:
                continue
            seen_nodes.add(node)

            cluster_label = cluster_names.get(node, f"cluster_{node}")
            # cluster_label is typically "gene – tissue (n/m)"; keep the
            # tissue/count suffix but swap in the annotation table's gene.
            parts = cluster_label.split("–", 1)
            suffix = parts[1].strip() if len(parts) == 2 else cluster_label

            label = f"[{node}] cluster {ref_gene} – {suffix}"
            cluster_entries.append((label, node, tissue, ref_gene))

        if not cluster_entries:
            st.warning(
                f"'{query}' matched a tissue name, but none of the matching clusters "
                "have curated genes in the annotation table.", width=750
            )
            st.stop()

        cluster_entries.sort(
            key=lambda e: (query.strip().lower() != e[2], -cluster_size(e[1]), e[1])
        )

        if len(cluster_entries) == 1:
            _, node, tissue, gene = cluster_entries[0]
        else:
            st.info(f"'{query}' matched {len(cluster_entries)} clusters. Pick one:", width=750)
            cluster_labels = [e[0] for e in cluster_entries]
            choice = st.selectbox("**Choose a cluster:**", cluster_labels, width=750)
            _, node, tissue, gene = next(e for e in cluster_entries if e[0] == choice)

        flared_genes_for_node = gp.get_flared_genes_for_cluster(node, template)
        if gene in flared_genes_for_node:
            match_info = f"tissue match for '{query}' → {tissue}"
            assoc_source = "flared"
        else:
            match_info = f"tissue match for '{query}' → {tissue}"
            assoc_source = "curated_table"

        association = gp.association_quality(gene, node, assoc_source, annotation_index=annotation_index)
        render_results(gene, node, match_info, gene_matrix_reindexed, tree, template, cluster_names,
                       threshold, association=association)
        st.page_link("pages/2_About.py", label="What do these terms mean? See the About & Glossary page.",
                     icon=":material/menu_book:")
else:
    st.info("Type a gene name or select a tissue to get started.", width=750)