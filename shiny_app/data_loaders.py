"""
Process-scoped data loaders for the HCC Explorer Shiny prototype.

Replaces the Streamlit version's @st.cache_resource with functools.lru_cache
on nullary loaders. Same practical effect for these read-once startup
fixtures: the file is parsed once per Python process and the resulting
object is shared thereafter.

Why nullary + lru_cache and not @reactive.calc:

- @reactive.calc is *session-scoped* in Shiny (recomputed per user
  session). Fine for cheap derivations, wrong for a ~570MB gene matrix.
- lru_cache(maxsize=None) at module scope is process-scoped -- the
  matrix is loaded exactly once at startup and every session shares it,
  matching what @st.cache_resource was actually doing here.

Compared to the Streamlit loaders, `load_cell_totals`, `load_gene_cluster_index`,
`load_tissue_index` are made nullary: they reach in and call the other loaders
themselves, so the caller doesn't have to plumb the dependency by hand.
"""

import os
import ast
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import gene_plots as gp

# The Shiny app lives in shiny_app/ and shares the repo-root data/ folder
# with the Streamlit app.
DATA_DIR = Path(os.environ.get("GENE_APP_DATA_DIR", Path(__file__).parent.parent / "data"))

MATRIX_REINDEXED_PATH = DATA_DIR / "HCC_gene_count_matrix_rpm_reindexed_unsupervised.tsv"
TREE_PATH             = DATA_DIR / "HCC_graph_pruned.pickle"
SPECTRUM_PATH         = DATA_DIR / "HCC_cluster_expression_spectrum.pkl"
NAMES_PATH            = DATA_DIR / "cluster_node_names_tea.pkl"
TRAVERSAL_PATH        = DATA_DIR / "HCC_node_traversal_order.pkl"
CELL_LINKAGE_PATH     = DATA_DIR / "HCC_cell_linkage.csv"
ANNOTATIONS_DIR       = DATA_DIR / "node_annotations"
COLOR_TABLE_PATH      = DATA_DIR / "genes_and_clusters_assigned_colors.tsv"



METADATA_PATH = DATA_DIR / "HCC_metadata_annotations.tsv"


@lru_cache(maxsize=None)
def load_cell_annotation_colors():
    """{cell_id: color} for the per-cell annotation strip.

    Reads the `selected` column, which stores RGBA tuple strings such as
    `"(0.0, 0.667, 0.572, 1.0)"`. Parses them into real tuples so the strip
    plotter can pass them straight to matplotlib's `facecolor=`. Falls back
    to `experiment_color` (hex) if `selected` is missing.
    """
    if not METADATA_PATH.exists():
        return {}
    df = pd.read_csv(METADATA_PATH, sep="\t", index_col=0)

    if "selected" in df.columns:
        def _parse(s):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
        parsed = df["selected"].map(_parse)
        return {cell: color for cell, color in parsed.items() if color is not None}

    if "experiment_color" in df.columns:
        return df["experiment_color"].to_dict()

    return {}



@lru_cache(maxsize=None)
def load_tree():
    with open(TREE_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def load_spectrum():
    with open(SPECTRUM_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def load_cluster_names():
    with open(NAMES_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def load_traversal_order():
    with open(TRAVERSAL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def load_cell_linkage():
    return np.loadtxt(CELL_LINKAGE_PATH, delimiter=",")


@lru_cache(maxsize=None)
def load_gene_matrix():
    # Same rationale as the Streamlit version: parse ONCE, cast float32
    # AFTER parsing (not via read_csv dtype=, which would try to cast the
    # barcode-string index column before index_col takes effect).
    df = pd.read_csv(MATRIX_REINDEXED_PATH, sep="\t", index_col=0)
    return df.astype(np.float32)


@lru_cache(maxsize=None)
def load_cluster_colors():
    return gp.load_cluster_color_table(COLOR_TABLE_PATH)


@lru_cache(maxsize=None)
def load_cell_totals():
    return gp.compute_cell_totals(load_gene_matrix())


@lru_cache(maxsize=None)
def load_gene_cluster_index():
    return gp.build_gene_cluster_index(load_spectrum())


@lru_cache(maxsize=None)
def load_tissue_index():
    return gp.build_tissue_index(load_cluster_names(), load_spectrum())


@lru_cache(maxsize=None)
def load_annotation_data():
    df = gp.load_node_annotation_tables(ANNOTATIONS_DIR)
    index = gp.build_annotation_index(df)
    return df, index


def matrix_exists() -> bool:
    return MATRIX_REINDEXED_PATH.exists()
