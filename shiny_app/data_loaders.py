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
from matplotlib.colors import to_rgba

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

# Per-cell colors for the annotation strip. The annotations file (with a
# `selected` RGBA column) is preferred; the reference metadata already in the
# repo (with an `experiment_color` hex column) is the fallback.
METADATA_PATHS = [
    DATA_DIR / "HCC_metadata_annotations.tsv",
    DATA_DIR / "HCC_metadata_reference.tsv",
]


def _rgba_key(color):
    """Hashable, rounding-tolerant key for comparing colors in any format."""
    return tuple(round(c, 3) for c in to_rgba(color))


def _looks_like_color(value):
    return isinstance(value, str) and value.strip().startswith(("#", "("))


@lru_cache(maxsize=None)
def _load_cell_metadata():
    """({cell_id: color}, {rgba_key: label}) for the per-cell annotation strip.

    Colors come from the `selected` column (RGBA tuple strings such as
    `"(0.0, 0.667, 0.572, 1.0)"`, parsed into real tuples) or, failing that,
    `experiment_color` (hex). Labels come from the first text column that
    gives each color a single label (e.g. experiment_type for
    experiment_color; one label may span several colors). The label map is
    empty if there is no such column.
    """
    path = next((p for p in METADATA_PATHS if p.exists()), None)
    if path is None:
        return {}, {}
    df = pd.read_csv(path, sep="\t", index_col=0)

    if "selected" in df.columns:
        def _parse(s):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
        colors = df["selected"].map(_parse)
    elif "experiment_color" in df.columns:
        colors = df["experiment_color"]
    else:
        return {}, {}
    colors = colors[colors.notna()]
    if colors.empty:
        return {}, {}

    keys = colors.map(_rgba_key)
    labels = {}
    for col in df.columns:
        values = df.loc[colors.index, col]
        if not pd.api.types.is_string_dtype(values) or values.isna().any() \
                or values.map(_looks_like_color).any():
            continue
        pairs = pd.DataFrame({"key": keys, "label": values})
        if (pairs.groupby("key")["label"].nunique() == 1).all():
            labels = pairs.drop_duplicates("key").set_index("key")["label"].to_dict()
            break
    return colors.to_dict(), labels


def load_cell_annotation_colors():
    """{cell_id: color} for the per-cell annotation strip."""
    return _load_cell_metadata()[0]


def load_cell_annotation_labels():
    """{rgba_key: label} naming each strip color (empty if unknown).
    Look colors up with `rgba_key(color)`."""
    return _load_cell_metadata()[1]


rgba_key = _rgba_key


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


def matrix_is_lfs_pointer() -> bool:
    """True when the matrix file is a Git LFS pointer, not the real data
    (the repo was cloned without `git lfs pull`)."""
    if not MATRIX_REINDEXED_PATH.exists() or MATRIX_REINDEXED_PATH.stat().st_size > 1024:
        return False
    with open(MATRIX_REINDEXED_PATH, "rb") as f:
        return f.read(64).startswith(b"version https://git-lfs")


def matrix_exists() -> bool:
    return MATRIX_REINDEXED_PATH.exists() and not matrix_is_lfs_pointer()
