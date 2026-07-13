"""
Shared, cached data loading for the multi-page app.

Both pages import from here rather than defining their own @st.cache_resource
loaders. This matters for correctness, not just tidiness: Streamlit's
cache_resource keys on the function object itself, so two independently
defined (even identical) `load_tree()` functions in two different page files
would each get their own cache entry -- silently loading and holding the
gene matrix / tree in memory twice. Importing the same function object from
here means both pages share one cached copy per server process.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import gene_plots as gp

DATA_DIR = Path(os.environ.get("GENE_APP_DATA_DIR", Path(__file__).parent / "data"))
MATRIX_PATH = DATA_DIR / "HCC_gene_count_matrix_rpm.tsv"
MATRIX_REINDEXED_PATH = DATA_DIR / "HCC_gene_count_matrix_rpm_reindexed_unsupervised.tsv"
TREE_PATH = DATA_DIR / "HCC_graph_pruned.pickle"
SPECTRUM_PATH = DATA_DIR / "HCC_cluster_expression_spectrum.pkl"
NAMES_PATH = DATA_DIR / "cluster_node_names_tea.pkl"
TRAVERSAL_PATH = DATA_DIR / "HCC_node_traversal_order.pkl"
CELL_LINKAGE_PATH = DATA_DIR / "HCC_cell_linkage.csv"
ANNOTATIONS_DIR = DATA_DIR / "node_annotations"
COLOR_TABLE_PATH = DATA_DIR / "genes_and_clusters_assigned_colors.tsv"

@st.cache_resource(show_spinner="Loading cluster tree...")
def load_tree():
    with open(TREE_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading cluster expression template...")
def load_spectrum():
    with open(SPECTRUM_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading cluster names...")
def load_cluster_names():
    with open(NAMES_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading node traversal order...")
def load_traversal_order():
    with open(TRAVERSAL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading cell linkage matrix...")
def load_cell_linkage():
    return np.loadtxt(CELL_LINKAGE_PATH, delimiter=',')


@st.cache_resource(show_spinner="Loading gene expression matrix (this can take a while)...")
def load_gene_matrix():
    gene_matrix = pd.read_csv(MATRIX_PATH, sep="\t", index_col=0)
    gene_matrix_reindexed = pd.read_csv(MATRIX_REINDEXED_PATH, sep="\t", index_col=0)
    return gene_matrix, gene_matrix_reindexed




@st.cache_resource(show_spinner="Loading cluster color table...")
def load_cluster_colors():
    return gp.load_cluster_color_table(COLOR_TABLE_PATH)

@st.cache_resource(show_spinner="Precomputing per-cell totals...")
def load_cell_totals(_gene_matrix):
    # leading underscore tells st.cache_resource not to hash the (large) df
    return gp.compute_cell_totals(_gene_matrix)


@st.cache_resource(show_spinner="Indexing genes to clusters...")
def load_gene_cluster_index(_template):
    return gp.build_gene_cluster_index(_template)


@st.cache_resource(show_spinner="Indexing tissue names to clusters...")
def load_tissue_index(_cluster_names, _template):
    return gp.build_tissue_index(_cluster_names, _template)


@st.cache_resource(show_spinner="Loading precomputed PCC / prior-annotation tables...")
def load_annotation_data():
    df = gp.load_node_annotation_tables(ANNOTATIONS_DIR)
    index = gp.build_annotation_index(df)
    return df, index


def matrix_exists():
    return MATRIX_PATH.exists()