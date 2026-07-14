import streamlit as st

st.title(":material/menu_book: About & Glossary")

st.write(
    """
    This page explains the terms used throughout the Gene Expression Explorer,
    for readers who haven't read the accompanying paper (or just want a refresher).
    """
)

st.markdown("### Cell composition spectrum (left plot)")
st.write(
    """
    Each vertical bar represents one cell. The colored segments show which
    genes make up most of that cell's measured expression, expressed as a
    percentage of the cell's total.

    **Threshold** controls which genes are colored in this plot: a gene is
    only highlighted in a cell if it represents at least this percentage of
    that cell's total expression. Lower threshold → more genes shown (more
    colors, busier plot). Higher threshold → only the most dominant gene(s)
    per cell are shown.
    """
)

st.markdown("### Gene expression across all cells (context, right plot)")
st.write(
    """
    Shows one gene's expression (in RPM, reads per million) across every
    cell in the dataset, in a fixed cell order shared by the whole app. The
    highlighted segment marks the cells belonging to the cluster/cell-group
    shown on the left, so you can see how specific (or not) that gene's
    expression is to that group versus the rest of the dataset.

    **Reference gene**: the single most highly expressed gene, among this
    group's curated candidate genes, for this specific cell group. It's
    shown alongside your searched gene as a point of comparison.
    """
)

st.markdown("### Gene annotation table")
st.write(
    """
    - **gene**: the gene's name.
    - **Predicted cell annotation**: the cell-type/tissue label this group of
      cells is predicted to correspond to, based on this dataset's own
      clustering and marker-gene analysis. Links to the matching WormBase
      anatomy term where available.
    - **Prior gene annotations**: other cell-types/tissues this *specific gene*
      has previously been reported to be expressed in, according to prior
      WormBase curation -- independent of this dataset's own clustering.
      Useful for cross-checking whether the predicted cell annotation above
      agrees with what was already known about this gene.
    - **mean expression (RPM)**: this gene's average expression level within
      this specific group of cells.
    - **PCC score**: Pearson correlation coefficient between this gene's
      expression pattern and the group's defining/reference gene pattern --
      a measure of how similarly they're expressed across cells. Closer to 1
      means more similar.
    - **Known marker**: whether this gene was one of the genes actually used
      to curate/establish this cell group's identity, versus being
      statistically associated after the fact.
    """
)

st.markdown("### \"Most expressed gene specific to this cell group\"")
st.write(
    """
    When a searched gene is reported this way, it means that -- among all
    the genes checked -- this one shows both high expression levels and high
    specificity (relatively restricted) to this particular group of cells,
    making it a strong candidate marker for that group.
    """
)