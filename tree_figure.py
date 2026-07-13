import matplotlib.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def node_from_click_index(fig, point_index, curve_number=None):
    """
    Map a Plotly click event's point index back to the node ID.

    Now that build_tree_figure adds a full dendrogram panel (many line
    traces) before the clickable scatter trace, the scatter is no longer
    reliably fig.data[0] -- pass curve_number (from the click event's
    "curveNumber" field) to look up the exact trace it came from. If
    curve_number isn't given, falls back to the last trace, which is where
    build_tree_figure always adds the click-scatter.
    """
    trace = fig.data[curve_number] if curve_number is not None else fig.data[-1]
    return trace.customdata[point_index]


def compute_dendrogram_layer_custom(cell_linkage_matrix, cell_order):
    """
    Build merged-line-trace coordinates for a top-oriented dendrogram of
    cell_linkage_matrix, with leaves placed at integer x-positions
    strictly according to `cell_order` as given -- NOT scipy's own
    dendrogram() traversal choice.

    This matters because scipy's dendrogram() is free to swap left/right
    children at every merge and still produce an equally valid dendrogram
    of the same linkage matrix -- its `ivl` leaf order isn't guaranteed to
    match whatever order was used to build `cell_order` (e.g. a reindexed
    matrix produced via a different tool, scipy version, or tie-break
    convention than a plain dendrogram() call would choose). Since every
    other page in this app trusts `cell_order` directly, letting this
    function silently pick a different order reorders whole subtrees
    relative to everything else. Building the coordinates ourselves from
    cell_order guarantees they always agree, by construction.

    Requires `cell_order` to actually be a valid leaf ordering for
    cell_linkage_matrix (true here, since the reindexed matrix was
    confirmed to be the result of this exact clustering) -- every cluster
    then occupies a contiguous block of cell_order positions, which is
    what makes a clean, non-crossing dendrogram possible at all.

    Returns
    -------
    leaf_order : list
        Just `cell_order` itself, returned for API symmetry with the
        previous scipy-based version (build_tree_figure uses this for
        cell_to_x).
    dendro_x, dendro_y : list
        None-separated coordinate arrays for a single go.Scatter(mode="lines")
        trace covering every merge in the dendrogram.
    """
    n = len(cell_order)

    # x-position and height of every node in the merge tree; leaves start
    # at their cell_order position with height 0.
    node_x = {i: float(i) for i in range(n)}
    node_height = {i: 0.0 for i in range(n)}

    dendro_x, dendro_y = [], []
    for merge_i, row in enumerate(cell_linkage_matrix):
        a, b, dist = int(row[0]), int(row[1]), float(row[2])
        node_id = n + merge_i
        xa, xb = node_x[a], node_x[b]
        ha, hb = node_height[a], node_height[b]

        left_x, right_x = (xa, xb) if xa <= xb else (xb, xa)
        left_h, right_h = (ha, hb) if xa <= xb else (hb, ha)

        # U-shaped link: up from each child to the merge height, then across
        dendro_x.extend([left_x, left_x, right_x, right_x, None])
        dendro_y.extend([left_h, dist, dist, right_h, None])

        node_x[node_id] = (xa + xb) / 2.0
        node_height[node_id] = dist

    return list(cell_order), dendro_x, dendro_y


def build_tree_figure(
    cluster_tree,
    cluster_node_to_plot,
    cell_order,
    cell_linkage_matrix=None,
    dendrogram_layer=None,
    node_colors=None,
    node_labels=None,
    show_labels=True,
    highlight_nodes=None,
    colormap="tab20",
    label_fontsize=10,
    height=None,
):
    """
    Interactive Plotly port of plots.plot_cluster_hierarchy: a full
    dendrogram over every cell (top panel) above the condensed cluster
    tree (bottom panel), sharing the same cell x-axis.

    Faithfully mirrors the reference matplotlib function:
    - identical greedy depth-packing algorithm for node x-spans (steps 1-2
      here match plot_cluster_hierarchy's steps 2-3 exactly)
    - identical "only label a box if it spans > 10 cells" rule
    - the top dendrogram is drawn from cell_linkage_matrix via
      scipy.cluster.hierarchy.dendrogram(orientation="top"), same as the
      reference function

    Note: box x-positions are derived from the dendrogram's own leaf order
    (scipy's `ivl`), not from `cell_order` as literally given. This is
    deliberately different from the reference matplotlib function (which
    trusts cell_order as already matching the linkage's natural order):
    that assumption doesn't always hold in practice (e.g. after a lossy
    CSV round-trip of the linkage matrix), and when it doesn't, the two
    panels silently drift apart. Deriving positions from ivl instead
    guarantees the tree and the boxes agree on where every cell is,
    regardless of what order cell_order happened to arrive in.

    Pass either `cell_linkage_matrix` (computes the dendrogram layer
    internally, fine for one-off/notebook use) or a precomputed
    `dendrogram_layer=compute_dendrogram_layer(...)` tuple (for repeated
    calls across a progressive-disclosure UI, where the dendrogram itself
    never changes between clicks and shouldn't be recomputed each time).

    Rectangles are drawn via layout.shapes (visual only, not clickable in
    Plotly). A transparent scatter marker is placed at the center of every
    rectangle to capture click events -- this is the standard Plotly
    pattern for making shape-like regions interactive, since `shapes`
    themselves don't fire click callbacks.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        fig.data[-1] is the clickable scatter trace; use
        node_from_click_index(fig, point_index, curve_number) with the
        click event's "curveNumber" to map back to a node ID.
    """
    if dendrogram_layer is not None:
        leaf_order, dendro_x, dendro_y = dendrogram_layer
    elif cell_linkage_matrix is not None:
        leaf_order, dendro_x, dendro_y = compute_dendrogram_layer_custom(cell_linkage_matrix, cell_order)
    else:
        raise ValueError("build_tree_figure requires either cell_linkage_matrix or dendrogram_layer")

    n_cells = len(cell_order)
    cell_to_x = {cell: i for i, cell in enumerate(leaf_order)}

    # 1. x_start / x_end per node
    node_spans = {}
    for node in cluster_node_to_plot:
        cells = cluster_tree.nodes[node].get("cells", [])
        xs = [cell_to_x[c] for c in cells if c in cell_to_x]
        if not xs:
            continue
        node_spans[node] = (min(xs), max(xs) + 1)

    # 2. greedy depth assignment (identical logic to plot_cluster_hierarchy)
    nodes_sorted = sorted(node_spans.keys(), key=lambda n: node_spans[n][0])
    occupied = {}
    node_depths = {}
    for node in nodes_sorted:
        x_start, x_end = node_spans[node]
        depth = 0
        while True:
            ranges = occupied.get(depth, [])
            overlap = any(x_start < r_end and x_end > r_start for r_start, r_end in ranges)
            if not overlap:
                node_depths[node] = depth
                occupied.setdefault(depth, []).append((x_start, x_end))
                break
            depth += 1
    max_depth = max(node_depths.values()) if node_depths else 0

    # 3. colors -- generate a tab20 fallback, or normalize whatever was
    # passed in (rgb/rgba tuples, hex, named colors) into 'rgba(...)'
    # strings Plotly expects
    if node_colors is None:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        node_colors = {
            node: "rgba({},{},{},0.85)".format(*[int(255 * c) for c in cmap(i % cmap.N)[:3]])
            for i, node in enumerate(cluster_node_to_plot)
        }
    else:
        node_colors = {
            node: "rgba({},{},{},0.85)".format(*[int(255 * c) for c in matplotlib.colors.to_rgb(color)])
            for node, color in node_colors.items()
        }

    node_labels = node_labels or {}

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.6], vertical_spacing=0.01,
    )

    # --- Top panel: full dendrogram over every cell (precomputed above,
    # either passed in via dendrogram_layer or just computed) ---
    fig.add_trace(go.Scatter(
        x=dendro_x, y=dendro_y, mode="lines",
        line=dict(color="black", width=0.5),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # --- Bottom panel: condensed cluster boxes ---
    shapes = []
    annotations = []
    xs_center, ys_center, sizes, texts, customdata = [], [], [], [], []

    gap = 0.02
    for node in cluster_node_to_plot:
        if node not in node_spans:
            continue
        x_start, x_end = node_spans[node]
        depth = node_depths[node]
        y_bottom = -(depth + 1)
        y_top = -depth

        if highlight_nodes is not None and node not in highlight_nodes:
            fillcolor = "rgba(211,211,211,0.85)"
        else:
            fillcolor = node_colors.get(node, "rgba(211,211,211,0.85)")

        shapes.append(dict(
            type="rect", xref="x2", yref="y2",
            x0=x_start, x1=x_end, y0=y_bottom + gap, y1=y_top - gap,
            line=dict(color="white", width=0.5),
            fillcolor=fillcolor, opacity=0.85,
        ))

        label = node_labels.get(node, str(node))
        bar_width = x_end - x_start
        if show_labels and bar_width > 10:
            annotations.append(dict(
                x=(x_start + x_end) / 2, y=(y_top + y_bottom) / 2,
                xref="x2", yref="y2", text=str(label), showarrow=False,
                font=dict(size=label_fontsize, color="white"),
                textangle=-90,
            ))

        xs_center.append((x_start + x_end) / 2)
        ys_center.append((y_bottom + y_top) / 2)
        # marker size roughly proportional to box width, capped for very wide/narrow boxes
        sizes.append(max(8, min(40, bar_width * 600 / max(n_cells, 1))))
        n_cells_node = x_end - x_start
        texts.append(f"node {node}<br>{label}<br>{n_cells_node} cells")
        customdata.append(node)

    # invisible click targets, one marker per node, centered in its box
    fig.add_trace(go.Scatter(
        x=xs_center, y=ys_center,
        mode="markers",
        marker=dict(size=sizes, color="rgba(0,0,0,0)"),  # fully transparent
        hovertext=texts,
        hoverinfo="text",
        customdata=customdata,
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        height=height if height is not None else max(180, 28 * (max_depth + 1) + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
    )
    fig.update_xaxes(range=[0, n_cells], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_xaxes(range=[0, n_cells], showgrid=False, zeroline=False, visible=False, row=2, col=1)
    fig.update_yaxes(range=[-(max_depth + 1.5), 0.5], showgrid=False, zeroline=False, visible=False, row=2, col=1)

    return fig