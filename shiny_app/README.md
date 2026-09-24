# HCC Explorer — Shiny for Python prototype

A drop-in prototype rewriting the Streamlit UI layer of `HCC_app` on top of
Shiny for Python, while keeping every piece of the compute/plot layer
(`gene_plots.py`, `tree_figure.py`) and every data file unchanged.

## Layout

The Shiny app lives in its own folder so it can sit next to the Streamlit
app (still deployed from `Home.py` + `pages/` at the repo root) without
touching it.

```
HCC_app/
├── Home.py, pages/, data_loaders.py, gene_plots.py, requirements.txt   ← Streamlit (unchanged)
├── data/                                                                ← shared by both apps
└── shiny_app/
    ├── app.py
    ├── data_loaders.py   ← Shiny version (lru_cache, nullary loaders)
    ├── gene_plots.py     ← copy of the root gene_plots.py
    ├── pyproject.toml
    └── README.md
```

Keep `pyproject.toml` and any lock file (e.g. `uv.lock`) inside `shiny_app/`:
Streamlit Community Cloud also looks for dependency files at the repo root,
and some formats take precedence over `requirements.txt`.

## Running

From inside `shiny_app/`:

```bash
uv sync            # or: pip install -e .
shiny run --reload app.py
```

Data is read from `../data` by default; set `GENE_APP_DATA_DIR` to override.
Opens on `http://localhost:8000` by default.

## What maps to what

| Streamlit                                     | Shiny for Python                              |
| --------------------------------------------- | --------------------------------------------- |
| `Home.py` + `st.navigation` + `pages/`        | `ui.page_navbar` with two `ui.nav_panel`s     |
| `st.set_page_config`                          | Args on `ui.page_navbar`                      |
| `st.cache_resource` on loaders                | `@lru_cache(maxsize=None)` (process-scoped)   |
| `st.text_input`, `st.selectbox`, `st.slider`  | `ui.input_text`, `ui.input_selectize`, `ui.input_slider` |
| `st.radio(..., horizontal=True)`              | `ui.input_radio_buttons(..., inline=True)`    |
| `st.toggle`                                   | `ui.input_switch`                             |
| `st.pyplot(fig)`                              | `@render.plot`                                |
| `st.markdown`, `st.info`, `st.success`        | `ui.markdown`, `ui.div(..., class_="alert alert-*")` |
| `st.data_editor` with a "Plot" checkbox column| ipyaggrid `Grid` (via shinywidgets) with a JS "Plot" button column that sends `input.plot_gene_click` |
| `st.fragment`-scoped rerun for the gene table | Automatic — falls out of the reactive graph, no scoping needed |
| Conditional widget appearance                 | `ui.panel_conditional("input.x == 'y'", ...)` or `@render.ui` |
| Multi-cluster tissue picker                   | Dynamic `input_select` rendered by `@render.ui` |

## The interesting architectural change

Streamlit runs the whole script top-to-bottom on any interaction and asks
you to slap `@st.cache_data` / `@st.cache_resource` / `@st.fragment` around
things to avoid redoing expensive work. Shiny builds an explicit reactive
graph: only downstream nodes of the changed input recompute.

For this app the graph is:

```
inputs ─┐
        │
        ▼
     resolved ──▶ cluster_ctx ─┬─▶ spectrum_inputs ─▶ spectrum_plot
                               │
                               ├─▶ ref_gene_pane / query_gene_plot / extra_plot_i (×20)
                               │        ▲
                               │        └── context_matrix ◀── range inputs
                               │
                               ├─▶ gene_table_df ─▶ gene_table
                               │
                               └─▶ has_cluster ─▶ context_section (tab layout)

plotted_genes (Plot buttons / add-gene box) ─▶ selected_extra_genes ─▶ per-slot values
```

Concretely:

- Moving the **threshold slider** invalidates only `spectrum_inputs` and
  `spectrum_plot`. Ref/query/extra plots and the table don't recompute.
- Changing the **cell range** (or zoom to cluster) invalidates
  `context_matrix` and the plots that read it. The spectrum plot and table
  don't recompute.
- Toggling **include sub-cluster annotations** invalidates `gene_table_df`
  only. All plots stay put.
- Clicking **Plot** in the gene table only re-renders the extra-plot slots
  whose gene changed. The grid itself isn't rebuilt: button state lives
  client-side in `window.HCC_plotted_genes`.
- **Navigating** between clusters doesn't rebuild the context tab layout
  (it depends on `has_cluster`, not `cluster_ctx`), so the range, the
  sub-cluster switch and the add-gene text are kept.

None of this needs `@st.fragment`.

## Prototype-level shortcuts

- **Extra-gene plots are pre-registered as 20 slots** (`MAX_EXTRA_GENES`).
  The output_ui for each slot returns nothing when the slot is empty, so
  the user only sees plots for the genes they actually selected. Beyond
  20, the header says how many are hidden.
- **Large clusters** (more than `MAX_NAV_CELLS` = 1000 cells) can't be
  reached with the nav buttons and get a message instead of the spectrum
  plot, whichever way they were reached.
- **ipyaggrid bundles AG Grid Enterprise** and logs an "evaluation
  licence" notice in the browser console. Only community features are used.
- **`@render.plot` DPI is not controlled here.** The Streamlit version
  used `dpi=500` on the spectrum plot, which is one of the OOM culprits
  on Streamlit Community Cloud. Shiny's `@render.plot` renders via
  matplotlib's Agg backend at a default DPI; if you port for real, this
  is a good moment to drop it to 150–200.
- **Deployment target.** Shiny apps can go on [shinyapps.io](https://www.shinyapps.io/)
  (free tier: 1 GB RAM, 25 active hours/month — comparable to Streamlit
  Community Cloud), Posit Connect Cloud, Hugging Face Spaces, or self-host.
  Nothing about the code changes, but the ~570 MB float32 matrix plus
  per-cluster copies is likely too much for a 1 GB instance. The matrix
  is stored with Git LFS, so the host must fetch LFS files.
