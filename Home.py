import streamlit as st

st.set_page_config(page_title="Simple-cell", layout="wide")

pg = st.navigation([
    st.Page("pages/0_Homepage.py", title="Homepage", icon=":material/home:"),
    st.Page("pages/1_Gene_Expression_Explorer.py", title="Gene Expression Explorer", icon=":material/biotech:"),
    st.Page("pages/2_About.py", title="About & Glossary", icon=":material/menu_book:"),
])
pg.run()