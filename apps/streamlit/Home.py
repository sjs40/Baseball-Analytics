from pathlib import Path

import polars as pl
import streamlit as st

st.set_page_config(page_title="Baseball Analytics", layout="wide")
st.title("Baseball Analytics — V0.1")
st.write("Inspectable neutral Foul Survival Value (FSV) research outputs.")
st.markdown(
    "FSV is credited only to ordinary two-strike fouls, valued against a strikeout counterfactual."
)
path = Path("data/processed/batter_fsv_leaderboard.parquet")
if not path.exists():
    st.info("Run the ingest → build-features → compute-fsv pipeline to populate this dashboard.")
else:
    st.dataframe(pl.read_parquet(path).to_pandas(), use_container_width=True, hide_index=True)
