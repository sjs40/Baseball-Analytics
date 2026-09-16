from pathlib import Path

import polars as pl
import streamlit as st

st.title("Batter foul profile")
profiles_path = Path("data/processed/batter_foul_profiles.parquet")
splits_path = Path("data/processed/batter_foul_splits.parquet")
if not profiles_path.exists() or not splits_path.exists():
    st.info("Run compute-fsv, train-swing-model, compute-fave, and build-foul-profiles first.")
    st.stop()
profiles = pl.read_parquet(profiles_path)
batter = st.selectbox("Batter", profiles.get_column("batter").sort().to_list())
st.dataframe(profiles.filter(pl.col("batter") == batter).to_pandas(), hide_index=True)
splits = pl.read_parquet(splits_path).filter(pl.col("batter") == batter)
dimension = st.selectbox("Split", splits.get_column("split_dimension").unique().sort().to_list())
view = splits.filter(pl.col("split_dimension") == dimension)
st.dataframe(view.to_pandas(), hide_index=True)
st.bar_chart(view.to_pandas(), x="split_value", y="total_fave")
