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
lookup_path = Path("data/interim/batter_lookup.parquet")
if lookup_path.exists():
    profiles = profiles.drop("batter_name", strict=False).join(
        pl.read_parquet(lookup_path).select("batter", "batter_name"), on="batter", how="left"
    )
if "batter_name" not in profiles.columns or profiles.get_column("batter_name").null_count():
    st.error("Batter names are required for this page. Run `baseball build-player-map` first.")
    st.stop()
batters = profiles.select("batter", "batter_name").unique().sort("batter_name")
name_to_id = dict(zip(batters.get_column("batter_name").to_list(), batters.get_column("batter").to_list()))
name = st.selectbox("Batter", list(name_to_id))
batter = name_to_id[name]
profile_display = profiles.filter(pl.col("batter") == batter).drop("batter")
st.dataframe(profile_display.to_pandas(), hide_index=True)
splits = pl.read_parquet(splits_path).filter(pl.col("batter") == batter)
dimension = st.selectbox("Split", splits.get_column("split_dimension").unique().sort().to_list())
view = splits.filter(pl.col("split_dimension") == dimension)
st.dataframe(view.drop("batter").to_pandas(), hide_index=True)
st.bar_chart(view.to_pandas(), x="split_value", y="total_fave")
