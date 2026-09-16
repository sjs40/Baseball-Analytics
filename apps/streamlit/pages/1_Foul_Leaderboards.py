from pathlib import Path

import polars as pl
import streamlit as st

from baseball_analytics.analytics.leaderboards import METRICS, foul_leaderboard

st.title("Foul Survival Value leaderboard")
st.caption("Batter perspective. Player names come from the persisted MLBAM batter lookup.")
path = Path("data/processed/pitch_fave.parquet")
lookup_path = Path("data/interim/batter_lookup.parquet")
if not path.exists():
    st.info("No research output yet. Run compute-fsv, train-swing-model, and compute-fave.")
else:
    minimum = st.number_input("Minimum two-strike opportunities", min_value=0, value=0)
    frame = pl.read_parquet(path)
    metric = st.selectbox("Metric", list(METRICS))
    seasons = frame.get_column("game_year").drop_nulls().unique().sort().to_list()
    season = st.selectbox(
        "Season", [None, *seasons], format_func=lambda value: "All" if value is None else value
    )
    pitch_types = frame.get_column("pitch_type").drop_nulls().unique().sort().to_list()
    pitch_type = st.selectbox(
        "Pitch type",
        [None, *pitch_types],
        format_func=lambda value: "All" if value is None else value,
    )
    zones = frame.get_column("attack_zone").drop_nulls().unique().sort().to_list()
    zone = st.selectbox(
        "Attack zone", [None, *zones], format_func=lambda value: "All" if value is None else value
    )
    lookup = pl.read_parquet(lookup_path) if lookup_path.exists() else None
    st.dataframe(
        foul_leaderboard(frame, metric, minimum, lookup, season, pitch_type, zone).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )
