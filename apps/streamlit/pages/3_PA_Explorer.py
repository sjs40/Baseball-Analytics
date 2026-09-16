from pathlib import Path

import polars as pl
import streamlit as st

from baseball_analytics.analytics.explorers import pa_pitches

st.title("Plate appearance explorer")
path = Path("data/processed/pitch_fave.parquet")
if not path.exists():
    st.info("Run compute-fave first.")
    st.stop()
pitches = pl.read_parquet(path)
game_pk = st.selectbox("Game", pitches.get_column("game_pk").unique().sort().to_list())
pas = (
    pitches.filter(pl.col("game_pk") == game_pk)
    .get_column("at_bat_number")
    .unique()
    .sort()
    .to_list()
)
at_bat = st.selectbox("At-bat", pas)
columns = [
    name
    for name in [
        "pitch_number",
        "balls",
        "strikes",
        "pitch_type",
        "release_speed",
        "plate_x",
        "plate_z",
        "attack_zone",
        "description",
        "swing_outcome",
        "fsv",
        "expected_whiff_probability",
        "expected_foul_probability",
        "expected_ball_in_play_probability",
        "fave",
    ]
    if name in pitches.columns
]
pa = pa_pitches(pitches, game_pk, at_bat)
st.dataframe(pa.select(columns).to_pandas(), hide_index=True)
st.subheader("Pitch locations")
locations = pa.filter(pl.col("plate_x").is_not_null() & pl.col("plate_z").is_not_null())
if locations.is_empty():
    st.info("This PA has no usable pitch-location observations.")
else:
    st.scatter_chart(locations.to_pandas(), x="plate_x", y="plate_z", color="is_two_strike_foul")
