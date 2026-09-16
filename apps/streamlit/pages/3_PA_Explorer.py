from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from baseball_analytics.analytics.explorers import game_display_options, pa_pitches

st.title("Plate appearance explorer")
path = Path("data/processed/pitch_fave.parquet")
if not path.exists():
    st.info("Run compute-fave first.")
    st.stop()
pitches = pl.read_parquet(path)
games = game_display_options(pitches)
label_to_game = dict(zip(games.get_column("game_label").to_list(), games.get_column("game_pk").to_list()))
game_label = st.selectbox("Game", list(label_to_game))
game_pk = label_to_game[game_label]
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
    zone_bottom = float(locations.get_column("sz_bot").drop_nulls().mean()) if "sz_bot" in locations.columns and locations.get_column("sz_bot").drop_nulls().len() else 1.5
    zone_top = float(locations.get_column("sz_top").drop_nulls().mean()) if "sz_top" in locations.columns and locations.get_column("sz_top").drop_nulls().len() else 3.5
    colors = ["#ff4b4b" if foul else "#4f9ddf" for foul in locations.get_column("is_two_strike_foul").to_list()]
    figure = go.Figure(go.Scatter(
        x=locations.get_column("plate_x").to_list(), y=locations.get_column("plate_z").to_list(),
        mode="markers+text", text=locations.get_column("pitch_number").cast(pl.String).to_list(),
        textposition="top center", marker={"size": 11, "color": colors}, name="Pitch",
    ))
    figure.add_shape(type="rect", x0=-0.83, x1=0.83, y0=zone_bottom, y1=zone_top, line={"color": "white", "width": 3})
    figure.update_layout(xaxis={"range": [-2, 2], "title": "Horizontal location (ft)"}, yaxis={"range": [0, 5], "title": "Vertical location (ft)", "scaleanchor": "x", "scaleratio": 1}, height=560, showlegend=False)
    st.plotly_chart(figure, use_container_width=True)
    st.caption("White rectangle is the batter-specific Statcast strike zone; red markers are two-strike fouls.")
