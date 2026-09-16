from pathlib import Path

import polars as pl
import streamlit as st

st.set_page_config(page_title="Baseball Analytics", layout="wide")
st.title("Baseball Analytics — v0.2 foul research")
st.write("Batter-centric neutral FSV, experimental FAVE, and difficult-pitch spoil research.")
path = Path("data/processed/pitch_fave.parquet")
report_path = Path("docs/research/foul_report.md")
if not path.exists():
    st.info("Run the v0.2 pipeline in README.md to populate research outputs.")
else:
    pitches = pl.read_parquet(path)
    one, two, three = st.columns(3)
    one.metric("Pitches", f"{pitches.height:,}")
    two.metric(
        "Modeled swings",
        f"{pitches.filter(pl.col('swing_outcome').is_in(['whiff', 'foul', 'ball_in_play'])).height:,}",
    )
    three.metric("Two-strike fouls", f"{pitches.get_column('is_two_strike_foul').sum():,}")
    st.caption(
        "FSV is neutral descriptive value. FAVE and spoil difficulty are experimental and require validation before skill claims."
    )
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
