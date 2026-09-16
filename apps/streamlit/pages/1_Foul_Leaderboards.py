from pathlib import Path

import polars as pl
import streamlit as st

st.title("Foul Survival Value leaderboard")
path = Path("data/processed/batter_fsv_leaderboard.parquet")
if not path.exists():
    st.info("No FSV output yet. Run `baseball compute-fsv`.")
else:
    minimum = st.number_input("Minimum two-strike fouls", min_value=0, value=0)
    st.dataframe(
        pl.read_parquet(path).filter(pl.col("two_strike_fouls") >= minimum).to_pandas(),
        use_container_width=True,
        hide_index=True,
    )
