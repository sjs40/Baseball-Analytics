import json
from pathlib import Path

import polars as pl
import streamlit as st

st.title("Model validation — experimental baseline")
path = Path("data/processed/foul_validation.json")
if not path.exists():
    st.info("Run train-swing-model, compute-fave, then validate-foul first.")
    st.stop()
report = json.loads(path.read_text(encoding="utf-8"))
st.caption(
    f"Holdout starts {report.get('holdout_start')}; do not interpret whole-sample calibration as holdout evidence."
)
left, right = st.columns(2)
left.metric("Holdout log loss", f"{report['multiclass_log_loss']:.4f}")
right.metric("Naive frequency log loss", f"{report['naive_frequency_log_loss']:.4f}")
target = st.selectbox("Outcome", ["whiff", "foul", "ball_in_play"])
st.subheader("Calibration")
st.dataframe(pl.DataFrame(report["calibration"][target]).to_pandas(), hide_index=True)
st.subheader("Brier scores")
st.dataframe(pl.DataFrame([report["brier_scores"]]).to_pandas(), hide_index=True)
st.subheader("Split-half and stabilization")
metric = st.selectbox("Metric", ["fsv", "fave", "spoil_difficulty"])
st.dataframe(pl.DataFrame(report[f"split_half_{metric}"]).to_pandas(), hide_index=True)
st.line_chart(
    pl.DataFrame(report[f"stabilization_{metric}"]).to_pandas(), x="opportunities", y="pearson"
)
