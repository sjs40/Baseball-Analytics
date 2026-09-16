"""Experimental FAVE v0.1, explicitly decomposed in delta-run-expectancy units."""

from pathlib import Path

import polars as pl

from baseball_analytics.models.swing_outcome import predict_swing_outcomes

FAVE_VERSION = "0.1-experimental-delta-re-decomposition"


def compute_fave(fsv_pitches: pl.DataFrame, model: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    """Value a foul against its expected swing outcome; do not assume fouls are positive."""
    required = {
        "pre_pitch_state_value",
        "actual_state_value",
        "whiff_state_value",
        "is_foul",
        "is_foul_bunt",
        "is_terminal_foul_tip",
    }
    missing = required - set(fsv_pitches.columns)
    if missing:
        raise ValueError(f"FAVE requires FSV v0.2 fields: {sorted(missing)}")
    values = predict_swing_outcomes(fsv_pitches, model).with_columns(
        foul_delta_re=pl.col("actual_state_value") - pl.col("pre_pitch_state_value"),
        whiff_delta_re=pl.col("whiff_state_value") - pl.col("pre_pitch_state_value"),
    )
    expected = (
        pl.col("expected_whiff_probability") * pl.col("whiff_delta_re")
        + pl.col("expected_foul_probability") * pl.col("foul_delta_re")
        + pl.col("expected_ball_in_play_probability") * pl.col("expected_bip_delta_re")
    )
    result = values.with_columns(
        expected_swing_value=expected,
        actual_foul_value=pl.col("foul_delta_re"),
        fave=pl.when(pl.col("is_foul") & ~pl.col("is_foul_bunt") & ~pl.col("is_terminal_foul_tip"))
        .then(pl.col("foul_delta_re") - expected)
        .otherwise(0.0),
        fave_version=pl.lit(FAVE_VERSION),
        spoil_difficulty=pl.when(pl.col("is_two_strike_foul"))
        .then(pl.col("expected_whiff_probability"))
        .otherwise(None),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return result
