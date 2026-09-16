"""Strictly lagged PA sequence features."""

import polars as pl


def add_sequence_features(frame: pl.DataFrame) -> pl.DataFrame:
    pa = ["game_pk", "at_bat_number"]
    t = pl.col("pitch_type").shift(1).over(pa) if "pitch_type" in frame.columns else pl.lit(None)
    v = (
        pl.col("release_speed").shift(1).over(pa)
        if "release_speed" in frame.columns
        else pl.lit(None)
    )
    x = pl.col("plate_x").shift(1).over(pa) if "plate_x" in frame.columns else pl.lit(None)
    z = pl.col("plate_z").shift(1).over(pa) if "plate_z" in frame.columns else pl.lit(None)
    return frame.with_columns(
        previous_pitch_type=t,
        previous_pitch_velocity=v,
        previous_pitch_result=pl.col("description").shift(1).over(pa),
        previous_plate_x=x,
        previous_plate_z=z,
        velocity_difference_from_previous=pl.col("release_speed") - v
        if "release_speed" in frame.columns
        else pl.lit(None),
        same_pitch_type_as_previous=pl.col("pitch_type") == t
        if "pitch_type" in frame.columns
        else pl.lit(None),
        pitch_type_sequence=t,
    )
