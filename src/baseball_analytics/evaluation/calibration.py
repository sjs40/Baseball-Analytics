import polars as pl


def calibration_table(pitches: pl.DataFrame, target: str, bins: int = 10) -> pl.DataFrame:
    prob = "expected_" + target + "_probability"
    return (
        pitches.filter(pl.col("swing_outcome").is_in(["whiff", "foul", "ball_in_play"]))
        .with_columns(bin=(pl.col(prob) * bins).floor().clip(upper_bound=bins - 1))
        .group_by("bin")
        .agg(
            observations=pl.len(),
            mean_prediction=pl.col(prob).mean(),
            observed=(pl.col("swing_outcome") == target).mean(),
        )
        .sort("bin")
    )


def brier_score(pitches: pl.DataFrame, target: str) -> float:
    prob = "expected_" + target + "_probability"
    return (
        pitches.filter(pl.col("swing_outcome").is_in(["whiff", "foul", "ball_in_play"]))
        .select(((pl.col(prob) - (pl.col("swing_outcome") == target).cast(pl.Float64)) ** 2).mean())
        .item()
    )
