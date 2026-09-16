"""FSV v0.2: value of normal two-strike foul versus a strikeout."""

from pathlib import Path

import polars as pl

from baseball_analytics.models.run_expectancy import lookup_state_value

FSV_VERSION = "0.2-neutral-strikeout-counterfactual"


def _state_value(
    pitches: pl.DataFrame, re: pl.DataFrame, prefix: str, outs: str, balls: str, strikes: str
) -> pl.DataFrame:
    x = pitches.with_columns(
        outs_when_up=pl.col(outs), balls=pl.col(balls), strikes=pl.col(strikes)
    )
    return lookup_state_value(x, re).select(
        "pitch_id",
        pl.col("state_value").alias(prefix + "_state_value"),
        pl.col("state_lookup_tier").alias(prefix + "_lookup_tier"),
    )


def compute_fsv(canonical: pl.DataFrame, re_table: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    if "pitch_id" not in canonical.columns:
        canonical = (
            canonical.with_row_index("_fsv_row")
            .with_columns(pitch_id=pl.col("_fsv_row").cast(pl.String))
            .drop("_fsv_row")
        )
    for column, default in {"is_terminal_foul_tip": False, "is_foul": False}.items():
        if column not in canonical.columns:
            canonical = canonical.with_columns(**{column: pl.lit(default)})
    x = canonical.with_columns(
        post_balls=pl.col("balls"),
        post_strikes=pl.when(pl.col("is_foul") & (pl.col("strikes") < 2))
        .then(pl.col("strikes") + 1)
        .otherwise(pl.col("strikes")),
        counter_outs=pl.col("outs_when_up") + 1,
        counter_balls=pl.lit(0),
        counter_strikes=pl.lit(0),
    )
    x = x.with_columns(
        whiff_outs=pl.when(pl.col("strikes") == 2)
        .then(pl.col("outs_when_up") + 1)
        .otherwise(pl.col("outs_when_up")),
        whiff_balls=pl.col("balls"),
        whiff_strikes=pl.when(pl.col("strikes") < 2)
        .then(pl.col("strikes") + 1)
        .otherwise(pl.lit(0)),
    )
    pre = _state_value(x, re_table, "pre_pitch", "outs_when_up", "balls", "strikes")
    actual = _state_value(x, re_table, "actual", "outs_when_up", "post_balls", "post_strikes")
    counter = _state_value(
        x, re_table, "strikeout_counterfactual", "counter_outs", "counter_balls", "counter_strikes"
    )
    whiff = _state_value(x, re_table, "whiff", "whiff_outs", "whiff_balls", "whiff_strikes")
    result = (
        canonical.join(pre, on="pitch_id", how="left")
        .join(actual, on="pitch_id", how="left")
        .join(counter, on="pitch_id", how="left")
        .join(whiff, on="pitch_id", how="left")
        .with_columns(
            fsv=pl.when(
                pl.col("is_two_strike_foul")
                & ~pl.col("is_foul_bunt")
                & ~pl.col("is_terminal_foul_tip")
            )
            .then(pl.col("actual_state_value") - pl.col("strikeout_counterfactual_state_value"))
            .otherwise(0.0),
            fsv_version=pl.lit(FSV_VERSION),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return result


def batter_fsv_leaderboard(
    pitches: pl.DataFrame, batter_lookup: pl.DataFrame | None = None
) -> pl.DataFrame:
    if "batter" not in pitches.columns:
        raise ValueError("Batter FSV leaderboard requires the Statcast batter identifier")
    # Statcast's `player_name` is the pitcher. Never use it as batter metadata.
    result = (
        pitches.group_by("batter")
        .agg(
            fsv=pl.col("fsv").sum(),
            two_strike_fouls=pl.col("is_two_strike_foul").sum(),
            total_fouls=pl.col("is_foul").sum(),
            pitches=pl.len(),
        )
        .with_columns(
            fsv_per_100_two_strike_pitches=100
            * pl.col("fsv")
            / pl.col("two_strike_fouls").clip(lower_bound=1)
        )
    )
    if batter_lookup is not None:
        result = result.join(batter_lookup.select("batter", "batter_name"), on="batter", how="left")
    return result.sort("fsv", descending=True)
