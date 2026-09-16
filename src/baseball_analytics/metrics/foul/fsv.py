"""FSV v0.1: neutral foul survival value versus a strike counterfactual."""

from pathlib import Path

import polars as pl

from baseball_analytics.models.run_expectancy import lookup_state_value


def compute_fsv(canonical: pl.DataFrame, re_table: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    valued = lookup_state_value(canonical, re_table)
    result = valued.with_columns(
        fsv=pl.when(pl.col("is_two_strike_foul") & ~pl.col("is_foul_bunt"))
        .then(pl.col("state_value"))
        .otherwise(0.0),
        fsv_version=pl.lit("0.1-neutral-strike-counterfactual"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return result


def batter_fsv_leaderboard(pitches: pl.DataFrame) -> pl.DataFrame:
    group = [x for x in ["batter", "player_name"] if x in pitches.columns]
    return (
        pitches.group_by(group)
        .agg(
            fsv=pl.col("fsv").sum(),
            two_strike_fouls=pl.col("is_two_strike_foul").sum(),
            total_fouls=pl.col("is_foul").sum(),
            pitches=pl.len(),
        )
        .sort("fsv", descending=True)
    )
