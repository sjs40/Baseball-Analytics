"""Batter-centric leaderboard queries used by both research interfaces."""

import polars as pl

METRICS = {
    "fsv": "total_fsv",
    "fsv_per_100_pa": "fsv_per_100_pa",
    "two_strike_foul_rate": "two_strike_foul_rate",
    "repeated_foul_rate": "repeated_foul_rate",
    "fave": "total_fave",
    "fave_per_100_opportunities": "fave_per_100_opportunities",
    "spoil_difficulty": "average_spoil_difficulty",
}


def foul_leaderboard(
    pitches: pl.DataFrame,
    metric: str,
    minimum_opportunities: int = 0,
    batter_lookup: pl.DataFrame | None = None,
    season: int | None = None,
    pitch_type: str | None = None,
    attack_zone: str | None = None,
) -> pl.DataFrame:
    """Return a requested batter metric after optional persisted-pitch filters."""
    if metric not in METRICS:
        raise ValueError(f"Unsupported leaderboard metric: {metric}")
    data = pitches
    if season is not None and "game_year" in data.columns:
        data = data.filter(pl.col("game_year") == season)
    if pitch_type is not None:
        data = data.filter(pl.col("pitch_type") == pitch_type)
    if attack_zone is not None:
        data = data.filter(pl.col("attack_zone") == attack_zone)
    result = (
        data.group_by("batter")
        .agg(
            opportunities=pl.col("is_two_strike").sum(),
            total_fsv=pl.col("fsv").sum(),
            total_fave=pl.col("fave").sum() if "fave" in data.columns else pl.lit(0.0),
            two_strike_fouls=pl.col("is_two_strike_foul").sum(),
            two_strike_swings=(pl.col("is_swing") & pl.col("is_two_strike")).sum(),
            repeated_two_strike_fouls=(pl.col("two_strike_foul_number") >= 2).sum(),
            average_spoil_difficulty=pl.col("spoil_difficulty").mean()
            if "spoil_difficulty" in data.columns
            else pl.lit(None, dtype=pl.Float64),
        )
        .with_columns(
            fsv_per_100_pa=100 * pl.col("total_fsv") / pl.col("opportunities").clip(lower_bound=1),
            two_strike_foul_rate=pl.col("two_strike_fouls")
            / pl.col("two_strike_swings").clip(lower_bound=1),
            repeated_foul_rate=pl.col("repeated_two_strike_fouls")
            / pl.col("two_strike_fouls").clip(lower_bound=1),
            fave_per_100_opportunities=100
            * pl.col("total_fave")
            / pl.col("opportunities").clip(lower_bound=1),
        )
        .filter(pl.col("opportunities") >= minimum_opportunities)
    )
    if batter_lookup is not None:
        result = result.join(batter_lookup.select("batter", "batter_name"), on="batter", how="left")
    return result.sort(METRICS[metric], descending=True)
