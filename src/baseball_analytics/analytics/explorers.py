"""Read-only query helpers for research interfaces."""

import polars as pl


def filter_leaderboard(
    leaderboard: pl.DataFrame, minimum_opportunities: int = 0, metric: str = "fsv"
) -> pl.DataFrame:
    if metric not in leaderboard.columns:
        raise ValueError(f"Leaderboard does not contain metric {metric!r}")
    return leaderboard.filter(pl.col("two_strike_fouls") >= minimum_opportunities).sort(
        metric, descending=True
    )


def player_pitches(pitches: pl.DataFrame, batter: int) -> pl.DataFrame:
    return pitches.filter(pl.col("batter") == batter).sort(
        [
            name
            for name in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
            if name in pitches.columns
        ]
    )


def pa_pitches(pitches: pl.DataFrame, game_pk: int, at_bat_number: int) -> pl.DataFrame:
    return pitches.filter(
        (pl.col("game_pk") == game_pk) & (pl.col("at_bat_number") == at_bat_number)
    ).sort("pitch_number")
