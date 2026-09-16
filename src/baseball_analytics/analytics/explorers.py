"""Read-only query helpers for research interfaces."""

import polars as pl


def game_display_options(pitches: pl.DataFrame) -> pl.DataFrame:
    """One display label per game; preserves game identifiers only for filtering."""
    required = {"game_pk", "game_date", "away_team", "home_team"}
    missing = required - set(pitches.columns)
    if missing:
        raise ValueError(f"Game display requires columns: {sorted(missing)}")
    return (
        pitches.group_by("game_pk")
        .agg(
            game_date=pl.col("game_date").first(),
            away_team=pl.col("away_team").first(),
            home_team=pl.col("home_team").first(),
        )
        .with_columns(
            game_label=pl.concat_str(
                [
                    pl.col("game_date").cast(pl.String),
                    pl.col("away_team"),
                    pl.lit("at"),
                    pl.col("home_team"),
                ],
                separator=" ",
            )
        )
        .sort(["game_date", "away_team", "home_team"])
    )


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
