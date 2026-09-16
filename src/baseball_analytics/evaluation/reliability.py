import polars as pl


def split_half_reliability(
    pitches: pl.DataFrame, metric: str, opportunity: str = "is_two_strike", player: str = "batter"
) -> pl.DataFrame:
    if "game_date" not in pitches.columns:
        raise ValueError("Reliability requires game_date")
    x = pitches.filter(pl.col(opportunity) == True)
    dates = x.get_column("game_date").drop_nulls().unique().sort().to_list()
    if len(dates) < 2:
        raise ValueError("Reliability requires at least two game dates")
    midpoint = dates[(len(dates) - 1) // 2]
    x = x.with_columns(
        half=pl.when(pl.col("game_date") <= midpoint)
        .then(pl.lit("first"))
        .otherwise(pl.lit("second"))
    )
    wide = (
        x.group_by([player, "half"])
        .agg(value=pl.col(metric).mean(), opportunities=pl.len())
        .pivot(on="half", index=player, values=["value", "opportunities"])
    )
    return wide.select(
        pearson=pl.corr("value_first", "value_second"),
        spearman=pl.corr("value_first", "value_second", method="spearman"),
        sample_players=pl.len(),
        first_opportunities=pl.col("opportunities_first").sum(),
        second_opportunities=pl.col("opportunities_second").sum(),
    )
