"""OOS foul and whiff-avoidance residual metrics (experimental V0.3)."""

import polars as pl

FAE_VERSION = "0.3-oos-foul-above-expected"
WHIFF_AVOIDANCE_VERSION = "0.3-oos-whiff-avoidance"


def add_residuals(pitches: pl.DataFrame, ledger: pl.DataFrame) -> pl.DataFrame:
    """Join OOS probabilities and add traceable pitch-level residuals."""
    result = pitches.join(ledger, on=["pitch_id", "game_date", "batter", "pitcher"], how="inner")
    return result.with_columns(
        foul_above_expected=(pl.col("swing_outcome") == "foul").cast(pl.Float64)
        - pl.col("expected_foul_probability"),
        whiffs_avoided_above_expected=pl.col("expected_whiff_probability")
        - (pl.col("swing_outcome") == "whiff").cast(pl.Float64),
        spoil_difficulty_v02=pl.when((pl.col("is_two_strike")) & (pl.col("swing_outcome") == "foul"))
        .then(pl.col("expected_whiff_probability")),
        fae_version=pl.lit(FAE_VERSION),
        whiff_avoidance_version=pl.lit(WHIFF_AVOIDANCE_VERSION),
    )


def batter_residual_leaderboard(pitches: pl.DataFrame, group: list[str] | None = None) -> pl.DataFrame:
    """Aggregate residuals without qualification thresholds; retain samples."""
    group = group or ["batter"]
    two = pl.col("is_two_strike")
    return pitches.group_by(group).agg(
        swings=pl.len(), actual_fouls=(pl.col("swing_outcome") == "foul").sum(),
        expected_fouls=pl.col("expected_foul_probability").sum(),
        fouls_above_expected=pl.col("foul_above_expected").sum(),
        expected_whiffs=pl.col("expected_whiff_probability").sum(),
        actual_whiffs=(pl.col("swing_outcome") == "whiff").sum(),
        whiffs_avoided_above_expected=pl.col("whiffs_avoided_above_expected").sum(),
        two_strike_swings=two.sum(),
        actual_two_strike_fouls=pl.when(two).then(pl.col("swing_outcome") == "foul").sum(),
        expected_two_strike_fouls=pl.when(two).then(pl.col("expected_foul_probability")).sum(),
        two_strike_fouls_above_expected=pl.when(two).then(pl.col("foul_above_expected")).sum(),
        average_spoil_difficulty=pl.col("spoil_difficulty_v02").mean(),
        difficult_spoils=(pl.col("spoil_difficulty_v02") >= 0.5).sum(),
    ).with_columns(
        fouls_above_expected_per_100_swings=100 * pl.col("fouls_above_expected") / pl.col("swings").clip(lower_bound=1),
        whiffs_avoided_above_expected_per_100_swings=100 * pl.col("whiffs_avoided_above_expected") / pl.col("swings").clip(lower_bound=1),
        two_strike_fae_per_100_swings=100 * pl.col("two_strike_fouls_above_expected") / pl.col("two_strike_swings").clip(lower_bound=1),
    )


def empirical_bayes_shrinkage(table: pl.DataFrame, total: str = "fouls_above_expected", opportunities: str = "swings") -> pl.DataFrame:
    """Interpretable normal-prior shrinkage for residual totals/rates."""
    raw_rate = pl.col(total) / pl.col(opportunities).clip(lower_bound=1)
    prior = table.select(raw_rate.mean()).item()
    between = table.select(raw_rate.var()).item() or 0.0
    # Bernoulli residual variance upper bound; conservative for small samples.
    noise = 0.25 / pl.col(opportunities).clip(lower_bound=1)
    weight = pl.lit(between) / (pl.lit(between) + noise)
    return table.with_columns(
        raw_residual_rate=raw_rate,
        league_prior_residual_rate=pl.lit(prior),
        shrinkage_weight=weight,
        shrunk_residual_rate=weight * raw_rate + (1 - weight) * pl.lit(prior),
        residual_standard_error=noise.sqrt(),
    )
