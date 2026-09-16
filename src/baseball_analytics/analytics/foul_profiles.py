"""Descriptive batter foul profiles without UI-level aggregation logic."""

import polars as pl


def batter_foul_profiles(pitches: pl.DataFrame) -> pl.DataFrame:
    """Return batter-season totals and rates from persisted pitch metrics."""
    # `player_name` in the Statcast feed names the pitcher, not the batter.
    group = [name for name in ["batter", "game_year"] if name in pitches.columns]
    pa = ["game_pk", "at_bat_number"]
    if not group:
        raise ValueError("Profiles require batter identity")

    pa_stats = pitches.group_by(group + pa).agg(
        pa_pitch_total=pl.len(),
        pa_reached_two_strikes=pl.col("is_two_strike").any(),
        pa_two_strike_fouls=pl.col("is_two_strike_foul").sum(),
    )
    pa_totals = pa_stats.group_by(group).agg(
        total_pas=pl.len(),
        pas_reaching_two_strikes=pl.col("pa_reached_two_strikes").sum(),
        pas_with_2_two_strike_fouls=(pl.col("pa_two_strike_fouls") >= 2).sum(),
        pas_with_3_two_strike_fouls=(pl.col("pa_two_strike_fouls") >= 3).sum(),
        pas_with_5_two_strike_fouls=(pl.col("pa_two_strike_fouls") >= 5).sum(),
        longest_pa=pl.col("pa_pitch_total").max(),
        max_two_strike_fouls_in_pa=pl.col("pa_two_strike_fouls").max(),
    )
    totals = pitches.group_by(group).agg(
        pitches_seen=pl.len(),
        two_strike_pitches=pl.col("is_two_strike").sum(),
        swings=pl.col("is_swing").sum(),
        fouls=pl.col("is_foul").sum(),
        two_strike_swings=(pl.col("is_swing") & pl.col("is_two_strike")).sum(),
        two_strike_fouls=pl.col("is_two_strike_foul").sum(),
        repeated_two_strike_fouls=(pl.col("two_strike_foul_number") >= 2).sum(),
        total_fsv=pl.col("fsv").sum(),
        total_fave=pl.col("fave").sum() if "fave" in pitches.columns else pl.lit(0.0),
        two_strike_fave=pl.when(pl.col("is_two_strike")).then(pl.col("fave")).otherwise(0.0).sum()
        if "fave" in pitches.columns
        else pl.lit(0.0),
        average_spoil_difficulty=pl.col("spoil_difficulty").mean()
        if "spoil_difficulty" in pitches.columns
        else pl.lit(None, dtype=pl.Float64),
        difficult_spoils=(pl.col("spoil_difficulty") >= 0.5).sum()
        if "spoil_difficulty" in pitches.columns
        else pl.lit(0),
    )
    return totals.join(pa_totals, on=group, how="left").with_columns(
        fouls_per_swing=pl.col("fouls") / pl.col("swings").clip(lower_bound=1),
        two_strike_fouls_per_two_strike_swing=pl.col("two_strike_fouls")
        / pl.col("two_strike_swings").clip(lower_bound=1),
        two_strike_fouls_per_pa_reaching_two_strikes=pl.col("two_strike_fouls")
        / pl.col("pas_reaching_two_strikes").clip(lower_bound=1),
        fsv_per_100_pa=100 * pl.col("total_fsv") / pl.col("total_pas").clip(lower_bound=1),
        fsv_per_100_two_strike_pitches=100
        * pl.col("total_fsv")
        / pl.col("two_strike_pitches").clip(lower_bound=1),
        fave_per_100_opportunities=100
        * pl.col("total_fave")
        / pl.col("two_strike_pitches").clip(lower_bound=1),
    )


def batter_foul_splits(pitches: pl.DataFrame) -> pl.DataFrame:
    """Persist batter descriptive splits so research UIs do not aggregate pitches."""
    specs: list[tuple[str, pl.Expr]] = [
        ("attack_zone", pl.col("attack_zone")),
        ("pitch_type", pl.col("pitch_type")),
        ("pitcher_handedness", pl.col("p_throws")),
        ("platoon", pl.col("platoon")),
        (
            "count",
            pl.concat_str(
                [pl.col("balls").cast(pl.String), pl.col("strikes").cast(pl.String)],
                separator="-",
            ),
        ),
    ]
    tables = []
    for dimension, value in specs:
        tables.append(
            pitches.with_columns(split_value=value.cast(pl.String))
            .group_by(["batter", "split_value"])
            .agg(
                pitches=pl.len(),
                swings=pl.col("is_swing").sum(),
                fouls=pl.col("is_foul").sum(),
                two_strike_fouls=pl.col("is_two_strike_foul").sum(),
                total_fsv=pl.col("fsv").sum(),
                total_fave=pl.col("fave").sum() if "fave" in pitches.columns else pl.lit(0.0),
            )
            .with_columns(
                split_dimension=pl.lit(dimension),
                fouls_per_swing=pl.col("fouls") / pl.col("swings").clip(lower_bound=1),
            )
        )
    return pl.concat(tables, how="vertical")
