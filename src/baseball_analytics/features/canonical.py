"""Create a traceable one-row-per-pitch Statcast research table."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from baseball_analytics.data.validate import validate_raw_pitches
from baseball_analytics.features.foul import (
    FOUL_BUNT_DESCRIPTIONS,
    FOUL_DESCRIPTIONS,
    swing_outcome,
)
from baseball_analytics.features.pitch_location import add_location_features
from baseball_analytics.features.pitch_sequence import add_sequence_features


def _col(frame: pl.DataFrame, column: str, default: object = None) -> pl.Expr:
    return pl.col(column) if column in frame.columns else pl.lit(default)


def build_canonical_pitches(raw_path: Path | Sequence[Path], output_path: Path) -> pl.DataFrame:
    """Build canonical pitches from one or more immutable raw Parquet sources."""
    raw_paths = [raw_path] if isinstance(raw_path, Path) else list(raw_path)
    if not raw_paths:
        raise ValueError("At least one raw Parquet file is required")
    frames = [pl.read_parquet(path) for path in raw_paths]
    for frame in frames:
        validate_raw_pitches(frame)
    frame = pl.concat(frames, how="diagonal_relaxed")
    validate_raw_pitches(frame)
    frame = frame.sort(
        [x for x in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if x in frame.columns]
    )
    desc = _col(frame, "description", "").fill_null("")
    tip = (desc == "foul_tip") & (_col(frame, "events", "") == "strikeout")
    outcome = desc.map_elements(swing_outcome, return_dtype=pl.String)
    pa = ["game_pk", "at_bat_number"]
    result = frame.with_columns(
        pitch_id=pl.concat_str(
            [pl.col(x).cast(pl.String) for x in ["game_pk", "at_bat_number", "pitch_number"]],
            separator="-",
        ),
        pa_id=pl.concat_str([pl.col(x).cast(pl.String) for x in pa], separator="-"),
        pa_pitch_count=pl.int_range(1, pl.len() + 1).over(pa),
        pitcher_pitch_count=pl.int_range(1, pl.len() + 1).over(["game_pk", "pitcher"]),
        swing_outcome=outcome,
        is_swing=outcome.is_in(["whiff", "foul", "ball_in_play", "bunt", "other"]),
        is_foul=desc.is_in(FOUL_DESCRIPTIONS) & ~tip,
        is_foul_bunt=desc.is_in(FOUL_BUNT_DESCRIPTIONS),
        is_terminal_foul_tip=tip,
        is_two_strike=pl.col("strikes") == 2,
        is_full_count=(pl.col("balls") == 3) & (pl.col("strikes") == 2),
        is_two_strike_foul=desc.is_in(FOUL_DESCRIPTIONS) & (pl.col("strikes") == 2) & ~tip,
        platoon=pl.when(_col(frame, "stand").is_null() | _col(frame, "p_throws").is_null())
        .then(pl.lit("unknown"))
        .when(_col(frame, "stand") == _col(frame, "p_throws"))
        .then(pl.lit("same"))
        .otherwise(pl.lit("opposite")),
    ).with_columns(
        two_strike_pitch_number=pl.when(pl.col("is_two_strike"))
        .then(pl.col("is_two_strike").cast(pl.Int64).cum_sum().over(pa))
        .otherwise(0),
        pitches_since_reaching_two_strikes=pl.when(pl.col("is_two_strike"))
        .then(pl.col("is_two_strike").cast(pl.Int64).cum_sum().over(pa) - 1)
        .otherwise(0),
        two_strike_foul_number=pl.when(pl.col("is_two_strike_foul"))
        .then(pl.col("is_two_strike_foul").cast(pl.Int64).cum_sum().over(pa))
        .otherwise(0),
        total_two_strike_fouls_in_pa_so_far=pl.col("is_two_strike_foul")
        .cast(pl.Int64)
        .cum_sum()
        .over(pa),
    )
    # Consecutive is reset by any non-two-strike-foul; it is intentionally distinct from total.
    result = (
        result.with_columns(
            _two_foul_cumsum=pl.col("is_two_strike_foul").cast(pl.Int64).cum_sum().over(pa)
        )
        .with_columns(
            _last_reset_total=pl.when(~pl.col("is_two_strike_foul"))
            .then(pl.col("_two_foul_cumsum"))
            .otherwise(None)
            .forward_fill()
            .over(pa)
            .fill_null(0)
        )
        .with_columns(
            consecutive_two_strike_foul_count=pl.when(pl.col("is_two_strike_foul"))
            .then(pl.col("_two_foul_cumsum") - pl.col("_last_reset_total"))
            .otherwise(0)
        )
        .drop(["_two_foul_cumsum", "_last_reset_total"])
    )
    result = add_sequence_features(add_location_features(result))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return result
