"""Create a traceable one-row-per-pitch Statcast research table."""

from pathlib import Path

import polars as pl

from baseball_analytics.data.validate import validate_raw_pitches

FOUL_DESCRIPTIONS = {"foul", "foul_pitchout", "foul_tip"}
FOUL_BUNT_DESCRIPTIONS = {"foul_bunt", "missed_bunt"}


def _col(frame: pl.DataFrame, column: str, default: object = None) -> pl.Expr:
    return pl.col(column) if column in frame.columns else pl.lit(default)


def build_canonical_pitches(raw_path: Path, output_path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(raw_path)
    validate_raw_pitches(frame)
    sort_keys = [
        x for x in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if x in frame.columns
    ]
    frame = frame.sort(sort_keys)
    description = pl.col("description").fill_null("")
    pa_id = pl.concat_str(
        [pl.col(x).cast(pl.String) for x in ["game_pk", "at_bat_number"]], separator="-"
    )
    foul = description.is_in(FOUL_DESCRIPTIONS)
    terminal_tip = (description == "foul_tip") & (_col(frame, "events", "") == "strikeout")
    result = frame.with_columns(
        pitch_id=pl.concat_str(
            [pl.col(x).cast(pl.String) for x in ["game_pk", "at_bat_number", "pitch_number"]],
            separator="-",
        ),
        pa_id=pa_id,
        pitcher_pitch_count=pl.int_range(1, pl.len() + 1).over(["game_pk", "pitcher"]),
        pa_pitch_count=pl.int_range(1, pl.len() + 1).over(["game_pk", "at_bat_number"]),
        is_foul=foul & ~terminal_tip,
        is_foul_bunt=description.is_in(FOUL_BUNT_DESCRIPTIONS),
        is_terminal_foul_tip=terminal_tip,
        is_two_strike_foul=foul & (pl.col("strikes") == 2) & ~terminal_tip,
        normalized_plate_x=_col(frame, "plate_x") / 0.83,
        normalized_plate_z=(_col(frame, "plate_z") - _col(frame, "sz_bot"))
        / (_col(frame, "sz_top") - _col(frame, "sz_bot")),
    ).with_columns(
        two_strike_foul_number=pl.when(pl.col("is_two_strike_foul"))
        .then(pl.col("is_two_strike_foul").cast(pl.Int64).cum_sum().over("pa_id"))
        .otherwise(0)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return result
