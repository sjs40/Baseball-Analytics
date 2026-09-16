import polars as pl

REQUIRED_RAW_COLUMNS = {
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "description",
}


def validate_raw_pitches(frame: pl.DataFrame) -> None:
    missing = REQUIRED_RAW_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Raw Statcast data is missing required columns: {sorted(missing)}")
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    duplicates = frame.select(pl.struct(keys).is_duplicated().sum()).item()
    if duplicates:
        raise ValueError(f"Raw Statcast data has {duplicates} duplicate pitch identities")


def validate_canonical_pitches(frame: pl.DataFrame) -> None:
    """Validate reproducibility-critical canonical pitch invariants."""
    required = REQUIRED_RAW_COLUMNS | {
        "pitch_id",
        "pa_id",
        "is_swing",
        "is_foul",
        "is_two_strike_foul",
        "attack_zone",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical pitches are missing required columns: {sorted(missing)}")
    invalid_counts = frame.filter((pl.col("balls") > 3) | (pl.col("strikes") > 2)).height
    if invalid_counts:
        raise ValueError(f"Canonical pitches contain {invalid_counts} impossible counts")
    duplicate_ids = frame.select(pl.col("pitch_id").is_duplicated().sum()).item()
    if duplicate_ids:
        raise ValueError(f"Canonical pitches contain {duplicate_ids} duplicate pitch IDs")
