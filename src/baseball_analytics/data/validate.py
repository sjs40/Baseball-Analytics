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
