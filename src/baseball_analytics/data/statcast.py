"""Statcast download with immutable, date-keyed Parquet caching."""

from datetime import date
from pathlib import Path

import pandas as pd


def ingest_statcast(start: date, end: date, raw_dir: Path) -> Path:
    if end < start:
        raise ValueError("end date must be on or after start date")
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"statcast_{start.isoformat()}_{end.isoformat()}.parquet"
    if destination.exists():
        return destination
    from pybaseball import statcast

    frame: pd.DataFrame = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if frame.empty:
        raise ValueError("Statcast returned no pitches for the requested range")
    frame.to_parquet(destination, index=False)
    return destination
