"""Statcast download with immutable, date-keyed Parquet caching."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd


def ingest_statcast(start: date, end: date, raw_dir: Path) -> Path:
    if end < start:
        raise ValueError("end date must be on or after start date")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if start.year != end.year:
        raise ValueError("Incremental immutable partitions may not span seasons; ingest one season at a time")
    partition = raw_dir / "statcast" / f"season={start.year}"
    partition.mkdir(parents=True, exist_ok=True)
    destination = partition / f"statcast_{start.isoformat()}_{end.isoformat()}.parquet"
    if destination.exists():
        return destination
    from pybaseball import statcast

    frame: pd.DataFrame = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if frame.empty:
        raise ValueError("Statcast returned no pitches for the requested range")
    frame.to_parquet(destination, index=False)
    destination.with_suffix(".metadata.json").write_text(
        json.dumps({
            "source": "pybaseball.statcast", "requested_start": start.isoformat(),
            "requested_end": end.isoformat(), "retrieved_at": datetime.now(UTC).isoformat(),
            "row_count": len(frame), "schema_version": "statcast-raw-v0.3",
        }, indent=2), encoding="utf-8"
    )
    return destination


def available_statcast_partitions(raw_dir: Path) -> list[Path]:
    """Return immutable source partitions, never synthesizing or overwriting them."""
    root = raw_dir / "statcast"
    return sorted(root.rglob("*.parquet")) if root.exists() else []
