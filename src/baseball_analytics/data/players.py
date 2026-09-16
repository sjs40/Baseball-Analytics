"""Persisted MLBAM player identity lookup for display, never metric logic."""

from pathlib import Path

import polars as pl


def build_batter_lookup(canonical: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    """Resolve canonical batter MLBAM IDs once and persist the result for interfaces."""
    if "batter" not in canonical.columns:
        raise ValueError("Canonical pitches do not contain batter MLBAM IDs")
    batter_ids = canonical.get_column("batter").drop_nulls().unique().to_list()
    try:
        from pybaseball import playerid_reverse_lookup

        lookup = pl.from_pandas(playerid_reverse_lookup(batter_ids))
    except Exception as error:
        raise RuntimeError(
            "Could not resolve batter names. Retry with network access; metrics remain usable by MLBAM ID."
        ) from error
    result = lookup.select(
        pl.col("key_mlbam").cast(canonical.schema["batter"]).alias("batter"),
        pl.concat_str(
            [pl.col("name_first").str.to_titlecase(), pl.col("name_last").str.to_titlecase()],
            separator=" ",
        ).alias("batter_name"),
    ).unique("batter")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return result
