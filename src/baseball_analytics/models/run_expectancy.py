from pathlib import Path

import polars as pl

STATE_COLUMNS = ["outs_when_up", "on_1b", "on_2b", "on_3b", "balls", "strikes"]


def build_run_expectancy(canonical: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    required = set(STATE_COLUMNS + ["game_pk", "inning", "inning_topbot", "bat_score"])
    missing = required - set(canonical.columns)
    if missing:
        raise ValueError(f"Cannot build run expectancy; missing columns: {sorted(missing)}")
    half = ["game_pk", "inning", "inning_topbot"]
    table = (
        canonical.with_columns(
            runs_to_end=(pl.col("bat_score").max().over(half) - pl.col("bat_score")).clip(
                lower_bound=0
            )
        )
        .filter(pl.col("outs_when_up") < 3)
        .group_by(STATE_COLUMNS)
        .agg(expected_runs=pl.col("runs_to_end").mean(), observations=pl.len())
        .sort(STATE_COLUMNS)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)
    return table


def lookup_state_value(pitches: pl.DataFrame, re_table: pl.DataFrame) -> pl.DataFrame:
    exact = pitches.join(
        re_table.select(STATE_COLUMNS + ["expected_runs"]), on=STATE_COLUMNS, how="left"
    )
    counts = re_table.group_by(["balls", "strikes"]).agg(
        count_expected_runs=pl.col("expected_runs").mean()
    )
    return exact.join(counts, on=["balls", "strikes"], how="left").with_columns(
        state_value=pl.coalesce("expected_runs", "count_expected_runs", pl.lit(0.0))
    )
