"""Count-aware empirical run expectancy with explicit weighted fallbacks."""

from pathlib import Path

import polars as pl

STATE_COLUMNS = ["outs_when_up", "on_1b", "on_2b", "on_3b", "balls", "strikes"]
RE_VERSION = "0.3-terminal-score-aware"


def add_state_ids(frame: pl.DataFrame) -> pl.DataFrame:
    runners = (
        pl.col("on_1b").is_not_null().cast(pl.Int8)
        + 2 * pl.col("on_2b").is_not_null().cast(pl.Int8)
        + 4 * pl.col("on_3b").is_not_null().cast(pl.Int8)
    )
    return frame.with_columns(
        runner_bitmask=runners,
        base_out_state=pl.concat_str(
            [pl.col("outs_when_up").cast(pl.String), runners.cast(pl.String)], separator="-"
        ),
        count_state=pl.concat_str(
            [pl.col("balls").cast(pl.String), pl.col("strikes").cast(pl.String)], separator="-"
        ),
        full_state_id=pl.concat_str(
            [pl.col(x).cast(pl.String) for x in STATE_COLUMNS], separator="-"
        ),
    )


def build_run_expectancy(
    canonical: pl.DataFrame, output_path: Path, train_end: str | None = None
) -> pl.DataFrame:
    """Fit RE from an explicit reference window.

    ``bat_score`` is the pre-pitch batting score; whenever Statcast supplies a
    post-pitch score, the final non-null value in the half inning is the
    authoritative terminal score.  Older/source-incomplete rows deliberately
    fall back to the largest observed pre-pitch score and advertise that fact.
    """
    missing = set(STATE_COLUMNS + ["game_pk", "inning", "inning_topbot", "bat_score"]) - set(
        canonical.columns
    )
    if missing:
        raise ValueError(f"Cannot build run expectancy; missing columns: {sorted(missing)}")
    if canonical.filter((pl.col("balls") > 3) | (pl.col("strikes") > 2)).height:
        raise ValueError("Impossible count in canonical pitches")
    if train_end is not None:
        if "game_date" not in canonical.columns:
            raise ValueError("Chronological RE fitting requires game_date")
        canonical = canonical.filter(pl.col("game_date") <= train_end)
    if canonical.is_empty():
        raise ValueError("No pitches remain in the requested RE reference window")
    half = ["game_pk", "inning", "inning_topbot"]
    terminal_score = pl.col("bat_score").max().over(half)
    score_source = pl.lit("pre_pitch_max_fallback")
    if "post_bat_score" in canonical.columns:
        terminal_score = pl.coalesce(
            pl.col("post_bat_score").drop_nulls().last().over(half), terminal_score
        )
        score_source = pl.when(pl.col("post_bat_score").drop_nulls().count().over(half) > 0).then(
            pl.lit("final_post_bat_score")
        ).otherwise(pl.lit("pre_pitch_max_fallback"))
    table = (
        canonical.with_columns(
            runs_to_end=(terminal_score - pl.col("bat_score")).clip(lower_bound=0),
            re_score_source=score_source,
        )
        .filter(pl.col("outs_when_up") < 3)
        .group_by(STATE_COLUMNS)
        .agg(
            expected_runs=pl.col("runs_to_end").mean(),
            observations=pl.len(),
            score_source=pl.col("re_score_source").mode().first(),
        )
        .sort(STATE_COLUMNS)
    )
    dates = canonical.get_column("game_date") if "game_date" in canonical.columns else None
    table = add_state_ids(table).with_columns(
        re_version=pl.lit(RE_VERSION),
        reference_start=pl.lit(str(dates.min()) if dates is not None else None, dtype=pl.String),
        reference_end=pl.lit(str(dates.max()) if dates is not None else None, dtype=pl.String),
        train_end=pl.lit(train_end, dtype=pl.String),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)
    return table


def lookup_state_value(pitches: pl.DataFrame, re_table: pl.DataFrame) -> pl.DataFrame:
    exact = re_table.select(STATE_COLUMNS + ["expected_runs", "observations"]).rename(
        {"expected_runs": "exact_expected_runs", "observations": "exact_observations"}
    )
    keys = ["outs_when_up", "on_1b", "on_2b", "on_3b"]
    base = re_table.group_by(keys).agg(
        base_out_expected_runs=(pl.col("expected_runs") * pl.col("observations")).sum()
        / pl.col("observations").sum()
    )
    league = float(
        re_table.select(
            (pl.col("expected_runs") * pl.col("observations")).sum() / pl.col("observations").sum()
        ).item()
    )
    x = pitches.join(exact, on=STATE_COLUMNS, how="left", nulls_equal=True).join(
        base, on=keys, how="left", nulls_equal=True
    )
    return x.with_columns(
        state_value=pl.when(pl.col("outs_when_up") >= 3)
        .then(0.0)
        .otherwise(
            pl.coalesce(
                "exact_expected_runs", "base_out_expected_runs", pl.lit(league), pl.lit(0.0)
            )
        ),
        state_lookup_tier=pl.when(pl.col("outs_when_up") >= 3)
        .then(pl.lit("terminal"))
        .when(pl.col("exact_expected_runs").is_not_null())
        .then(pl.lit("exact"))
        .when(pl.col("base_out_expected_runs").is_not_null())
        .then(pl.lit("base_out_weighted"))
        .otherwise(pl.lit("league")),
    )
