"""Transparent, chronologically configurable empirical swing-outcome baseline."""

from datetime import date
from pathlib import Path

import polars as pl

MODEL_VERSION = "0.2-smoothed-empirical-count-attack-zone"
TARGETS = ["whiff", "foul", "ball_in_play"]
FEATURES = ["balls", "strikes", "attack_zone"]


def _eligible_swings(pitches: pl.DataFrame) -> pl.DataFrame:
    required = {"swing_outcome", *FEATURES}
    missing = required - set(pitches.columns)
    if missing:
        raise ValueError(f"Swing model requires columns: {sorted(missing)}")
    swings = pitches.filter(pl.col("swing_outcome").is_in(TARGETS))
    if swings.is_empty():
        raise ValueError("No unambiguous whiff/foul/ball-in-play swings available")
    return swings


def train_swing_outcome_baseline(
    pitches: pl.DataFrame, output_path: Path, train_end: date | None = None, alpha: float = 20.0
) -> pl.DataFrame:
    """Fit Model B with prior-weighted multinomial smoothing.

    Each count×attack-zone cell has all three classes and probabilities sum to
    one: ``(n_cell,k + alpha * P_global,k) / (n_cell + alpha)``.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    swings = _eligible_swings(pitches)
    if train_end is not None:
        if "game_date" not in swings.columns:
            raise ValueError("Chronological training requires game_date")
        cutoff: date | str = (
            train_end.isoformat()
            if swings.schema["game_date"] in {pl.String, pl.Utf8}
            else train_end
        )
        swings = swings.filter(pl.col("game_date") <= cutoff)
    if swings.is_empty():
        raise ValueError("No swings remain in the requested training period")
    observed = (
        swings.group_by(FEATURES + ["swing_outcome"])
        .agg(
            observations=pl.len(),
            mean_delta_run_exp=pl.col("delta_run_exp").mean()
            if "delta_run_exp" in swings.columns
            else pl.lit(None, dtype=pl.Float64),
        )
    )
    globals_ = swings.group_by("swing_outcome").agg(global_observations=pl.len())
    total_global = swings.height
    global_grid = pl.DataFrame({"swing_outcome": TARGETS}).join(globals_, on="swing_outcome", how="left")
    global_grid = global_grid.with_columns(
        global_observations=pl.col("global_observations").fill_null(0),
        global_probability=pl.col("global_observations").fill_null(0) / total_global,
    )
    cells = swings.select(FEATURES).unique()
    table = (
        cells.join(global_grid, how="cross")
        .join(observed, on=FEATURES + ["swing_outcome"], how="left")
        .with_columns(
            observations=pl.col("observations").fill_null(0),
            total=pl.col("observations").fill_null(0).sum().over(FEATURES),
            probability=(pl.col("observations").fill_null(0) + alpha * pl.col("global_probability"))
            / (pl.col("observations").fill_null(0).sum().over(FEATURES) + alpha),
            smoothing_alpha=pl.lit(alpha),
            model_version=pl.lit(MODEL_VERSION),
            train_end=pl.lit(train_end.isoformat() if train_end else None, dtype=pl.String),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)
    return table


def predict_swing_outcomes(pitches: pl.DataFrame, model: pl.DataFrame) -> pl.DataFrame:
    """Attach probabilities and BIP delta-RE fallback values to every pitch."""
    probabilities = model.pivot(
        on="swing_outcome", index=FEATURES, values="probability", aggregate_function="first"
    ).rename(
        {target: f"expected_{target}_probability" for target in TARGETS}
    )
    bip_values = model.filter(pl.col("swing_outcome") == "ball_in_play").select(
        FEATURES + [pl.col("mean_delta_run_exp").alias("expected_bip_delta_re")]
    )
    result = pitches.join(probabilities, on=FEATURES, how="left").join(
        bip_values, on=FEATURES, how="left"
    )
    frequency = model.group_by("swing_outcome").agg(
        probability=pl.col("observations").sum() / pl.col("observations").sum()
    )
    fallback = {row["swing_outcome"]: row["probability"] for row in frequency.to_dicts()}
    bip_fallback = (
        model.filter(pl.col("swing_outcome") == "ball_in_play")
        .select(pl.col("mean_delta_run_exp").mean())
        .item()
    )
    bip_fallback = 0.0 if bip_fallback is None else bip_fallback
    result = result.with_columns(
        *[
            pl.col(f"expected_{target}_probability")
            .fill_null(fallback.get(target, 0.0))
            .alias(f"expected_{target}_probability")
            for target in TARGETS
        ],
        pl.col("expected_bip_delta_re").fill_null(bip_fallback).alias("expected_bip_delta_re"),
        model_version=pl.lit(MODEL_VERSION),
    )
    probability_columns = [f"expected_{target}_probability" for target in TARGETS]
    sums = result.select(sum(pl.col(column) for column in probability_columns).alias("sum"))
    if sums.filter((pl.col("sum") - 1.0).abs() > 1e-9).height:
        raise AssertionError("Swing outcome probabilities must sum to one")
    return result
