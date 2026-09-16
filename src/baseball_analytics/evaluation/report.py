"""Persisted validation artifacts for transparent model review."""

import json
import math
from pathlib import Path

import polars as pl

from baseball_analytics.evaluation.calibration import brier_score, calibration_table
from baseball_analytics.evaluation.reliability import split_half_reliability
from baseball_analytics.evaluation.stabilization import stabilization_curve

TARGETS = ["whiff", "foul", "ball_in_play"]


def swing_validation_report(
    pitches: pl.DataFrame, output_path: Path, holdout_start: str | None = None
) -> dict[str, object]:
    """Write calibration, loss, frequency, and split-half reliability evidence."""
    swings = pitches.filter(pl.col("swing_outcome").is_in(TARGETS))
    if holdout_start is not None:
        swings = swings.filter(pl.col("game_date") >= holdout_start)
    if swings.is_empty():
        raise ValueError("No modeled swings are available for validation")
    frequencies = {
        row["swing_outcome"]: row["observations"]
        for row in swings.group_by("swing_outcome").agg(observations=pl.len()).to_dicts()
    }
    total = sum(frequencies.values())
    naive_probabilities = {target: frequencies.get(target, 0) / total for target in TARGETS}
    log_loss = swings.select(
        (
            -pl.when(pl.col("swing_outcome") == "whiff")
            .then(pl.col("expected_whiff_probability"))
            .when(pl.col("swing_outcome") == "foul")
            .then(pl.col("expected_foul_probability"))
            .otherwise(pl.col("expected_ball_in_play_probability"))
            .clip(lower_bound=1e-12)
            .log()
        ).mean()
    ).item()
    report: dict[str, object] = {
        "model_version": swings.get_column("model_version").unique().to_list(),
        "modeled_swings": swings.height,
        "class_frequencies": frequencies,
        "multiclass_log_loss": log_loss,
        "naive_frequency_log_loss": -sum(
            probability * math.log(probability)
            for probability in naive_probabilities.values()
            if probability > 0
        ),
        "holdout_start": holdout_start,
        "brier_scores": {target: brier_score(swings, target) for target in TARGETS},
        "calibration": {target: calibration_table(swings, target).to_dicts() for target in TARGETS},
    }
    for metric in ["fsv", "fave", "spoil_difficulty"]:
        if metric in pitches.columns:
            report[f"split_half_{metric}"] = split_half_reliability(pitches, metric).to_dicts()
            report[f"stabilization_{metric}"] = stabilization_curve(pitches, metric).to_dicts()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
