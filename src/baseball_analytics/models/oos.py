"""Chronological out-of-sample swing prediction ledger."""

from pathlib import Path

import polars as pl

from baseball_analytics.models.swing_outcome import (
    TARGETS,
    predict_swing_outcomes,
    train_swing_outcome_baseline,
)


def build_oos_prediction_ledger(
    pitches: pl.DataFrame, output_path: Path, model_dir: Path, alpha: float = 20.0
) -> pl.DataFrame:
    """Score each season only with observations strictly before that season.

    The first available season is intentionally omitted because it has no
    historical training window.  The resulting ledger is the sole permitted
    input to residual-skill aggregation.
    """
    if "game_date" not in pitches.columns or "pitch_id" not in pitches.columns:
        raise ValueError("OOS ledger requires stable pitch_id and game_date")
    swings = pitches.filter(pl.col("swing_outcome").is_in(TARGETS)).with_columns(
        _year=pl.col("game_date").cast(pl.String).str.slice(0, 4).cast(pl.Int32)
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    ledgers: list[pl.DataFrame] = []
    for year in sorted(swings.get_column("_year").unique().to_list()):
        prediction = swings.filter(pl.col("_year") == year)
        training = swings.filter(pl.col("_year") < year)
        if training.is_empty():
            continue
        train_end = str(training.get_column("game_date").max())
        model = train_swing_outcome_baseline(training, model_dir / f"swing_model_through_{year - 1}.parquet", alpha=alpha)
        scored = predict_swing_outcomes(prediction, model).select(
            ["pitch_id", "game_date", "batter", "pitcher", "swing_outcome", "model_version"]
            + [f"expected_{target}_probability" for target in TARGETS]
        ).with_columns(
            training_end=pl.lit(train_end), prediction_window=pl.lit(str(year)),
        )
        if scored.filter(pl.col("game_date").cast(pl.String) <= pl.col("training_end")).height:
            raise AssertionError("OOS ledger contains an observation in its training period")
        ledgers.append(scored)
    if not ledgers:
        raise ValueError("OOS scoring needs at least two chronological seasons")
    ledger = pl.concat(ledgers).sort("game_date")
    if ledger.get_column("pitch_id").n_unique() != ledger.height:
        raise ValueError("OOS ledger has duplicate pitch predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_parquet(output_path)
    return ledger
