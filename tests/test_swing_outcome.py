from datetime import date

import polars as pl

from baseball_analytics.models.swing_outcome import train_swing_outcome_baseline


def test_training_cutoff_excludes_future_swings(tmp_path) -> None:
    pitches = pl.DataFrame(
        {
            "game_date": ["2025-04-01", "2025-04-01", "2025-04-20"],
            "balls": [0, 0, 0],
            "strikes": [0, 0, 0],
            "attack_zone": ["heart", "heart", "heart"],
            "swing_outcome": ["whiff", "foul", "ball_in_play"],
            "delta_run_exp": [-0.1, 0.0, 0.4],
        }
    )
    model = train_swing_outcome_baseline(pitches, tmp_path / "model.parquet", date(2025, 4, 1))
    assert model.get_column("observations").sum() == 2
    assert model.get_column("train_end").unique().to_list() == ["2025-04-01"]
