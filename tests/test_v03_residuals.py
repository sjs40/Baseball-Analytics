import polars as pl
import pytest

from baseball_analytics.metrics.foul.fae import (
    add_residuals,
    batter_residual_leaderboard,
    empirical_bayes_shrinkage,
)
from baseball_analytics.metrics.foul.fsv import batter_fsv_leaderboard
from baseball_analytics.models.run_expectancy import build_run_expectancy
from baseball_analytics.models.swing_outcome import (
    predict_swing_outcomes,
    train_swing_outcome_baseline,
)


def test_smoothed_sparse_cell_probabilities_sum_to_one(tmp_path) -> None:
    pitches = pl.DataFrame({"balls": [0, 0], "strikes": [0, 0], "attack_zone": ["heart", "heart"], "swing_outcome": ["whiff", "foul"]})
    model = train_swing_outcome_baseline(pitches, tmp_path / "model.parquet", alpha=5)
    scored = predict_swing_outcomes(pitches, model)
    assert scored.select((pl.col("expected_whiff_probability") + pl.col("expected_foul_probability") + pl.col("expected_ball_in_play_probability")).min()).item() == pytest.approx(1.0)


def test_fsv_rate_uses_two_strike_pitch_denominator() -> None:
    leaders = batter_fsv_leaderboard(pl.DataFrame({"batter": [1, 1, 1, 1], "fsv": [2.0, 0.0, 0.0, 0.0], "is_two_strike": [True, True, False, False], "is_two_strike_foul": [True, False, False, False], "is_foul": [True, False, False, False]}))
    assert leaders["fsv_per_100_two_strike_pitches"].item() == pytest.approx(100.0)


def test_re_uses_final_post_pitch_score(tmp_path) -> None:
    pitches = pl.DataFrame({"game_pk": [1, 1], "inning": [1, 1], "inning_topbot": ["Top", "Top"], "bat_score": [0, 0], "post_bat_score": [0, 2], "outs_when_up": [0, 2], "on_1b": [None, None], "on_2b": [None, None], "on_3b": [None, None], "balls": [0, 0], "strikes": [0, 0]})
    table = build_run_expectancy(pitches, tmp_path / "re.parquet")
    assert table["expected_runs"].to_list() == pytest.approx([2.0, 2.0])
    assert table["re_version"].unique().item() == "0.3-terminal-score-aware"


def test_fae_aggregation_and_small_sample_shrinkage() -> None:
    pitches = pl.DataFrame({"pitch_id": ["a", "b", "c"], "game_date": ["2024-01-01"] * 3, "batter": [1, 1, 2], "pitcher": [9, 9, 9], "swing_outcome": ["foul", "whiff", "foul"], "is_two_strike": [True, False, True]})
    ledger = pl.DataFrame({"pitch_id": ["a", "b", "c"], "game_date": ["2024-01-01"] * 3, "batter": [1, 1, 2], "pitcher": [9, 9, 9], "expected_foul_probability": [.2, .4, .5], "expected_whiff_probability": [.3, .4, .3], "expected_ball_in_play_probability": [.5, .2, .2], "swing_outcome": ["foul", "whiff", "foul"], "model_version": ["x"] * 3, "training_end": ["2023-12-31"] * 3, "prediction_window": ["2024"] * 3})
    result = batter_residual_leaderboard(add_residuals(pitches, ledger))
    assert result.filter(pl.col("batter") == 1)["fouls_above_expected"].item() == pytest.approx(.4)
    shrunk = empirical_bayes_shrinkage(result)
    assert "shrunk_residual_rate" in shrunk.columns
