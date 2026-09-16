import polars as pl
import pytest

from baseball_analytics.metrics.foul.fave import compute_fave


def test_fave_uses_probability_weighted_delta_re_components(tmp_path) -> None:
    pitches = pl.DataFrame(
        {
            "balls": [1],
            "strikes": [2],
            "attack_zone": ["heart"],
            "pre_pitch_state_value": [0.5],
            "actual_state_value": [0.4],
            "whiff_state_value": [0.0],
            "is_foul": [True],
            "is_foul_bunt": [False],
            "is_terminal_foul_tip": [False],
            "is_two_strike_foul": [True],
        }
    )
    model = pl.DataFrame(
        {
            "balls": [1, 1, 1],
            "strikes": [2, 2, 2],
            "attack_zone": ["heart"] * 3,
            "swing_outcome": ["whiff", "foul", "ball_in_play"],
            "observations": [2, 3, 5],
            "probability": [0.2, 0.3, 0.5],
            "mean_delta_run_exp": [-0.5, -0.1, 0.2],
            "total": [10, 10, 10],
            "model_version": ["test"] * 3,
        }
    )
    result = compute_fave(pitches, model, tmp_path / "fave.parquet")
    # Actual foul delta is -.1; expected = .2*-.5 + .3*-.1 + .5*.2 = -.03.
    assert result["fave"].item() == pytest.approx(-0.07)
    assert result["spoil_difficulty"].item() == 0.2
