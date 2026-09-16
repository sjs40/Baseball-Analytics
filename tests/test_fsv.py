import polars as pl

from baseball_analytics.metrics.foul.fsv import compute_fsv


def test_only_normal_two_strike_fouls_receive_fsv(tmp_path):
    pitches = pl.DataFrame(
        {
            "game_pk": [1, 1, 1],
            "at_bat_number": [1, 2, 3],
            "pitch_number": [1, 1, 1],
            "outs_when_up": [0, 0, 0],
            "on_1b": [1, 1, 1],
            "on_2b": [None, None, None],
            "on_3b": [None, None, None],
            "balls": [1, 1, 1],
            "strikes": [2, 1, 2],
            "is_two_strike_foul": [True, False, True],
            "is_foul": [True, False, True],
            "is_foul_bunt": [False, False, True],
        }
    )
    re = pl.DataFrame(
        {
            "outs_when_up": [0, 1],
            "on_1b": [1, 1],
            "on_2b": [None, None],
            "on_3b": [None, None],
            "balls": [1, 0],
            "strikes": [2, 0],
            "expected_runs": [0.80, 0.25],
            "observations": [10, 10],
        }
    )
    result = compute_fsv(pitches, re, tmp_path / "fsv.parquet")
    assert result["fsv"].to_list() == [0.55, 0.0, 0.0]
    assert result["actual_state_value"].item(0) == 0.80
    assert result["strikeout_counterfactual_state_value"].item(0) == 0.25
    assert {
        "fsv",
        "fsv_version",
        "actual_state_value",
        "strikeout_counterfactual_state_value",
        "actual_lookup_tier",
    } <= set(result.columns)


def test_inning_ending_strikeout_counterfactual_has_zero_value(tmp_path):
    pitches = pl.DataFrame(
        {
            "game_pk": [1],
            "at_bat_number": [1],
            "pitch_number": [1],
            "outs_when_up": [2],
            "on_1b": [None],
            "on_2b": [None],
            "on_3b": [None],
            "balls": [0],
            "strikes": [2],
            "is_two_strike_foul": [True],
            "is_foul": [True],
            "is_foul_bunt": [False],
        }
    )
    re = pl.DataFrame(
        {
            "outs_when_up": [2],
            "on_1b": [None],
            "on_2b": [None],
            "on_3b": [None],
            "balls": [0],
            "strikes": [2],
            "expected_runs": [0.2],
            "observations": [10],
        }
    )
    result = compute_fsv(pitches, re, tmp_path / "fsv.parquet")
    assert result["strikeout_counterfactual_state_value"].item() == 0.0
    assert result["fsv"].item() == 0.2


def test_terminal_foul_tip_is_excluded(tmp_path):
    pitches = pl.DataFrame(
        {
            "game_pk": [1],
            "at_bat_number": [1],
            "pitch_number": [1],
            "outs_when_up": [0],
            "on_1b": [None],
            "on_2b": [None],
            "on_3b": [None],
            "balls": [0],
            "strikes": [2],
            "is_two_strike_foul": [True],
            "is_foul": [False],
            "is_foul_bunt": [False],
            "is_terminal_foul_tip": [True],
        }
    )
    re = pl.DataFrame(
        {
            "outs_when_up": [0, 1],
            "on_1b": [None, None],
            "on_2b": [None, None],
            "on_3b": [None, None],
            "balls": [0, 0],
            "strikes": [2, 0],
            "expected_runs": [0.5, 0.2],
            "observations": [10, 10],
        }
    )
    assert compute_fsv(pitches, re, tmp_path / "tip.parquet")["fsv"].item() == 0.0
