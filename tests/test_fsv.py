import polars as pl

from baseball_analytics.metrics.foul.fsv import compute_fsv


def test_only_normal_two_strike_fouls_receive_fsv(tmp_path):
    pitches = pl.DataFrame(
        {
            "game_pk": [1, 1, 1],
            "at_bat_number": [1, 2, 3],
            "pitch_number": [1, 1, 1],
            "outs_when_up": [0, 0, 0],
            "on_1b": [None, None, None],
            "on_2b": [None, None, None],
            "on_3b": [None, None, None],
            "balls": [1, 1, 1],
            "strikes": [2, 1, 2],
            "is_two_strike_foul": [True, False, True],
            "is_foul_bunt": [False, False, True],
        }
    )
    re = pl.DataFrame(
        {
            "outs_when_up": [0],
            "on_1b": [None],
            "on_2b": [None],
            "on_3b": [None],
            "balls": [1],
            "strikes": [2],
            "expected_runs": [0.33],
            "observations": [10],
        }
    )
    assert compute_fsv(pitches, re, tmp_path / "fsv.parquet")["fsv"].to_list() == [0.33, 0.0, 0.0]
