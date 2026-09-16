import polars as pl

from baseball_analytics.analytics.foul_profiles import batter_foul_profiles


def test_fsv_per_100_pa_uses_all_plate_appearances() -> None:
    pitches = pl.DataFrame(
        {
            "batter": [1, 1, 1],
            "game_year": [2025] * 3,
            "game_pk": [1, 1, 1],
            "at_bat_number": [1, 1, 2],
            "is_two_strike": [True, True, False],
            "is_swing": [True, True, False],
            "is_foul": [True, False, False],
            "is_two_strike_foul": [True, False, False],
            "two_strike_foul_number": [1, 0, 0],
            "fsv": [1.0, 0.0, 0.0],
        }
    )
    profile = batter_foul_profiles(pitches)
    assert profile["total_pas"].item() == 2
    assert profile["fsv_per_100_pa"].item() == 50.0
