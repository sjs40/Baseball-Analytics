import polars as pl

from baseball_analytics.analytics.foul_profiles import batter_foul_splits


def test_persisted_split_dimensions_cover_required_views() -> None:
    pitches = pl.DataFrame(
        {
            "batter": [1],
            "balls": [2],
            "strikes": [2],
            "attack_zone": ["shadow"],
            "pitch_type": ["FF"],
            "p_throws": ["R"],
            "platoon": ["opposite"],
            "is_swing": [True],
            "is_foul": [True],
            "is_two_strike_foul": [True],
            "fsv": [0.1],
            "fave": [0.02],
        }
    )
    result = batter_foul_splits(pitches)
    assert set(result.get_column("split_dimension")) == {
        "attack_zone",
        "pitch_type",
        "pitcher_handedness",
        "platoon",
        "count",
    }
