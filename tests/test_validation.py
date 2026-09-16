import polars as pl
import pytest

from baseball_analytics.data.validate import validate_canonical_pitches


def test_canonical_validation_rejects_impossible_count() -> None:
    frame = pl.DataFrame(
        {
            "game_pk": [1],
            "at_bat_number": [1],
            "pitch_number": [1],
            "balls": [4],
            "strikes": [0],
            "description": ["ball"],
            "pitch_id": ["1-1-1"],
            "pa_id": ["1-1"],
            "is_swing": [False],
            "is_foul": [False],
            "is_two_strike_foul": [False],
            "attack_zone": ["heart"],
        }
    )
    with pytest.raises(ValueError, match="impossible counts"):
        validate_canonical_pitches(frame)
