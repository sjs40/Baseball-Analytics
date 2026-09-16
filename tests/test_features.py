import polars as pl

from baseball_analytics.features.foul import swing_outcome
from baseball_analytics.features.pitch_location import add_location_features


def test_swing_classification_preserves_ambiguity() -> None:
    assert swing_outcome("swinging_strike") == "whiff"
    assert swing_outcome("foul") == "foul"
    assert swing_outcome("hit_into_play") == "ball_in_play"
    assert swing_outcome("blocked_ball") == "ambiguous"


def test_open_attack_zone_boundaries_and_missing_location() -> None:
    pitches = pl.DataFrame(
        {
            "plate_x": [0.0, 1.0, 1.3, 3.0, None],
            "plate_z": [2.0, 2.0, 2.0, 2.0, None],
            "sz_bot": [1.0] * 5,
            "sz_top": [3.0] * 5,
        }
    )
    result = add_location_features(pitches)
    assert result["attack_zone"].to_list() == ["heart", "shadow", "chase", "waste", "unknown"]
    assert result["in_rulebook_zone"].to_list()[:2] == [True, False]
