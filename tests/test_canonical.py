import polars as pl

from baseball_analytics.features.canonical import build_canonical_pitches


def test_canonical_orders_pitches_and_uses_only_prior_sequence_context(tmp_path) -> None:
    raw = pl.DataFrame(
        {
            "game_date": ["2025-04-01"] * 5,
            "game_pk": [1] * 5,
            "at_bat_number": [1] * 5,
            "pitch_number": [3, 1, 2, 4, 5],
            "pitcher": [10] * 5,
            "batter": [20] * 5,
            "balls": [0, 0, 0, 0, 0],
            "strikes": [2, 1, 2, 2, 2],
            "description": ["foul", "called_strike", "foul", "ball", "foul"],
            "events": [None] * 5,
            "pitch_type": ["SL", "FF", "CH", "FF", "SL"],
            "release_speed": [83.0, 95.0, 87.0, 94.0, 82.0],
            "plate_x": [0.1] * 5,
            "plate_z": [2.0] * 5,
            "sz_bot": [1.0] * 5,
            "sz_top": [3.0] * 5,
            "stand": ["L"] * 5,
            "p_throws": ["R"] * 5,
        }
    )
    source = tmp_path / "raw.parquet"
    raw.write_parquet(source)
    result = build_canonical_pitches(source, tmp_path / "canonical.parquet")
    assert result.get_column("pitch_number").to_list() == [1, 2, 3, 4, 5]
    assert result.get_column("previous_pitch_type").to_list() == [None, "FF", "CH", "SL", "FF"]
    assert result.get_column("two_strike_foul_number").to_list() == [0, 1, 2, 0, 3]
    assert result.get_column("consecutive_two_strike_foul_count").to_list() == [0, 1, 2, 0, 1]
    assert result.get_column("total_two_strike_fouls_in_pa_so_far").to_list() == [0, 1, 2, 2, 3]


def test_canonical_accepts_incremental_raw_files(tmp_path) -> None:
    columns = {
        "game_date": ["2025-04-01"],
        "game_pk": [1],
        "at_bat_number": [1],
        "pitch_number": [1],
        "pitcher": [10],
        "batter": [20],
        "balls": [0],
        "strikes": [0],
        "description": ["ball"],
    }
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pl.DataFrame(columns).write_parquet(first)
    second_frame = pl.DataFrame({**columns, "game_pk": [2]})
    second_frame.write_parquet(second)
    result = build_canonical_pitches([first, second], tmp_path / "canonical.parquet")
    assert result.height == 2
