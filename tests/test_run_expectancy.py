import polars as pl
import pytest

from baseball_analytics.models.run_expectancy import build_run_expectancy, lookup_state_value


def test_lookup_uses_weighted_base_out_fallback_and_terminal_zero() -> None:
    table = pl.DataFrame(
        {
            "outs_when_up": [0, 0],
            "on_1b": [None, None],
            "on_2b": [None, None],
            "on_3b": [None, None],
            "balls": [0, 1],
            "strikes": [0, 0],
            "expected_runs": [0.2, 0.8],
            "observations": [3, 1],
        }
    )
    pitches = pl.DataFrame(
        {
            "outs_when_up": [0, 3],
            "on_1b": [None, None],
            "on_2b": [None, None],
            "on_3b": [None, None],
            "balls": [2, 0],
            "strikes": [1, 0],
        }
    )
    result = lookup_state_value(pitches, table)
    assert result["state_lookup_tier"].to_list() == ["base_out_weighted", "terminal"]
    assert result["state_value"].to_list() == pytest.approx([0.35, 0.0])


def test_run_expectancy_rejects_impossible_counts(tmp_path) -> None:
    pitches = pl.DataFrame(
        {
            "game_pk": [1],
            "inning": [1],
            "inning_topbot": ["Top"],
            "bat_score": [0],
            "outs_when_up": [0],
            "on_1b": [None],
            "on_2b": [None],
            "on_3b": [None],
            "balls": [4],
            "strikes": [0],
        }
    )
    with pytest.raises(ValueError, match="Impossible count"):
        build_run_expectancy(pitches, tmp_path / "re.parquet")
