"""Generate a concise, reproducible descriptive research report from persisted outputs."""

import json
from pathlib import Path

import polars as pl


def _markdown_table(frame: pl.DataFrame, columns: list[str], limit: int = 10) -> str:
    view = frame.select([column for column in columns if column in frame.columns]).head(limit)
    if view.is_empty():
        return "_No qualifying rows._"
    headers = view.columns
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in view.iter_rows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_foul_research_report(
    pitches: pl.DataFrame, profiles: pl.DataFrame, validation_path: Path, output_path: Path
) -> Path:
    """Write a fact-bound report; no claims extend beyond stored sample evidence."""
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    dates = pitches.get_column("game_date").drop_nulls()
    total_pitches = pitches.height
    swings = pitches.filter(pl.col("swing_outcome").is_in(["whiff", "foul", "ball_in_play"])).height
    two_strike_fouls = pitches.get_column("is_two_strike_foul").sum()
    qualified = profiles.filter(pl.col("two_strike_fouls") >= 10)
    fsv = qualified.sort("total_fsv", descending=True)
    fave = qualified.sort("total_fave", descending=True)
    profile = qualified.sort("two_strike_fouls_per_two_strike_swing", descending=True)
    reliability = validation.get("split_half_spoil_difficulty", [{}])[0]
    lines = [
        "# Foul-ball research report — EXPERIMENTAL",
        "",
        "## Scope",
        "",
        f"- Date range: {dates.min()} through {dates.max()}",
        f"- Pitches: {total_pitches:,}",
        f"- Modeled swings: {swings:,}",
        f"- Normal two-strike fouls: {two_strike_fouls:,}",
        "- FSV is neutral run value; FAVE and spoil difficulty remain experimental.",
        "",
        "## Descriptive two-strike foul profiles",
        "",
        _markdown_table(
            profile,
            ["batter_name", "batter", "two_strike_fouls", "two_strike_fouls_per_two_strike_swing"],
        ),
        "",
        "## FSV leaderboard (minimum 10 two-strike fouls)",
        "",
        _markdown_table(
            fsv, ["batter_name", "batter", "total_fsv", "two_strike_fouls", "fsv_per_100_pa"]
        ),
        "",
        "## Experimental FAVE leaderboard (minimum 10 two-strike fouls)",
        "",
        _markdown_table(fave, ["batter_name", "batter", "total_fave", "two_strike_fouls"]),
        "",
        "## Holdout swing-model check",
        "",
        f"- Holdout start: {validation.get('holdout_start')}",
        f"- Holdout multiclass log loss: {validation.get('multiclass_log_loss'):.4f}",
        f"- Naive frequency log loss: {validation.get('naive_frequency_log_loss'):.4f}",
        "- Calibration tables and Brier scores are in `data/processed/foul_validation.json`.",
        "",
        "## Reliability and cautions",
        "",
        f"- Split-half spoil-difficulty Pearson: {reliability.get('pearson')}; Spearman: {reliability.get('spearman')}.",
        "- The short one-month sample yields weak FSV/FAVE split-half reliability; do not interpret these leaderboards as established player skill.",
        "- The model is intentionally simple, and fair-ball value uses conditional mean Statcast delta RE. This is a starting benchmark, not a final player model.",
        "",
        "## Recommended next step",
        "",
        "Ingest multiple complete seasons, repeat chronological holdouts, then evaluate stabilization before setting qualification thresholds or interpreting year-over-year rankings.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
