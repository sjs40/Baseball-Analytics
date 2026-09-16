from datetime import date
from pathlib import Path

import polars as pl
import typer

from baseball_analytics.data.statcast import ingest_statcast
from baseball_analytics.data.validate import validate_raw_pitches
from baseball_analytics.features.canonical import build_canonical_pitches
from baseball_analytics.metrics.foul.fsv import batter_fsv_leaderboard, compute_fsv
from baseball_analytics.models.run_expectancy import build_run_expectancy
from baseball_analytics.paths import data_path

app = typer.Typer(help="Baseball-Analytics V0.1 research pipeline")


@app.command()
def ingest(
    start: str = typer.Option(..., help="Inclusive ISO date, e.g. 2025-04-01"),
    end: str = typer.Option(..., help="Inclusive ISO date, e.g. 2025-04-30"),
) -> None:
    """Download and cache an inclusive Statcast date range."""
    typer.echo(
        ingest_statcast(date.fromisoformat(start), date.fromisoformat(end), data_path("raw"))
    )


@app.command("build-features")
def build_features(raw: str = typer.Option(..., help="Raw Parquet filename or path")) -> None:
    raw_path = Path(raw) if Path(raw).is_absolute() else data_path("raw") / raw
    result = build_canonical_pitches(raw_path, data_path("interim") / "canonical_pitches.parquet")
    typer.echo(f"Built {result.height:,} canonical pitches")


@app.command("compute-fsv")
def compute_fsv_command() -> None:
    canonical = pl.read_parquet(data_path("interim") / "canonical_pitches.parquet")
    table = build_run_expectancy(canonical, data_path("processed") / "run_expectancy.parquet")
    pitches = compute_fsv(canonical, table, data_path("processed") / "pitch_fsv.parquet")
    leaders = batter_fsv_leaderboard(pitches)
    leaders.write_parquet(data_path("processed") / "batter_fsv_leaderboard.parquet")
    typer.echo(f"Calculated FSV for {pitches.height:,} pitches; {leaders.height:,} batters")


@app.command()
def validate() -> None:
    validate_raw_pitches(pl.read_parquet(data_path("interim") / "canonical_pitches.parquet"))
    typer.echo("Canonical pitch dataset passed structural validation.")


@app.command()
def tui() -> None:
    from apps.tui.app import BaseballTui

    BaseballTui().run()
