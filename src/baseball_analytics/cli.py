from datetime import date
from pathlib import Path

import polars as pl
import typer

from baseball_analytics.analytics.foul_profiles import batter_foul_profiles, batter_foul_splits
from baseball_analytics.analytics.research_report import build_foul_research_report
from baseball_analytics.data.players import build_batter_lookup
from baseball_analytics.data.statcast import ingest_statcast
from baseball_analytics.data.validate import validate_canonical_pitches
from baseball_analytics.evaluation.report import swing_validation_report
from baseball_analytics.features.canonical import build_canonical_pitches
from baseball_analytics.metrics.foul.fave import compute_fave
from baseball_analytics.metrics.foul.fsv import batter_fsv_leaderboard, compute_fsv
from baseball_analytics.models.run_expectancy import build_run_expectancy
from baseball_analytics.models.swing_outcome import train_swing_outcome_baseline
from baseball_analytics.paths import data_path

app = typer.Typer(help="Baseball-Analytics V0.2 foul-ball research pipeline")


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
def build_features(
    raw: list[str] = typer.Option(
        ..., help="Raw Parquet filename or path; repeat --raw for increments"
    ),
) -> None:
    raw_paths = [
        Path(value) if Path(value).is_absolute() else data_path("raw") / value for value in raw
    ]
    missing = [str(path) for path in raw_paths if not path.exists()]
    if missing:
        raise typer.BadParameter(f"Raw Parquet input not found: {', '.join(missing)}")
    result = build_canonical_pitches(raw_paths, data_path("interim") / "canonical_pitches.parquet")
    typer.echo(f"Built {result.height:,} canonical pitches")


@app.command("compute-fsv")
def compute_fsv_command() -> None:
    canonical = pl.read_parquet(data_path("interim") / "canonical_pitches.parquet")
    table = build_run_expectancy(canonical, data_path("processed") / "run_expectancy.parquet")
    pitches = compute_fsv(canonical, table, data_path("processed") / "pitch_fsv.parquet")
    lookup_path = data_path("interim") / "batter_lookup.parquet"
    lookup = pl.read_parquet(lookup_path) if lookup_path.exists() else None
    leaders = batter_fsv_leaderboard(pitches, lookup)
    leaders.write_parquet(data_path("processed") / "batter_fsv_leaderboard.parquet")
    typer.echo(f"Calculated FSV for {pitches.height:,} pitches; {leaders.height:,} batters")


@app.command("build-re")
def build_re_command() -> None:
    canonical = pl.read_parquet(data_path("interim") / "canonical_pitches.parquet")
    result = build_run_expectancy(canonical, data_path("processed") / "run_expectancy.parquet")
    typer.echo(f"Built {result.height:,} count-aware RE states")


@app.command("build-player-map")
def build_player_map() -> None:
    canonical = pl.read_parquet(data_path("interim") / "canonical_pitches.parquet")
    result = build_batter_lookup(canonical, data_path("interim") / "batter_lookup.parquet")
    typer.echo(f"Resolved {result.height:,} batter identities")


@app.command("train-swing-model")
def train_swing_model(
    train_end: str | None = typer.Option(
        None, help="Optional inclusive ISO training cutoff for chronological validation"
    ),
) -> None:
    canonical = pl.read_parquet(data_path("interim") / "canonical_pitches.parquet")
    result = train_swing_outcome_baseline(
        canonical,
        data_path("processed") / "swing_outcome_baseline.parquet",
        date.fromisoformat(train_end) if train_end else None,
    )
    typer.echo(f"Trained empirical baseline on {result['observations'].sum():,} swings")


@app.command("compute-fave")
def compute_fave_command() -> None:
    pitches = pl.read_parquet(data_path("processed") / "pitch_fsv.parquet")
    model = pl.read_parquet(data_path("processed") / "swing_outcome_baseline.parquet")
    result = compute_fave(pitches, model, data_path("processed") / "pitch_fave.parquet")
    typer.echo(f"Calculated experimental FAVE for {result.height:,} pitches")


@app.command("build-foul-profiles")
def build_foul_profiles() -> None:
    path = data_path("processed") / "pitch_fave.parquet"
    pitches = pl.read_parquet(
        path if path.exists() else data_path("processed") / "pitch_fsv.parquet"
    )
    result = batter_foul_profiles(pitches)
    lookup_path = data_path("interim") / "batter_lookup.parquet"
    if lookup_path.exists():
        result = result.join(pl.read_parquet(lookup_path), on="batter", how="left")
    result.write_parquet(data_path("processed") / "batter_foul_profiles.parquet")
    batter_foul_splits(pitches).write_parquet(data_path("processed") / "batter_foul_splits.parquet")
    typer.echo(f"Built {result.height:,} batter-season foul profiles")


@app.command("validate-foul")
def validate_foul(
    holdout_start: str | None = typer.Option(
        None, help="Optional inclusive ISO holdout start; use after --train-end"
    ),
) -> None:
    pitches = pl.read_parquet(data_path("processed") / "pitch_fave.parquet")
    report = swing_validation_report(
        pitches, data_path("processed") / "foul_validation.json", holdout_start
    )
    typer.echo(f"Wrote validation report for {report['modeled_swings']:,} modeled swings")


@app.command("report-foul")
def report_foul() -> None:
    pitches = pl.read_parquet(data_path("processed") / "pitch_fave.parquet")
    profiles = pl.read_parquet(data_path("processed") / "batter_foul_profiles.parquet")
    result = build_foul_research_report(
        pitches,
        profiles,
        data_path("processed") / "foul_validation.json",
        Path("docs/research/foul_report.md"),
    )
    typer.echo(f"Wrote research report: {result}")


@app.command()
def validate() -> None:
    validate_canonical_pitches(pl.read_parquet(data_path("interim") / "canonical_pitches.parquet"))
    typer.echo("Canonical pitch dataset passed structural validation.")


@app.command()
def tui() -> None:
    from baseball_analytics.ui.tui import BaseballTui

    BaseballTui().run()
