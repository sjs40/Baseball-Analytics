"""Textual research interface backed exclusively by persisted package outputs."""

import polars as pl
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane

from baseball_analytics.analytics.explorers import pa_pitches
from baseball_analytics.analytics.leaderboards import METRICS, foul_leaderboard
from baseball_analytics.paths import data_path


class BaseballTui(App[None]):
    """Read-only batter-centric leaderboard, profile, and PA research views."""

    TITLE = "Baseball Analytics — Foul Research"

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Foul Leaderboard", id="leaders-tab"):
                yield Static("Batter-centric FSV leaderboard", id="leader-subtitle")
                yield Input(value="fsv", placeholder="Metric key", id="metric-input")
                yield DataTable(id="leaders")
            with TabPane("Player Profile", id="profile-tab"):
                yield Input(placeholder="Batter MLBAM ID", id="batter-input")
                yield Static("Select a batter ID from the leaderboard.", id="profile-subtitle")
                yield DataTable(id="profile")
                yield Static("Persisted count, zone, pitch-type, handedness, and platoon splits")
                yield DataTable(id="profile-splits")
            with TabPane("PA Explorer", id="pa-tab"):
                yield Input(placeholder="game_pk-at_bat_number", id="pa-input")
                with VerticalScroll():
                    yield Static("Select a plate appearance.", id="pa-subtitle")
                    yield DataTable(id="pa")
        yield Footer()

    def on_mount(self) -> None:
        self.leaderboard = self._read("batter_fsv_leaderboard.parquet")
        self.profiles = self._read("batter_foul_profiles.parquet")
        self.splits = self._read("batter_foul_splits.parquet")
        self.pitches = self._read("pitch_fave.parquet", fallback="pitch_fsv.parquet")
        self.lookup = self._read_lookup()
        self._load_leaders()
        if self.leaderboard is not None and not self.leaderboard.is_empty():
            self.query_one("#batter-input", Input).value = str(
                self.leaderboard.get_column("batter").item(0)
            )
            self._load_profile(str(self.leaderboard.get_column("batter").item(0)))
        if self.pitches is not None and not self.pitches.is_empty():
            candidate = self.pitches.filter(pl.col("is_two_strike_foul")).head(1)
            if not candidate.is_empty():
                pa_key = f"{candidate['game_pk'].item(0)}-{candidate['at_bat_number'].item(0)}"
                self.query_one("#pa-input", Input).value = pa_key
                self._load_pa(pa_key)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "metric-input" and event.value in METRICS:
            self._load_leaders(event.value)
        if event.input.id == "batter-input" and event.value:
            self._load_profile(event.value)
        if event.input.id == "pa-input" and event.value:
            self._load_pa(event.value)

    def _read(self, filename: str, fallback: str | None = None) -> pl.DataFrame | None:
        path = data_path("processed") / filename
        if not path.exists() and fallback:
            path = data_path("processed") / fallback
        return pl.read_parquet(path) if path.exists() else None

    def _read_lookup(self) -> pl.DataFrame | None:
        path = data_path("interim") / "batter_lookup.parquet"
        return pl.read_parquet(path) if path.exists() else None

    @staticmethod
    def _replace(table: DataTable, frame: pl.DataFrame, columns: list[str]) -> None:
        table.clear(columns=True)
        selected = [column for column in columns if column in frame.columns]
        table.add_columns(*selected)
        for row in frame.select(selected).iter_rows():
            table.add_row(*(str(value) if value is not None else "" for value in row))

    def _load_leaders(self, metric: str = "fsv") -> None:
        table = self.query_one("#leaders", DataTable)
        if self.leaderboard is None or self.pitches is None:
            self.query_one("#leader-subtitle", Static).update("Run `baseball compute-fsv` first.")
            return
        leaders = foul_leaderboard(self.pitches, metric, batter_lookup=self.lookup)
        self.query_one("#leader-subtitle", Static).update(f"Batter-centric leaderboard — {metric}")
        self._replace(
            table,
            leaders.head(50),
            [
                "batter_name",
                "batter",
                METRICS[metric],
                "opportunities",
                "two_strike_fouls",
            ],
        )

    def _load_profile(self, batter_text: str) -> None:
        if self.profiles is None:
            self.query_one("#profile-subtitle", Static).update(
                "Run `baseball build-foul-profiles` first."
            )
            return
        try:
            batter = int(batter_text)
        except ValueError:
            return
        profile = self.profiles.filter(pl.col("batter") == batter)
        self.query_one("#profile-subtitle", Static).update(
            "No profile found." if profile.is_empty() else f"Batter {batter} foul profile"
        )
        self._replace(self.query_one("#profile", DataTable), profile, profile.columns)
        if self.splits is not None:
            splits = self.splits.filter(pl.col("batter") == batter)
            self._replace(
                self.query_one("#profile-splits", DataTable),
                splits,
                [
                    "split_dimension",
                    "split_value",
                    "pitches",
                    "fouls",
                    "two_strike_fouls",
                    "total_fsv",
                    "total_fave",
                ],
            )

    def _load_pa(self, key: str) -> None:
        if self.pitches is None or "-" not in key:
            return
        try:
            game_pk, at_bat_number = (int(value) for value in key.split("-", maxsplit=1))
        except ValueError:
            return
        pa = pa_pitches(self.pitches, game_pk, at_bat_number)
        self.query_one("#pa-subtitle", Static).update(
            "No pitches found." if pa.is_empty() else f"PA {game_pk}-{at_bat_number}"
        )
        self._replace(
            self.query_one("#pa", DataTable),
            pa,
            [
                "pitch_number",
                "balls",
                "strikes",
                "pitch_type",
                "release_speed",
                "plate_x",
                "plate_z",
                "attack_zone",
                "description",
                "swing_outcome",
                "fsv",
                "expected_whiff_probability",
                "expected_foul_probability",
                "expected_ball_in_play_probability",
                "fave",
            ],
        )
