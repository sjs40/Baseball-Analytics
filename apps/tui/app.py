from pathlib import Path

import polars as pl
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static


class BaseballTui(App[None]):
    """Read-only V0.1 FSV leaderboard; analytics live in the package."""

    TITLE = "Baseball Analytics — FSV"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Neutral FSV leaderboard (two-strike normal fouls only)", id="subtitle")
        yield DataTable(id="leaders")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#leaders", DataTable)
        table.add_columns("Batter", "FSV", "2-strike fouls", "Total fouls", "Pitches")
        path = Path("data/processed/batter_fsv_leaderboard.parquet")
        if not path.exists():
            self.query_one("#subtitle", Static).update("Run `baseball compute-fsv` first.")
            return
        for row in pl.read_parquet(path).head(50).iter_rows(named=True):
            table.add_row(
                str(row.get("player_name") or row.get("batter")),
                f"{row['fsv']:.3f}",
                str(row["two_strike_fouls"]),
                str(row["total_fouls"]),
                str(row["pitches"]),
            )


if __name__ == "__main__":
    BaseballTui().run()
