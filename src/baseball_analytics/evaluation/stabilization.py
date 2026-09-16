"""Configurable repeated split-sample reliability curves for batter metrics."""

import polars as pl


def stabilization_curve(
    pitches: pl.DataFrame,
    metric: str,
    opportunity: str = "is_two_strike",
    sample_sizes: tuple[int, ...] = (25, 50, 100, 200, 400, 800),
) -> pl.DataFrame:
    """Estimate within-season odd/even-game reliability at increasing opportunities."""
    required = {"batter", "game_pk", metric, opportunity}
    missing = required - set(pitches.columns)
    if missing:
        raise ValueError(f"Stabilization requires columns: {sorted(missing)}")
    eligible = pitches.filter(pl.col(opportunity) == True).with_columns(
        split=(pl.col("game_pk") % 2).cast(pl.String)
    )
    rows: list[dict[str, float | int | None]] = []
    for size in sample_sizes:
        by_split = (
            eligible.group_by(["batter", "split"])
            .agg(value=pl.col(metric).mean(), opportunities=pl.len())
            .filter(pl.col("opportunities") >= size)
            .pivot(on="split", index="batter", values="value")
        )
        if not {"0", "1"}.issubset(by_split.columns):
            rows.append({"opportunities": size, "players": 0, "pearson": None, "spearman": None})
            continue
        rows.append(
            {
                "opportunities": size,
                "players": by_split.height,
                "pearson": by_split.select(pl.corr("0", "1")).item(),
                "spearman": by_split.select(pl.corr("0", "1", method="spearman")).item(),
            }
        )
    return pl.DataFrame(rows)
