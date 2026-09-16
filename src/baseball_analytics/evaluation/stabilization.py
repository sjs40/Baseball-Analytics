"""True N-opportunity sequential split-sample stabilization experiments."""

import polars as pl


def stabilization_curve(
    pitches: pl.DataFrame,
    metric: str,
    opportunity: str = "is_two_strike",
    sample_sizes: tuple[int, ...] = (25, 50, 100, 200, 400, 800, 1500),
) -> pl.DataFrame:
    """Correlate each hitter's first N eligible chances with its next N.

    This intentionally is not a qualification-threshold curve: no estimate can
    use more than N observations from either chronological sample.
    """
    required = {"batter", metric, opportunity}
    missing = required - set(pitches.columns)
    if missing:
        raise ValueError(f"Stabilization requires columns: {sorted(missing)}")
    order = [x for x in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if x in pitches.columns]
    eligible = pitches.filter(pl.col(opportunity) == True).sort(["batter", *order]).with_columns(
        _opportunity_index=pl.int_range(0, pl.len()).over("batter")
    )
    rows: list[dict[str, float | int | None]] = []
    for n in sample_sizes:
        samples = eligible.filter(pl.col("_opportunity_index") < 2 * n).group_by("batter").agg(
            available=pl.len(),
            first_n=pl.when(pl.col("_opportunity_index") < n).then(pl.col(metric)).mean(),
            next_n=pl.when(pl.col("_opportunity_index") >= n).then(pl.col(metric)).mean(),
        ).filter(pl.col("available") >= 2 * n)
        players = samples.height
        rows.append({
            "n": n, "opportunities": n, "eligible_players": players, "players": players,
            "pearson": samples.select(pl.corr("first_n", "next_n")).item() if players > 1 else None,
            "spearman": samples.select(pl.corr("first_n", "next_n", method="spearman")).item() if players > 1 else None,
            "opportunities_represented": players * 2 * n,
            "method": "chronological_sequential_first_n_vs_next_n",
        })
    return pl.DataFrame(rows)
