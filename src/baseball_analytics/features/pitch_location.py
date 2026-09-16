"""Open geometric attack-zone approximation, not a proprietary classification."""

import polars as pl


def add_location_features(frame: pl.DataFrame) -> pl.DataFrame:
    plate_x = pl.col("plate_x") if "plate_x" in frame.columns else pl.lit(None)
    plate_z = pl.col("plate_z") if "plate_z" in frame.columns else pl.lit(None)
    bot = pl.col("sz_bot") if "sz_bot" in frame.columns else pl.lit(None)
    top = pl.col("sz_top") if "sz_top" in frame.columns else pl.lit(None)
    nx, nz = plate_x / 0.83, (plate_z - bot) / (top - bot)
    dx = pl.when(nx < -1).then(-1 - nx).when(nx > 1).then(nx - 1).otherwise(0.0)
    dz = pl.when(nz < 0).then(-nz).when(nz > 1).then(nz - 1).otherwise(0.0)
    return frame.with_columns(
        normalized_plate_x=nx,
        normalized_plate_z=nz,
        in_rulebook_zone=(nx.abs() <= 1) & (nz >= 0) & (nz <= 1),
        horizontal_distance_from_zone=dx,
        vertical_distance_from_zone=dz,
        distance_from_zone_boundary=(dx.pow(2) + dz.pow(2)).sqrt(),
    ).with_columns(
        attack_zone=pl.when(
            pl.col("normalized_plate_x").is_null() | pl.col("normalized_plate_z").is_null()
        )
        .then(pl.lit("unknown"))
        .when(
            (pl.col("normalized_plate_x").abs() <= 0.55)
            & pl.col("normalized_plate_z").is_between(0.225, 0.775)
        )
        .then(pl.lit("heart"))
        .when(pl.col("distance_from_zone_boundary") <= 0.25)
        .then(pl.lit("shadow"))
        .when(pl.col("distance_from_zone_boundary") <= 0.75)
        .then(pl.lit("chase"))
        .otherwise(pl.lit("waste"))
    )
