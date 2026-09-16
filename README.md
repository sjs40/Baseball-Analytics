# Baseball Analytics

Extensible, reproducible research infrastructure for metrics that separate immediate plate-appearance value from process value. V0.3 adds chronological out-of-sample foul and whiff-avoidance residual infrastructure while keeping FSV and FAVE definitions explicit.

## V0.2 pipeline

```powershell
uv sync --extra dashboard --extra dev
uv run baseball ingest --start 2025-04-01 --end 2025-04-30
uv run baseball build-features --raw statcast_2025-04-01_2025-04-30.parquet
uv run baseball build-player-map
uv run baseball compute-fsv
uv run baseball train-swing-model --train-end 2025-04-15
uv run baseball build-oos-ledger
uv run baseball compute-fae
uv run baseball compute-fave
uv run baseball build-foul-profiles
uv run baseball validate-foul --holdout-start 2025-04-16
uv run baseball report-foul
uv run baseball tui
uv run streamlit run apps/streamlit/Home.py
```

Raw downloads are immutable season partitions in `data/raw/statcast/season=YYYY/`; canonical data is in `data/interim/`; metric outputs are Parquet files in `data/processed/`. Read [FSV](docs/metrics/FSV.md), [FAVE](docs/metrics/FAVE.md), and [FAE](docs/metrics/FAE.md) for exact definitions. FAVE, FAE, and spoil difficulty are experimental. WEAR and PEST remain intentionally unimplemented.

Every output is traceable by `game_pk + at_bat_number + pitch_number`; interfaces only read persisted outputs. Consult [AGENTS.md](AGENTS.md) before changing a metric definition.
