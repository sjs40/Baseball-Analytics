# Baseball Analytics

Extensible, reproducible research infrastructure for metrics that separate immediate plate-appearance value from process value. V0.1 implements neutral **Foul Survival Value (FSV)** only.

## V0.1 pipeline

```powershell
uv sync --extra dashboard --extra dev
uv run baseball ingest --start 2025-04-01 --end 2025-04-30
uv run baseball build-features --raw statcast_2025-04-01_2025-04-30.parquet
uv run baseball compute-fsv
uv run baseball validate
uv run baseball tui
uv run streamlit run apps/streamlit/Home.py
```

Raw downloads are immutable in `data/raw/`; canonical data is in `data/interim/`; metric outputs are Parquet files in `data/processed/`. Read [FSV](docs/metrics/FSV.md) for its exact definition. FAVE, WEAR, and PEST are intentionally not calculated yet.

Every output is traceable by `game_pk + at_bat_number + pitch_number`; interfaces only read persisted outputs. Consult [AGENTS.md](AGENTS.md) before changing a metric definition.
