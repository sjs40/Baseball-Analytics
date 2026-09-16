# Research rules

1. Never silently change a metric definition; version and document it.
2. Never fabricate baseball data. Preserve raw downloads as immutable source data.
3. Separate descriptive or retrospective outputs from predictive models.
4. Establish naive baselines before adding model complexity; validate chronologically.
5. Guard against target leakage, report sample sizes and uncertainty, and keep player skill distinct from context where possible.
6. Metric outputs must be reproducible from stored source data and traceable by game, PA, pitch, batter, and pitcher.
7. Streamlit and Textual interfaces may display data but must not contain analytical logic.
8. Do not promote experimental statistics into the core package before validation tests and exact documentation exist.
