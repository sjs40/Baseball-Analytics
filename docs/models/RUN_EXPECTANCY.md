# Run expectancy v0.2

States are outs, runner occupancy, balls, and strikes. Lookup uses exact state, then observation-weighted base/out value, then league average. Three outs is always zero.

The table version is `0.2-count-aware-hierarchical` and persists expected runs, observations, runner bitmask, base/out state, count state, and full-state ID. Values are reproducible from immutable raw input using future runs in the offensive half-inning. The fallback deliberately never averages unrelated base/out states merely because they share a count. Estimates remain descriptive and sensitive to sample size, scorer corrections, and the game contexts represented in the downloaded data.
