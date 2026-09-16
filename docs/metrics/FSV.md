# FSV — Foul Survival Value (v0.2)

V0.1 outputs remain versioned as historical results. V0.2 values an ordinary two-strike foul against the counterfactual that the same pitch became a third strike:

`FSV = V(actual post-foul state) - V(strikeout counterfactual)`.

`V` is empirical expected runs to the end of the offensive half-inning, calculated from the ingested Statcast sample. The counterfactual adds an out, preserves runners, and begins the next batter at 0–0. It is zero only when the extra out ends the inning. Each value persists its exact/base-out-weighted/league lookup tier.

Only normal two-strike fouls receive FSV in v0.2. Earlier-count fouls receive zero because a strike produces the same count. Foul bunts and terminal foul tips remain explicit in canonical data and are excluded. This is neutral, not batter-adjusted, FSV; it is descriptive. FAVE and spoil difficulty are separate experimental outputs; WEAR remains unimplemented.

## Inputs, units, and limitations

Inputs are the canonical pre-pitch base/out/count state, normal-foul flags, and the versioned count-aware run-expectancy table. Units are expected future offensive runs in the current half-inning. The metric is traceable by `pitch_id`, including actual and counterfactual lookup tiers. It does not estimate the value of a particular batter remaining at the plate, account for score/inning leverage, or establish a repeatable batter skill. Sparse states use a weighted base/out fallback and then league fallback; this can blur count effects when source data are thin.
