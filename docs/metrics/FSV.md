# FSV — Foul Survival Value (v0.1)

FSV values an ordinary two-strike foul against the counterfactual that the same pitch became a third strike:

`FSV = V(outs, runners, balls, 2 strikes) - V(strikeout)`.

`V` is empirical expected runs to the end of the offensive half-inning, calculated from the ingested Statcast sample. A strikeout ends the PA and has zero continuation value in this pitch-level counterfactual.

Only normal two-strike fouls receive FSV in v0.1. Earlier-count fouls receive zero because a strike produces the same count. Foul bunts and terminal foul tips remain explicit in canonical data and are excluded. This is neutral, not batter-adjusted, FSV; it is descriptive and does not include FAVE, pitch difficulty, or WEAR.
