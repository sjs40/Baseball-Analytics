# Fouls Above Expected (FAE) v0.3

FAE is an experimental, retrospective hitter residual evaluated only from the
chronological out-of-sample prediction ledger. For swing opportunity *i*:

`FAE_i = I(outcome_i = foul) - P_OOS(foul_i)`.

The unit is fouls above expectation. Batter totals are sums; `FAE per 100
swings` is 100 times the total divided by modeled swing opportunities. Two-strike
variants restrict both numerator and denominator to swings beginning with two
strikes. The dependency is Model B `0.2-smoothed-empirical-count-attack-zone`
until a chronologically validated successor is selected.

FAE measures foul propensity conditional on a swing, not pitch selection or
overall offensive value. It is experimental and must always be accompanied by
opportunities and its shrunk estimate; it is not evidence of persistent skill
until reliability testing supports that conclusion.
