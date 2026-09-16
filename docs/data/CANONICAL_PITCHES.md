# Canonical pitches

One sorted Statcast row per pitch with stable pitch/PA IDs, lagged sequence features, workload counts, explicit swing classifications, and open geometric location features. Heart is x ±0.55 and z .225–.775; shadow is within .25 zone units, chase within .75, otherwise waste.

Missing plate or strike-zone coordinates are labeled `unknown`, never silently treated as waste. Sequence fields use only preceding pitches in the same PA. `two_strike_foul_number` is total two-strike fouls so far in the PA, while `consecutive_two_strike_foul_count` resets after any non-two-strike-foul. Canonical validation rejects duplicate IDs and impossible counts.
