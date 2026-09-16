# Swing outcome baseline v0.1

Empirical probabilities for whiff, foul, and ball in play condition on pre-pitch balls, strikes, and open attack zone. It uses unambiguous swings only and no future data. `baseball train-swing-model --train-end YYYY-MM-DD` persists a training cutoff; evaluate later dates with `baseball validate-foul --holdout-start YYYY-MM-DD`. A whole-sample fit is descriptive only, not a holdout result.
