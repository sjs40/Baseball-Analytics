# Spoil difficulty — EXPERIMENTAL

For normal two-strike fouls, spoil difficulty is the model's expected whiff probability. It is a difficulty feature, not run value; FSV and FAVE remain separate.

## Definition and limitations

Version `0.1-experimental-delta-re-decomposition` uses the baseline swing model's pre-pitch whiff probability. Inputs are count, attack zone, and the model version; units are probability, not runs. It excludes foul bunts and terminal foul tips through the two-strike-foul flag. It is not a final branded statistic and has only weak one-month split-half evidence, so it must not be treated as established batter skill.
