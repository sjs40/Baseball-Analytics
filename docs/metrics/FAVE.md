# FAVE v0.1 — EXPERIMENTAL

Foul Added Value versus Expected asks how a foul's actual run-expectancy change compares with the expected value of that swing. `0.1-experimental-delta-re-decomposition` uses whiff, foul, and fair-ball probabilities from count and the documented open attack zone. Whiff and foul values are pre-to-post state transitions; the fair-ball component is the observed conditional mean Statcast `delta_run_exp`. It is inspectable and neutral, but not validated player skill. Foul bunts and terminal foul tips are excluded.
