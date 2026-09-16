# Foul-ball research report — EXPERIMENTAL

## Scope

- Date range: 2025-04-01 through 2025-04-30
- Pitches: 114,767
- Modeled swings: 53,691
- Normal two-strike fouls: 7,734
- FSV is neutral run value; FAVE and spoil difficulty remain experimental.

## Descriptive two-strike foul profiles

| batter_name | batter | two_strike_fouls | two_strike_fouls_per_two_strike_swing |
| --- | --- | --- | --- |
| Austin Hedges | 595978 | 13 | 0.5909090909090909 |
| Justin Turner | 457759 | 36 | 0.5901639344262295 |
| Mike Tauchman | 643565 | 10 | 0.5882352941176471 |
| Will Wagner | 695238 | 36 | 0.5714285714285714 |
| Jonathan Aranda | 666018 | 48 | 0.5714285714285714 |
| Yohel Pozo | 650968 | 13 | 0.5652173913043478 |
| Andy Pages | 681624 | 49 | 0.5632183908045977 |
| Curtis Mead | 678554 | 18 | 0.5625 |
| Matt Thaiss | 642136 | 21 | 0.5526315789473685 |
| Nico Hoerner | 663538 | 34 | 0.5483870967741935 |

## FSV leaderboard (minimum 10 two-strike fouls)

| batter_name | batter | total_fsv | two_strike_fouls | fsv_per_100_pa |
| --- | --- | --- | --- | --- |
| Alec Burleson | 676475 | 24.569367372207033 | 22 | 31.499188938726967 |
| Ian Happ | 664023 | 24.298245349927843 | 34 | 20.946763232696416 |
| Rhys Hoskins | 656555 | 22.402095831340418 | 31 | 24.350104164500454 |
| Cal Raleigh | 663728 | 22.295885411241866 | 53 | 19.387726444558144 |
| Hunter Goodman | 696100 | 21.473684363206658 | 26 | 22.844345067241125 |
| Anthony Volpe | 683011 | 20.939964506714336 | 38 | 17.74573263280876 |
| Pete Alonso | 624413 | 20.547942552899706 | 51 | 17.123285460749752 |
| Michael Harris | 671739 | 20.466653657401157 | 32 | 19.127713698505755 |
| Gabriel Moreno | 672515 | 20.250324875816254 | 28 | 27.000433167755006 |
| Spencer Torkelson | 679529 | 19.558260513588753 | 38 | 17.46273260141853 |

## Experimental FAVE leaderboard (minimum 10 two-strike fouls)

| batter_name | batter | total_fave | two_strike_fouls |
| --- | --- | --- | --- |
| Blake Dunn | 694362 | 4.757280821127817 | 15 |
| Joey Ortiz | 687401 | 4.634883389499153 | 30 |
| Austin Wells | 669224 | 4.3784989247328445 | 40 |
| Kyle Farmer | 571657 | 4.363640147662865 | 23 |
| Alec Burleson | 676475 | 4.315504539429681 | 22 |
| Brice Turang | 668930 | 4.201433509832291 | 35 |
| Michael Harris | 671739 | 4.196002575605405 | 32 |
| Manny Machado | 592518 | 4.061382408684362 | 48 |
| Randy Arozarena | 668227 | 4.0327891367076605 | 37 |
| Nico Hoerner | 663538 | 3.8843794237156013 | 34 |

## Holdout swing-model check

- Holdout start: 2025-04-16
- Holdout multiclass log loss: 1.0212
- Naive frequency log loss: 1.0717
- Calibration tables and Brier scores are in `data/processed/foul_validation.json`.

## Reliability and cautions

- Split-half spoil-difficulty Pearson: 0.23529947393556197; Spearman: 0.18952654068212121.
- The short one-month sample yields weak FSV/FAVE split-half reliability; do not interpret these leaderboards as established player skill.
- The model is intentionally simple, and fair-ball value uses conditional mean Statcast delta RE. This is a starting benchmark, not a final player model.

## Recommended next step

Ingest multiple complete seasons, repeat chronological holdouts, then evaluate stabilization before setting qualification thresholds or interpreting year-over-year rankings.
