"""Deterministic Statcast swing and foul classifications."""

FOUL_DESCRIPTIONS = {"foul", "foul_pitchout", "foul_tip"}
FOUL_BUNT_DESCRIPTIONS = {"foul_bunt", "missed_bunt"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
BIP_DESCRIPTIONS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
BUNT_DESCRIPTIONS = {"bunt_foul_tip", "foul_bunt", "missed_bunt", "bunt"}


def swing_outcome(description: object) -> str:
    value = str(description or "")
    if value in WHIFF_DESCRIPTIONS:
        return "bunt" if value == "missed_bunt" else "whiff"
    if value in FOUL_DESCRIPTIONS:
        return "foul"
    if value in BIP_DESCRIPTIONS:
        return "ball_in_play"
    if value in BUNT_DESCRIPTIONS:
        return "bunt"
    if value == "swinging_pitchout":
        return "other"
    return "ambiguous"
