from src.feature_engineering.composition_features import (
    get_team_composition_features,
    get_lane_advantage_features,
)

# Thresholds below are calibrated against the actual distributions in
# feature_matrices.json (checked directly, not guessed):
#   - champ_ap_ratios spans ~0-1 with mean 0.52, so 0.35/0.65 sensibly
#     separates "AP-heavy" / "balanced" / "AD-heavy" team averages.
#   - counter_matrix has mean 0.0, stdev ~0.045, range +/-0.19, so +/-0.05
#     (about one stdev) is used as the cutoff for a "meaningful" lane edge
#     rather than noise around zero.
#   - role_freqs: a champion's most-played role sits at ~0.80-0.84 on
#     average, so a team averaging below 0.5 on their assigned roles is a
#     real signal of off-role/unconventional picks, not normal variance.
_WINRATE_STRONG = 0.52
_WINRATE_WEAK = 0.48
_AP_BALANCED_LOW = 0.35
_AP_BALANCED_HIGH = 0.65
_OFF_ROLE_THRESHOLD = 0.5
_LANE_ADVANTAGE_THRESHOLD = 0.05


# Translate feature comparisons into human-readable text
def generate_explanation(team, champion_data, champion_winrates=None, feature_matrices=None):

    features = get_team_composition_features(team, champion_data, champion_winrates, feature_matrices)

    reasons = []
    weaknesses = []

    # Team Tankiness
    if features.get("num_tanks", 0) >= 2:
        reasons.append("Strong frontline")
    else:
        weaknesses.append("Squishy composition")

    # Team magic power (AP) — avg_ap_ratio is the team-averaged share of each
    # champion's damage that is magic rather than physical, sourced from real
    # match/build data (feature_matrices.json) rather than Riot's static
    # info.magic/info.attack ratings.
    magic_ratio = features.get("avg_ap_ratio", 0.5)

    if _AP_BALANCED_LOW <= magic_ratio <= _AP_BALANCED_HIGH:
        reasons.append("Balanced mixed damage (AD/AP)")
    elif magic_ratio > _AP_BALANCED_HIGH:
        weaknesses.append("Heavy Magic Damage (easy for enemies to itemize MR)")
    else:
        weaknesses.append("Heavy Physical Damage (easy for enemies to itemize Armor)")

    # Champion win rate — how historically strong the picked champions are,
    # averaged across the team (Bayesian-smoothed, see encode_champions.py).
    winrate = features.get("avg_champion_winrate", 0.5)
    if winrate >= _WINRATE_STRONG:
        reasons.append("Historically strong champion picks (above-average win rate)")
    elif winrate <= _WINRATE_WEAK:
        weaknesses.append("Historically weaker champion picks (below-average win rate)")

    # Off-role picks — how often each champion is actually played in the
    # role they were drafted into here. A team full of "off-meta" role
    # assignments tends to reflect unfamiliar or non-standard play.
    role_freq = features.get("avg_role_freq", 0.8)
    if role_freq < _OFF_ROLE_THRESHOLD:
        weaknesses.append("Contains off-role or unconventional picks")

    # Composition shape from champion tags
    if features.get("num_assassins", 0) >= 2:
        reasons.append("High pick potential (assassin-heavy)")

    if features.get("num_mages", 0) >= 3:
        reasons.append("High magic burst potential")

    if features.get("num_marksmen", 0) == 0:
        weaknesses.append("No dedicated marksman for sustained late-game damage")

    return {
        "reasons": reasons,
        "weaknesses": weaknesses,
    }


def generate_matchup_explanation(blue_team, red_team, feature_matrices=None):
    """
    Cross-team lane matchup ("counter") reasoning, using counter_matrix from
    feature_matrices.json. Only covers the three solo lanes (top, mid,
    support) since that's all the data provides.

    Returns which lanes clearly favor Blue, which clearly favor Red, and
    which are roughly even (within +/-_LANE_ADVANTAGE_THRESHOLD of zero,
    i.e. within noise given counter_matrix's ~0.045 stdev).
    """
    counter_matrix = (feature_matrices or {}).get("counter_matrix", {})
    lane_features = get_lane_advantage_features(blue_team, red_team, counter_matrix)

    blue_favored = []
    red_favored = []
    even_lanes = []

    for lane_key, advantage in lane_features.items():
        lane_name = lane_key.replace("_lane_advantage", "").capitalize()

        if advantage > _LANE_ADVANTAGE_THRESHOLD:
            blue_favored.append(lane_name)
        elif advantage < -_LANE_ADVANTAGE_THRESHOLD:
            red_favored.append(lane_name)
        else:
            even_lanes.append(lane_name)

    return {
        "blue_favored_lanes": blue_favored,
        "red_favored_lanes": red_favored,
        "even_lanes": even_lanes,
    }