# Fixed draft order used throughout the project: Top, Jungle, Mid, ADC, Support.
# role_freqs / counter_matrix in feature_matrices.json use "bot" for ADC and
# "jungle"/"support" spelled out in full, so this is the mapping into that
# naming convention, in the same positional order as team_champions lists.
ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"]

# counter_matrix only has entries for the three solo lanes (no 2v2 bot lane,
# no jungle, since those don't have a clean 1v1 "matchup" the same way).
_LANE_ADVANTAGE_ROLES = ["top", "mid", "support"]


def get_team_composition_features(team_champions, champion_data, champion_winrates=None, feature_matrices=None):
    features = {
        "num_fighters": 0,
        "num_tanks": 0,
        "num_mages": 0,
        "num_assassins": 0,
        "num_marksmen": 0,
        "num_supports": 0,

        "num_ranged": 0,
        "num_melee": 0,

        "avg_attack": 0.0,
        "avg_defense": 0.0,
        "avg_magic": 0.0,
        "avg_difficulty": 0.0,

        "avg_hp": 0.0,
        "avg_armor": 0.0,
        "avg_attackdamage": 0.0,

        # Empirical win rate of each picked champion, averaged across the team.
        # Computed from historical match data (see encode_champions.py).
        "avg_champion_winrate": 0.0,

        # Share of the champion's damage that is magic (AP) rather than
        # physical, averaged across the team. Higher = more magic-damage-heavy
        # team composition.
        "avg_ap_ratio": 0.0,

        # How consistent/build-dependent that AP ratio is across games
        # (variance). Higher = the champion's damage type swings more
        # depending on build (e.g. flex picks).
        "avg_ap_variance": 0.0,

        # How often each champion is actually played in the role it was
        # drafted into here, averaged across the team. Lower = more
        # "off-role" picks, which can correlate with unfamiliarity/weaker play.
        "avg_role_freq": 0.0,
    }

    champ_ap_ratios = (feature_matrices or {}).get("champ_ap_ratios", {})
    champ_ap_variances = (feature_matrices or {}).get("champ_ap_variances", {})
    role_freqs = (feature_matrices or {}).get("role_freqs", {})

    num_champions = len(team_champions)

    for i, champion in enumerate(team_champions):

        champion_info = champion_data[champion]

        tags = champion_info["tags"]

        if "Fighter" in tags:
            features["num_fighters"] += 1

        if "Tank" in tags:
            features["num_tanks"] += 1

        if "Mage" in tags:
            features["num_mages"] += 1

        if "Assassin" in tags:
            features["num_assassins"] += 1

        if "Marksman" in tags:
            features["num_marksmen"] += 1

        if "Support" in tags:
            features["num_supports"] += 1

        stats = champion_info["stats"]

        if stats["attackrange"] >= 400:
            features["num_ranged"] += 1
        else:
            features["num_melee"] += 1

        features["avg_attack"] += champion_info["info"]["attack"]
        features["avg_defense"] += champion_info["info"]["defense"]
        features["avg_magic"] += champion_info["info"]["magic"]
        features["avg_difficulty"] += champion_info["info"]["difficulty"]

        features["avg_hp"] += stats["hp"]
        features["avg_armor"] += stats["armor"]
        features["avg_attackdamage"] += stats["attackdamage"]

        # Fall back to 0.5 (an unbiased "average" win rate) for champions we
        # have no historical win-rate data for.
        if champion_winrates is not None:
            winrate = champion_winrates.get(champion, 0.5)
        else:
            winrate = 0.5
        features["avg_champion_winrate"] += winrate

        # AP ratio / variance: fall back to 0.0 (assume physical damage,
        # no known variance) if the champion is missing from the matrix.
        features["avg_ap_ratio"] += champ_ap_ratios.get(champion, 0.0)
        features["avg_ap_variance"] += champ_ap_variances.get(champion, 0.0)

        # Role frequency: how often this champion is played in the role it
        # was assigned to in this draft. Fall back to 0.2 (uniform across the
        # 5 roles, i.e. "no information") if the champion is missing entirely.
        assigned_role = ROLE_ORDER[i] if i < len(ROLE_ORDER) else None
        champ_role_freqs = role_freqs.get(champion)
        if champ_role_freqs is not None and assigned_role is not None:
            features["avg_role_freq"] += champ_role_freqs.get(assigned_role, 0.2)
        else:
            features["avg_role_freq"] += 0.2

    for key in [
        "avg_attack",
        "avg_defense",
        "avg_magic",
        "avg_difficulty",
        "avg_hp",
        "avg_armor",
        "avg_attackdamage",
        "avg_champion_winrate",
        "avg_ap_ratio",
        "avg_ap_variance",
        "avg_role_freq",
    ]:
        features[key] /= num_champions

    return features


def get_lane_advantage_features(blue_team, red_team, counter_matrix=None):
    """
    Computes per-lane matchup advantage for the three solo lanes (top, mid,
    support) using counter_matrix, which only contains 1v1 lane matchup data
    (no jungle, no 2v2 bot lane).

    Positive value = Blue favored in that lane, negative = Red favored,
    based on the "{role}:{champA}_vs_{champB}" convention in counter_matrix
    (confirmed antisymmetric: A_vs_B == -B_vs_A). Missing matchups default
    to 0.0 (no known advantage either way).

    blue_team / red_team must be in the fixed [Top, Jungle, Mid, ADC, Support]
    order used throughout this project.
    """
    counter_matrix = counter_matrix or {}
    features = {}

    role_to_index = {role: i for i, role in enumerate(ROLE_ORDER)}

    for role in _LANE_ADVANTAGE_ROLES:
        idx = role_to_index[role]
        blue_champ = blue_team[idx]
        red_champ = red_team[idx]
        key = f"{role}:{blue_champ}_vs_{red_champ}"
        features[f"{role}_lane_advantage"] = counter_matrix.get(key, 0.0)

    return features