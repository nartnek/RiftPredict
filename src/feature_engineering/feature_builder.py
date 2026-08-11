from .composition_features import get_team_composition_features, get_lane_advantage_features


def build_features(blue_team, red_team, champion_data, champion_winrates=None, feature_matrices=None):
    blue_features = get_team_composition_features(
        blue_team, champion_data, champion_winrates, feature_matrices
    )

    red_features = get_team_composition_features(
        red_team, champion_data, champion_winrates, feature_matrices
    )

    feature_vector = {}

    for key, value in blue_features.items():
        feature_vector[f"blue_{key}"] = value

    for key, value in red_features.items():
        feature_vector[f"red_{key}"] = value

    # Lane matchup advantage is inherently cross-team (blue champ vs. the
    # red champ in the same lane), so it isn't a per-team stat like the
    # others above — computed once and added directly.
    counter_matrix = (feature_matrices or {}).get("counter_matrix", {})
    lane_features = get_lane_advantage_features(blue_team, red_team, counter_matrix)
    feature_vector.update(lane_features)

    return feature_vector