from .composition_features import get_team_composition_features



def build_features(blue_team, red_team, champion_data):
    blue_features = get_team_composition_features(blue_team, champion_data)

    red_features = get_team_composition_features(red_team, champion_data)

    if blue_features is None or red_features is None:
        return None

    feature_vector = {}

    for key, value in blue_features.items():
        feature_vector[f"blue_{key}"] = value

    for key, value in red_features.items():
        feature_vector[f"red_{key}"] = value

    return feature_vector