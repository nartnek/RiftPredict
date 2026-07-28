from src.feature_engineering.feature_builder import build_features


# Complete mock champion data used by all tests
CHAMPION_DATA = {
    "Aatrox": {
        "tags": ["Fighter"],
        "info": {
            "attack": 8,
            "defense": 4,
            "magic": 3,
            "difficulty": 4,
        },
        "stats": {
            "attackrange": 175,
            "hp": 650,
            "armor": 38,
            "attackdamage": 60,
        },
    },

    "Ahri": {
        "tags": ["Mage", "Assassin"],
        "info": {
            "attack": 3,
            "defense": 4,
            "magic": 8,
            "difficulty": 5,
        },
        "stats": {
            "attackrange": 550,
            "hp": 590,
            "armor": 21,
            "attackdamage": 53,
        },
    },
}


def test_build_features_adds_blue_and_red_prefixes():
    blue_team = ["Aatrox"]
    red_team = ["Ahri"]

    features = build_features(
        blue_team,
        red_team,
        CHAMPION_DATA,
    )

    # Check that Blue Team features have "blue_" prefixes
    assert "blue_num_fighters" in features

    # Check that Red Team features have "red_" prefixes
    assert "red_num_mages" in features

    # Check that values belong to the correct team
    assert features["blue_num_fighters"] == 1
    assert features["red_num_mages"] == 1


def test_build_features_keeps_blue_and_red_features_separate():
    blue_team = ["Aatrox"]
    red_team = ["Ahri"]

    features = build_features(
        blue_team,
        red_team,
        CHAMPION_DATA,
    )

    # Blue Team has one Fighter
    assert features["blue_num_fighters"] == 1

    # Red Team has no Fighters
    assert features["red_num_fighters"] == 0

    # Red Team has one Mage
    assert features["red_num_mages"] == 1

    # Blue Team has no Mages
    assert features["blue_num_mages"] == 0


def test_build_features_returns_dictionary():
    features = build_features(
        ["Aatrox"],
        ["Aatrox"],
        CHAMPION_DATA,
    )

    # The output should be a dictionary
    assert isinstance(features, dict)


def test_all_feature_names_have_team_prefixes():
    features = build_features(
        ["Aatrox"],
        ["Aatrox"],
        CHAMPION_DATA,
    )

    # Every feature should begin with either
    # "blue_" or "red_"
    for feature_name in features:
        assert (feature_name.startswith("blue_") or feature_name.startswith("red_"))