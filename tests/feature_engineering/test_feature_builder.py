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

    # Neutral filler with no tags, used to pad teams to a full 5-champion
    # draft (Top/Jungle/Mid/ADC/Support) without affecting tag-count
    # assertions. build_features requires full 5-length teams since it also
    # computes lane-matchup features, which index into fixed Top/Mid/Support
    # positions (0, 2, 4).
    "Filler": {
        "tags": [],
        "info": {
            "attack": 5,
            "defense": 5,
            "magic": 5,
            "difficulty": 1,
        },
        "stats": {
            "attackrange": 400,
            "hp": 500,
            "armor": 30,
            "attackdamage": 50,
        },
    },
}

# Aatrox/Ahri in the Top slot (index 0), Filler everywhere else, so tag
# counts stay exactly 1 as before while satisfying the 5-champion length
# build_features needs for lane-advantage indexing.
BLUE_TEAM = ["Aatrox", "Filler", "Filler", "Filler", "Filler"]
RED_TEAM = ["Ahri", "Filler", "Filler", "Filler", "Filler"]


def test_build_features_adds_blue_and_red_prefixes():
    features = build_features(
        BLUE_TEAM,
        RED_TEAM,
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
    features = build_features(
        BLUE_TEAM,
        RED_TEAM,
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
        BLUE_TEAM,
        BLUE_TEAM,
        CHAMPION_DATA,
    )

    # The output should be a dictionary
    assert isinstance(features, dict)


def test_all_feature_names_have_team_prefixes():
    features = build_features(
        BLUE_TEAM,
        BLUE_TEAM,
        CHAMPION_DATA,
    )

    # Lane-advantage features are intentionally cross-team (Blue champ vs.
    # Red champ in the same lane), so they don't get a blue_/red_ prefix
    # like the per-team aggregate features do.
    CROSS_TEAM_FEATURES = {
        "top_lane_advantage",
        "mid_lane_advantage",
        "support_lane_advantage",
    }

    # Every other feature should begin with either "blue_" or "red_"
    for feature_name in features:
        if feature_name in CROSS_TEAM_FEATURES:
            continue
        assert (feature_name.startswith("blue_") or feature_name.startswith("red_"))