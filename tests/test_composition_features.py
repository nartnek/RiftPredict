from src.feature_engineering.composition_features import (get_team_composition_features,)

import pytest

# Small fake champion dataset for testing
@pytest.fixture
def champion_data():
    return {
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

        "Jinx": {
            "tags": ["Marksman"],
            "info": {
                "attack": 9,
                "defense": 2,
                "magic": 4,
                "difficulty": 6,
            },
            "stats": {
                "attackrange": 525,
                "hp": 630,
                "armor": 26,
                "attackdamage": 59,
            },
        },
    }


def test_team_composition_counts_tags(champion_data):
    team = [
        "Aatrox",
        "Ahri",
        "Jinx"
    ]

    features = get_team_composition_features(
        team,
        champion_data
    )

    # Aatrox
    assert features["num_fighters"] == 1

    # Ahri
    assert features["num_mages"] == 1
    assert features["num_assassins"] == 1

    # Jinx
    assert features["num_marksmen"] == 1


def test_team_composition_range_classification(champion_data):
    team = [
        "Aatrox",
        "Ahri"
    ]

    features = get_team_composition_features(
        team,
        champion_data
    )

    # Aatrox = melee
    # Ahri = ranged
    assert features["num_melee"] == 1
    assert features["num_ranged"] == 1


def test_team_composition_average_stats(champion_data):
    team = [
        "Aatrox",
        "Ahri"
    ]

    features = get_team_composition_features(
        team,
        champion_data
    )


    # (8 + 3) / 2 = 5.5
    assert features["avg_attack"] == 5.5

    # (650 + 590) / 2 = 620
    assert features["avg_hp"] == 620

    # (60 + 53) / 2 = 56.5
    assert features["avg_attackdamage"] == 56.5


def test_all_feature_keys_exist(champion_data):
    team = ["Aatrox"]

    features = get_team_composition_features(
        team,
        champion_data
    )

    expected_keys = [
        "num_fighters",
        "num_tanks",
        "num_mages",
        "num_assassins",
        "num_marksmen",
        "num_supports",
        "num_ranged",
        "num_melee",
        "avg_attack",
        "avg_defense",
        "avg_magic",
        "avg_difficulty",
        "avg_hp",
        "avg_armor",
        "avg_attackdamage",
    ]

    for key in expected_keys:
        assert key in features


def test_unknown_champion_raises_error(champion_data):
    team = [
        "UnknownChampion"
    ]

    try:
        get_team_composition_features(
            team,
            champion_data
        )
        assert False

    except KeyError:
        assert True