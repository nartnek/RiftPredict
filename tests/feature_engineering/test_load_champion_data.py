import json
import pytest

from src.feature_engineering.load_champion_data import (load_champion_data,)


def test_load_champion_data_returns_data_section(tmp_path):
    # Create temporary test JSON data
    test_data = {
        "type": "champion",
        "version": "15.12.1",
        "data": {
            "Aatrox": {
                "id": "Aatrox",
                "name": "Aatrox",
                "tags": ["Fighter"],
            },
            "Ahri": {
                "id": "Ahri",
                "name": "Ahri",
                "tags": ["Mage", "Assassin"],
            },
        },
    }

    # Create a temporary JSON file
    test_file = tmp_path / "champions.json"

    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(test_data, file)

    # Run the function
    champion_data = load_champion_data(test_file)

    # Check that only the "data" section is returned
    assert champion_data == test_data["data"]

    # Check that expected champions exist
    assert "Aatrox" in champion_data
    assert "Ahri" in champion_data


def test_load_champion_data_returns_dictionary(tmp_path):
    test_data = {
        "data": {
            "Jinx": {
                "id": "Jinx",
                "name": "Jinx",
            }
        }
    }

    test_file = tmp_path / "champions.json"

    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(test_data, file)

    champion_data = load_champion_data(test_file)

    assert isinstance(champion_data, dict)


def test_load_champion_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_champion_data("data/file_that_does_not_exist.json")