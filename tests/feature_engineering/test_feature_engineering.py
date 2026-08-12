# This is for testing the feature engineering functions. It is not part of the main application.
from src.feature_engineering.feature_builder import build_features
from src.feature_engineering.load_champion_data import load_champion_data
from  src.feature_engineering.composition_features import get_team_composition_features
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]

champion_data = load_champion_data(PROJECT_ROOT / "data" / "champions.json")

blue_team = [
    "Aatrox",
    "Ahri",
    "Ashe",
    "Leona",
    "LeeSin"
]

red_team = [
    "Darius",
    "Diana",
    "Draven",
    "Jinx",
    "Thresh"
]

features = get_team_composition_features(blue_team, champion_data)

team_features = build_features(blue_team, red_team, champion_data)
# for team blue
print(features)

# for both teams
print(team_features)

