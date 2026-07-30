from pathlib import Path

from src.models.analyze_team import generate_explanation
from src.feature_engineering.load_champion_data import load_champion_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = PROJECT_ROOT / "data" / "champions.json"

#test: python -m pytest tests/models/test_analyze_team.py

def test_generate_explaination():
    champion_data = load_champion_data(JSON_PATH)

    team = ["Ahri", "Jinx", "Malphite", "LeeSin", "Thresh"]

    result = generate_explanation(team, champion_data)

    assert "Strong frontline" in result["reasons"]
    assert "Balanced mixed damage (AD/AP)" in result["reasons"]
    assert len(result["weaknesses"]) == 0