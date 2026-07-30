from pathlib import Path
import pandas as pd

from src.feature_engineering.load_champion_data import load_champion_data
from src.preprocessing.encode_champions import transform_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = PROJECT_ROOT / "data" / "champions.json"

def test_transform_dataset():
    columns = [
        "match_id", "queue_type", "patch", "duration", "winner",
        "blue_top", "blue_jg", "blue_mid", "blue_adc", "blue_sup",
        "red_top", "red_jg", "red_mid", "red_adc", "red_sup"
    ]

    mock_match = [
        "NA1_5609637949", 400, "16.14.794.9266", 1754, "blue",
        "Gangplank", "Vayne", "Rengar", "Nocturne", "Ryze",
        "Akali", "Caitlyn", "Jinx", "Lux", "Seraphine"
    ]

    mock_df = pd.DataFrame([mock_match], columns=columns)

    champ_data = load_champion_data(JSON_PATH)

    result = transform_dataset(mock_df, champ_data)

    # Testing
    assert result["winner"].iloc[0] == 1    # winner = blue
    assert len(result) == 1                 # row count = 1
    assert result.shape[1] > 1              # column count > 1
    assert "blue_top" not in result.columns # dropped column not in result

also added Eugene's champion name resolver for names like MonkeyKing -> Wukong, Chogath -> Cho'gath, etc.

Explanations currently focus on each team separately and do not consider if, for example, team red counters team blue.

Probability of a team winning is calculated using Scikit-learn's predict_proba(). consistently outputs 60:40 ratio? 