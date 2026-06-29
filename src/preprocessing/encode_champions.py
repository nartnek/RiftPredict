import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# python -m src.feature_engineering.feature_builder
from src.feature_engineering.composition_features import (
    get_team_composition_features,
)
from src.feature_engineering.feature_builder import build_features

# Get champion data
def get_live_champion_data():
    version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    latest_version = requests.get(version_url).json()[0]

    data_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
    response = requests.get(data_url).json()

    return response["data"]


live_champ_data = get_live_champion_data()

# Mock data for testing
roles = [
    "blue_top",
    "blue_jungle",
    "blue_mid",
    "blue_adc",
    "blue_support",
    "red_top",
    "red_jungle",
    "red_mid",
    "red_adc",
    "red_support",
]

np.random.seed(42) # Random number generator picks same sequence
num_matches = 100
mock_matches = {
    role: np.random.choice(["Ahri", "Jinx", "Thresh", "Aatrox"], num_matches) # Randomly pick between champions and assign role
    for role in roles
}
mock_matches["winner"] = np.random.choice([0, 1], num_matches)
df_matches = pd.DataFrame(mock_matches)

# Transform data into feature matrix
def transform_dataset(df_raw, champion_data):
    processed_rows = []

    # For each row in the dataset
    # assemble blue and red team lists 
    for _, row in df_raw.iterrows():

        blue_team = [
            row["blue_top"],
            row["blue_jungle"],
            row["blue_mid"],
            row["blue_adc"],
            row["blue_support"],
        ]
        red_team = [
            row["red_top"],
            row["red_jungle"],
            row["red_mid"],
            row["red_adc"],
            row["red_support"],
        ]

        # Map champion text names to numerical stats
        match_vector = build_features(blue_team, red_team, champion_data)

        match_vector["winner"] = row["winner"]

        processed_rows.append(match_vector)

    return pd.DataFrame(processed_rows)


df_features = transform_dataset(df_matches, live_champ_data)

X = df_features.drop(columns=["winner"])
y = df_features["winner"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"Features created:\n{list(X_train.columns)}")
print(X_train.head(3)) # print first three rows of training matrix