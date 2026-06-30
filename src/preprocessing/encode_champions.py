import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.feature_engineering.composition_features import (
    get_team_composition_features,
)
from src.feature_engineering.feature_builder import build_features
from src.feature_engineering.load_champion_data import load_champion_data

# Run using:
# python -m src.preprocessing.encode_champions

# Get champion data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = PROJECT_ROOT / "data" / "champions.json"
live_champ_data = load_champion_data(JSON_PATH)

#rank_mapping = {
#    "Unranked": 0, # normal games
#    "Iron": 1, 
#    "Bronze": 2, 
#    "Silver": 3, 
#    "Gold": 4, 
#    "Platinum": 5, 
#    "Emerald": 6, 
#    "Diamond": 7, 
#    "Master": 8, 
#    "Grandmaster": 9, 
#    "Challenger": 10
#}

roles = [
    "blue_top",
    "blue_jg",
    "blue_mid",
    "blue_adc",
    "blue_sup",
    "red_top",
    "red_jg",
    "red_mid",
    "red_adc",
    "red_sup",
]

# Implement real match data from riot API
df_matches = pd.read_csv("clean_matches.csv")

# Transform data into feature matrix
def transform_dataset(df_raw, champion_data):
    processed_rows = []

    # For each row in the dataset
    # assemble blue and red team lists 
    for _, row in df_raw.iterrows():

        blue_team = [
            row["blue_top"],
            row["blue_jg"],
            row["blue_mid"],
            row["blue_adc"],
            row["blue_sup"],
        ]
        red_team = [
            row["red_top"],
            row["red_jg"],
            row["red_mid"],
            row["red_adc"],
            row["red_sup"],
        ]

        # Map champion text names to numerical stats
        match_vector = build_features(blue_team, red_team, champion_data)

        # Add rank, queue, winner lists
        #match_vector["rank"] = row["rank"]
        #match_vector["queue_type"] = row["queue_type"]
        if row["winner"] == "blue":
            match_vector["winner"] = 1
        else:
            match_vector["winner"] = 0

        processed_rows.append(match_vector)

    return pd.DataFrame(processed_rows)


df_features = transform_dataset(df_matches, live_champ_data)


#df_features["rank"] = df_features["rank"].fillna("Unranked") # fill empty rank with "Unranked" for normal games
#df_features["rank_encoded"] = df_features["rank"].map(rank_mapping) # Map ranks to numerical values

#df_features = pd.get_dummies(df_features, columns=["queue_type"], drop_first=True, dtype=int) # Use one-hot encoding
#df_features = df_features.drop(columns=["rank"])


X = df_features.drop(columns=["winner"]) # drop old winner column
y = df_features["winner"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"Features created:\n{list(X_train.columns)}")
print(X_train.head(3)) # print first three rows of training matrix
print(X_train[[col for col in X_train.columns if 'rank' in col or 'queue' in col]].head(3))
