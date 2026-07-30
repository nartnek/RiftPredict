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

# --- Champion name resolution ----------------------------------------------
# Riot's Match API and Data Dragon don't always agree on the exact string
# for a champion:
#   - Casing differs for some champs, e.g. the Match API's championName
#     "FiddleSticks" vs. Data Dragon's key "Fiddlesticks" (a known,
#     long-standing mismatch in Riot's own data).
#   - Punctuation/apostrophes are stripped inconsistently in Data Dragon
#     keys, e.g. "Kai'Sa" -> "Kaisa", "Bel'Veth" -> "Belveth".
#   - Wukong is a genuine outlier, not just a casing issue: Data Dragon's
#     display name is "Wukong", but the key champion_data actually stores
#     him under is "MonkeyKing". No amount of normalizing "Wukong" will
#     ever produce "MonkeyKing", so this needs a manual mapping.
#
# champion_data[champion] elsewhere in the pipeline (inside build_features)
# needs an exact match against champion_data's keys, so every champion name
# coming out of clean_matches.csv is resolved to the correct key here, once,
# before any features get built -- rather than trusting the raw CSV string.

def _normalize_name(name):
    """Lowercase and strip everything but letters/digits, so casing and
    punctuation differences (FiddleSticks/Fiddlesticks, Kai'Sa/Kaisa,
    Bel'Veth/Belveth, ...) collapse to the same comparable string."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())

# Known cases where the raw name isn't just a casing/punctuation variant of
# the champion_data key. Add more here if you hit a similar mismatch.
_MANUAL_NAME_OVERRIDES = {
    "wukong": "MonkeyKing",
}

_CHAMPION_ALIASES = {
    _normalize_name(key): key for key in live_champ_data.keys()
}

def resolve_champion_name(raw_name):
    """
    Map a champion name as it appears in clean_matches.csv to the exact
    key used in champion_data (live_champ_data), so downstream lookups
    like champion_data[champion] never miss due to a naming mismatch.
    """
    normalized = _normalize_name(raw_name)

    if normalized in _MANUAL_NAME_OVERRIDES:
        return _MANUAL_NAME_OVERRIDES[normalized]

    resolved = _CHAMPION_ALIASES.get(normalized)
    if resolved is None:
        raise KeyError(
            f"Champion '{raw_name}' has no match in champion_data (loaded "
            f"from {JSON_PATH}) even after normalizing case/punctuation. "
            "Check that data/champions.json is up to date and contains "
            "this champion, or add a manual override to "
            "_MANUAL_NAME_OVERRIDES if it's a known naming quirk like "
            "Wukong/MonkeyKing."
        )
    return resolved


# Implement real match data from riot API
df_matches = pd.read_csv(PROJECT_ROOT / "data" / "clean_matches.csv")

# Transform data into feature matrix
def transform_dataset(df_raw, champion_data):
    processed_rows = []

    # For each row in the dataset
    # assemble blue and red team lists 
    for _, row in df_raw.iterrows():

        blue_team = [
            resolve_champion_name(row["blue_top"]),
            resolve_champion_name(row["blue_jg"]),
            resolve_champion_name(row["blue_mid"]),
            resolve_champion_name(row["blue_adc"]),
            resolve_champion_name(row["blue_sup"]),
        ]
        red_team = [
            resolve_champion_name(row["red_top"]),
            resolve_champion_name(row["red_jg"]),
            resolve_champion_name(row["red_mid"]),
            resolve_champion_name(row["red_adc"]),
            resolve_champion_name(row["red_sup"]),
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