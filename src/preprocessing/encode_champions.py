import numpy as np
import pandas as pd
import requests
import joblib
import json
from collections import defaultdict
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
WINRATES_PATH = PROJECT_ROOT / "data" / "champion_winrates.joblib"
FEATURE_MATRICES_PATH = PROJECT_ROOT / "data" / "feature_matrices.json"
live_champ_data = load_champion_data(JSON_PATH)

# champ_ap_ratios, champ_ap_variances, role_freqs, counter_matrix.
# NOTE: this file is static/precomputed (not derived from our train/test
# split the way champion_winrates is). If it was built from the same match
# data that ended up in our test split, it could leak test-set information
# into training features - worth confirming with whoever generated it.
with open(FEATURE_MATRICES_PATH) as f:
    feature_matrices = json.load(f)

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

_BLUE_ROLES = ["blue_top", "blue_jg", "blue_mid", "blue_adc", "blue_sup"]
_RED_ROLES = ["red_top", "red_jg", "red_mid", "red_adc", "red_sup"]

def _normalize_name(name):
    """Lowercase and strip everything but letters/digits, so casing and
    punctuation differences (FiddleSticks/Fiddlesticks, Kai'Sa/Kaisa,
    Bel'Veth/Belveth, ...) collapse to the same comparable string."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())

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
# Using raw_matches.csv (the larger ~15k-row dataset) instead of the older,
# smaller clean_matches.csv. Same core columns (blue_top, blue_jg, ...,
# winner), plus extra metadata columns (match_id, queue_id, game_version,
# game_duration) that we don't use for features but keep around for the
# sanity checks below.
df_matches = pd.read_csv(PROJECT_ROOT / "data" / "raw_matches.csv")

_matches_before = len(df_matches)

# Drop exact duplicate matches, if the same match_id appears more than once
# (e.g. from re-running the crawler over overlapping match ID ranges).
if "match_id" in df_matches.columns:
    df_matches = df_matches.drop_duplicates(subset="match_id")

# Filter out likely remakes: League voids/remakes games that end very early
# (a player disconnects in the first few minutes), and those games don't
# reflect a real completed draft-vs-outcome relationship. 300s (5 min) is a
# conservative cutoff - adjust if you know Riot's exact remake threshold for
# your queue type.
if "game_duration" in df_matches.columns:
    df_matches = df_matches[df_matches["game_duration"] >= 300]

_matches_after = len(df_matches)
print(f"Loaded {_matches_before} raw matches, {_matches_after} remain after "
      f"dropping duplicates/remakes.")


def calculate_champion_winrates(df_raw, smoothing_k=15):
    """
    Computes each champion's empirical win rate (wins / games played) across
    both sides of the draft, using only the matches in df_raw. Must only be
    called on the TRAINING split to avoid leaking test-set outcomes into the
    features.

    Applies Bayesian ("Laplace-style") smoothing toward the global win rate:
        smoothed_winrate = (wins + k * global_winrate) / (games + k)
    Champions with very few appearances in the training data get pulled
    toward the global average instead of keeping an extreme, noisy estimate
    (e.g. a champion picked 3 times with 1 loss would otherwise show a
    "67% win rate" from pure sample noise). k controls how many "virtual"
    average games are blended in; higher k = more smoothing for rare picks.
    """
    wins = defaultdict(int)
    games = defaultdict(int)

    for _, row in df_raw.iterrows():
        blue_won = row["winner"] == "blue"

        for col in _BLUE_ROLES:
            champ = resolve_champion_name(row[col])
            games[champ] += 1
            if blue_won:
                wins[champ] += 1

        for col in _RED_ROLES:
            champ = resolve_champion_name(row[col])
            games[champ] += 1
            if not blue_won:
                wins[champ] += 1

    total_wins = sum(wins.values())
    total_games = sum(games.values())
    global_winrate = total_wins / total_games if total_games > 0 else 0.5

    return {
        champ: (wins[champ] + smoothing_k * global_winrate) / (games[champ] + smoothing_k)
        for champ in games
    }


# Transform data into feature matrix
def transform_dataset(df_raw, champion_data, champion_winrates=None, feature_matrices=None):
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
        match_vector = build_features(
            blue_team, red_team, champion_data, champion_winrates, feature_matrices
        )

        # Add rank, queue, winner lists
        #match_vector["rank"] = row["rank"]
        #match_vector["queue_type"] = row["queue_type"]
        if row["winner"] == "blue":
            match_vector["winner"] = 1
        else:
            match_vector["winner"] = 0

        processed_rows.append(match_vector)

    return pd.DataFrame(processed_rows)


# Split RAW matches first (before feature building) so champion win rates can
# be computed from the training matches only. Using match outcomes from the
# test split to compute win-rate features would leak test-set information
# into the model and inflate evaluation accuracy.
df_matches_train, df_matches_test = train_test_split(
    df_matches, test_size=0.2, random_state=42
)

champion_winrates = calculate_champion_winrates(df_matches_train)

# Save win rates so the live prediction interface (user_interface.py) can use
# the exact same values the model was trained on.
WINRATES_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(champion_winrates, WINRATES_PATH)

df_features_train = transform_dataset(df_matches_train, live_champ_data, champion_winrates, feature_matrices)
df_features_test = transform_dataset(df_matches_test, live_champ_data, champion_winrates, feature_matrices)


#df_features["rank"] = df_features["rank"].fillna("Unranked") # fill empty rank with "Unranked" for normal games
#df_features["rank_encoded"] = df_features["rank"].map(rank_mapping) # Map ranks to numerical values

#df_features = pd.get_dummies(df_features, columns=["queue_type"], drop_first=True, dtype=int) # Use one-hot encoding
#df_features = df_features.drop(columns=["rank"])


X_train = df_features_train.drop(columns=["winner"])
y_train = df_features_train["winner"]

X_test = df_features_test.drop(columns=["winner"])
y_test = df_features_test["winner"]

"""
print(f"X_train shape: {X_train.shape}")
print(f"Features created:\n{list(X_train.columns)}")
print(X_train.head(3)) # print first three rows of training matrix
print(X_train[[col for col in X_train.columns if 'rank' in col or 'queue' in col]].head(3))
"""