"""
Predict the winner (and win probability) of a League of Legends draft
using the best model saved by train_and_evaluate.py.

Run using:
    python3 -m src.models.predict
"""

from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering.feature_builder import build_features
from src.feature_engineering.load_champion_data import load_champion_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
CHAMPION_JSON_PATH = PROJECT_ROOT / "data" / "champions.json"

ROLES = ["top", "jg", "mid", "adc", "sup"]
ROLE_LABELS = {
    "top": "Top",
    "jg": "Jungle",
    "mid": "Mid",
    "adc": "ADC",
    "sup": "Support",
}


def load_model_bundle(model_path=MODEL_PATH):
    """
    Loads the {model, model_name, feature_names, metrics} bundle that
    train_and_evaluate.py saves for the best-performing model.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run train_and_evaluate.py (e.g. `python main.py ... --train`) first."
        )

    return joblib.load(model_path)


def _normalize(text):
    """
    Lowercases and strips everything but letters/digits, so 'Kai'Sa',
    'kaisa', and 'Kai Sa' all normalize to the same string.
    """
    return "".join(ch for ch in text.lower() if ch.isalnum())


def build_name_lookup(champion_data):
    """
    Maps a normalized champion name to the exact key used in champion_data
    (and in Riot's match data), accepting either the internal id
    ('MonkeyKing') or the display name ('Wukong') for every champion.
    """
    lookup = {}

    for champion_key, info in champion_data.items():
        for candidate in (champion_key, info.get("name", "")):
            normalized = _normalize(candidate)
            if normalized:
                lookup[normalized] = champion_key

    return lookup


def resolve_champion(raw_name, name_lookup):
    """
    Turns whatever the user typed into the exact champion_data key.
    Raises ValueError with a helpful message if nothing matches.
    """
    normalized = _normalize(raw_name)

    if normalized in name_lookup:
        return name_lookup[normalized]

    raise ValueError(
        f"Unknown champion: '{raw_name}'. Check the spelling and try again."
    )


def prompt_for_draft(name_lookup):
    """
    Interactively asks for all 10 picks and returns a dict keyed
    blue_top, blue_jg, ..., red_sup -> champion_data key.
    """
    draft = {}
    picked = set()

    print(
        "Enter a champion for each pick (either the display name or the "
        "in-game id works, e.g. 'Wukong' or 'MonkeyKing').\n"
    )

    for team in ["blue", "red"]:
        print(f"-- {team.capitalize()} team --")
        for role in ROLES:
            while True:
                raw = input(f"{team.capitalize()} {ROLE_LABELS[role]}: ")

                try:
                    champion_key = resolve_champion(raw, name_lookup)
                except ValueError as error:
                    print(f"  {error}")
                    continue

                if champion_key in picked:
                    print(
                        f"  {champion_key} is already picked in this draft. "
                        "Pick someone else."
                    )
                    continue

                picked.add(champion_key)
                draft[f"{team}_{role}"] = champion_key
                break
        print()

    return draft


def predict_draft(draft, model_bundle, champion_data):
    """
    draft: dict with keys blue_top, blue_jg, blue_mid, blue_adc, blue_sup,
                          red_top,  red_jg,  red_mid,  red_adc,  red_sup
           (values must already be valid champion_data keys — run them
           through resolve_champion first if they came from free-text input).

    Returns (winner, blue_win_probability, red_win_probability).
    """
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    blue_team = [draft[f"blue_{role}"] for role in ROLES]
    red_team = [draft[f"red_{role}"] for role in ROLES]

    feature_vector = build_features(blue_team, red_team, champion_data)
    features_df = pd.DataFrame([feature_vector])

    missing = [name for name in feature_names if name not in features_df.columns]
    if missing:
        raise ValueError(
            f"Built features are missing columns the model expects: {missing}"
        )

    # Enforce the exact column order the model was trained on.
    features_df = features_df[feature_names]

    prediction = model.predict(features_df)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_df)[0]
        class_index = {label: i for i, label in enumerate(model.classes_)}
        blue_prob = probabilities[class_index[1]]
        red_prob = probabilities[class_index[0]]
    else:
        # Fallback for a model with no predict_proba: report a hard 0/100.
        blue_prob = float(prediction)
        red_prob = 1.0 - blue_prob

    winner = "blue" if prediction == 1 else "red"

    return winner, blue_prob, red_prob


def main():
    model_bundle = load_model_bundle()
    champion_data = load_champion_data(CHAMPION_JSON_PATH)
    name_lookup = build_name_lookup(champion_data)

    reported_accuracy = model_bundle["metrics"].get("Accuracy")
    print(
        f"Loaded model: {model_bundle['model_name']} "
        f"(test accuracy: {reported_accuracy})\n"
    )

    draft = prompt_for_draft(name_lookup)

    winner, blue_prob, red_prob = predict_draft(draft, model_bundle, champion_data)

    print("--- Draft ---")
    for team in ["blue", "red"]:
        picks = ", ".join(draft[f"{team}_{role}"] for role in ROLES)
        print(f"{team.capitalize()}: {picks}")

    print("\n--- Prediction ---")
    print(f"Predicted winner: {winner.capitalize()} team")
    print(f"Blue win probability: {blue_prob * 100:.1f}%")
    print(f"Red win probability:  {red_prob * 100:.1f}%")


if __name__ == "__main__":
    main()
