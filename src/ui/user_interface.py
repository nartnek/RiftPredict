# Full user interface, prompting the user for all 10 players in the game
# outputs predicted winner of the game with reasoning

# run: python -m src.ui.user_interface

import joblib
import pandas as pd
from pathlib import Path

from src.feature_engineering.feature_builder import build_features
from src.feature_engineering.load_champion_data import load_champion_data
from src.models.analyze_team import generate_explanation
from src.preprocessing.encode_champions import resolve_champion_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_model.joblib"
JSON_PATH = PROJECT_ROOT / "data" / "champions.json"

def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run 'train_and_evaluate.py' first"
        )

    # Load model and champion dataset
    model = joblib.load(MODEL_PATH)
    champion_data = load_champion_data(JSON_PATH)

    # Extract required feature names directly from the fitted model
    feature_columns = model.feature_names_in_

    return champion_data, model, feature_columns

def get_team_inputs(team_name, roles, champion_data, picked_champions):
    team = []

    for role in roles:
        while True:
            pick = input(f"  {team_name} {role.capitalize()}: ").strip()

            # Empty selection
            if not pick:
                print("  Selection cannot be empty. Try again.")
                continue

            try:
                # resolve name so Cho'gath = Chogath, Wukong = MonkeyKing, etc
                resolved_pick = resolve_champion_name(pick)

                # Exact name match or resolved name match
                if resolved_pick not in champion_data:
                    print(f"'{pick}' is not a valid champion name. Try again.")
                    continue

                # A champion can only be picked once per match (both teams combined)
                if resolved_pick in picked_champions:
                    print(f"'{pick}' has already been picked this match. Try again.")
                    continue

                team.append(resolved_pick)
                picked_champions.add(resolved_pick)
                break

            except KeyError:
                print(f"'{pick}' is not a valid champion name. Try again.")

    return team

def main():
    # Load champion data and trained model
    champion_data, model, feature_columns = load_artifacts()

    roles = ["Top", "Jungle", "Mid", "ADC", "Support"]

    print("\n==========================================")
    print("               RIFTPREDICT                ")
    print("==========================================")
    print("Please enter the full draft:")
    print("")

    # Tracks every champion picked so far across both teams
    picked_champions = set()

    # Prompt user for selections
    blue_team = get_team_inputs("Blue Team", roles, champion_data, picked_champions)
    print("")
    red_team = get_team_inputs("Red Team", roles, champion_data, picked_champions)

    # Obtain features dictionary for each team
    try:
        features_dict = build_features(blue_team, red_team, champion_data)
    except KeyError as e:
        print(f"Error building features")
        return

    input_df = pd.DataFrame([features_dict])[feature_columns]

    probabilities = model.predict_proba(input_df)[0]  # [Prob_Red_Win, Prob_Blue_Win]
    prediction = model.predict(input_df)[0]           # 1 = Blue, 0 = Red

    blue_prob = probabilities[1] * 100
    red_prob = probabilities[0] * 100

    blue_analysis = generate_explanation(blue_team, champion_data)
    red_analysis  = generate_explanation(red_team, champion_data)

    if prediction == 1:
        print(f"\nPredicted Winner: BLUE TEAM ({blue_prob:.1f}% confidence)")
        print("\n🔵 Blue Team Strengths:", blue_analysis["reasons"])
        print("🔴 Red Team Weaknesses:", red_analysis["weaknesses"])
    else:
        print(f"\nPredicted Winner: RED TEAM ({red_prob:.1f}% confidence)")
        print("\n🔴 Red Team Strengths:", red_analysis["reasons"])
        print("🔵 Blue Team Weaknesses:", blue_analysis["weaknesses"])


    print(f"\n📊 Win Chances:")
    print(f"   • Blue Team: {blue_prob:.1f}%")
    print(f"   • Red Team:  {red_prob:.1f}%")
    print("==========================================\n")


if __name__ == "__main__":
    main()