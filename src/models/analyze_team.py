from src.feature_engineering.composition_features import get_team_composition_features

# Translate feature comparisons into human-readable text
def generate_explanation(team, champion_data):

    features = get_team_composition_features(team, champion_data)

    reasons = []
    weaknesses = []

    # Team Tankiness
    if features.get("num_tanks", 0) >= 2:
        reasons.append(f"Strong frontline")
    else:
        weaknesses.append("Squishy composition")

    # Team magic power (AP)
    avg_magic = features.get("avg_magic", 0)
    avg_attack = features.get("avg_attack", 0)
    total_offense = avg_magic + avg_attack

    if total_offense > 0:

        magic_ratio = avg_magic / total_offense

        if 0.35 <= magic_ratio <= 0.65:
            reasons.append("Balanced mixed damage (AD/AP)")
        elif magic_ratio > 0.65:
            weaknesses.append("Heavy Magic Damage (easy for enemies to itemize MR)")
        elif magic_ratio < 0.35:
            weaknesses.append("Heavy Physical Damage (easy for enemies to itemize Armor)")

    else:
        weaknesses.append("Unknown damage")

    return {
        "reasons": reasons,
        "weaknesses": weaknesses
    }
