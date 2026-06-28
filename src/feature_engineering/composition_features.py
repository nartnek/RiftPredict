def get_team_composition_features(team_champions,champion_data):
    features = {
        "num_fighters": 0,
        "num_tanks": 0,
        "num_mages": 0,
        "num_assassins": 0,
        "num_marksmen": 0,
        "num_supports": 0,

        "num_ranged": 0,
        "num_melee": 0,

        "avg_attack": 0.0,
        "avg_defense": 0.0,
        "avg_magic": 0.0,
        "avg_difficulty": 0.0,

        "avg_hp": 0.0,
        "avg_armor": 0.0,
        "avg_attackdamage": 0.0
    }

    num_champions = len(team_champions)

    for champion in team_champions:

        champion_info = champion_data[champion]

        tags = champion_info["tags"]

        if "Fighter" in tags:
            features["num_fighters"] += 1

        if "Tank" in tags:
            features["num_tanks"] += 1

        if "Mage" in tags:
            features["num_mages"] += 1

        if "Assassin" in tags:
            features["num_assassins"] += 1

        if "Marksman" in tags:
            features["num_marksmen"] += 1

        if "Support" in tags:
            features["num_supports"] += 1

        stats = champion_info["stats"]

        if stats["attackrange"] >= 400:
            features["num_ranged"] += 1
        else:
            features["num_melee"] += 1

        features["avg_attack"] += champion_info["info"]["attack"]
        features["avg_defense"] += champion_info["info"]["defense"]
        features["avg_magic"] += champion_info["info"]["magic"]
        features["avg_difficulty"] += champion_info["info"]["difficulty"]

        features["avg_hp"] += stats["hp"]
        features["avg_armor"] += stats["armor"]
        features["avg_attackdamage"] += stats["attackdamage"]

    for key in [
        "avg_attack",
        "avg_defense",
        "avg_magic",
        "avg_difficulty",
        "avg_hp",
        "avg_armor",
        "avg_attackdamage"
    ]:
        features[key] /= num_champions

    return features