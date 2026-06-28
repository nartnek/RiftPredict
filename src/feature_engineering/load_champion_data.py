import json

def load_champion_data(filepath):
    with open(filepath, "r", encoding="utf-8") as filep:
        champion_file = json.load(filep)

    return champion_file["data"]