import os
import argparse
import requests
from urllib.parse import quote


API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise ValueError("Missing RIOT_API_KEY. Run: export RIOT_API_KEY='your_key_here'")

REGION = "americas"


def get_puuid(game_name, tag_line):
    game_name_encoded = quote(game_name)
    tag_line_encoded = quote(tag_line)

    url = (
        f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/"
        f"by-riot-id/{game_name_encoded}/{tag_line_encoded}"
    )

    headers = {
        "X-Riot-Token": API_KEY
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.status_code)
        print(response.text)
        response.raise_for_status()

    data = response.json()
    return data["puuid"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-name", required=True)
    parser.add_argument("--tag-line", required=True)
    args = parser.parse_args()

    puuid = get_puuid(args.game_name, args.tag_line)
    print("PUUID:")
    print(puuid)


if __name__ == "__main__":
    main()