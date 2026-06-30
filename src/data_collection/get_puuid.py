import os
import argparse
import requests
from urllib.parse import quote

## to run this file: python3 get_puuid.py --game-name MunchyPunchyLOL --tag-line TTV1

API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise ValueError("Missing RIOT_API_KEY. Run: export RIOT_API_KEY='your_key_here'")

REGION = "americas"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-name", required=True, help="Riot ID game name")
    parser.add_argument("--tag-line", required=True, help="Riot ID tag line")
    args = parser.parse_args()

    game_name_encoded = quote(args.game_name)
    tag_line_encoded = quote(args.tag_line)

    url = (
        f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/"
        f"by-riot-id/{game_name_encoded}/{tag_line_encoded}"
    )

    headers = {"X-Riot-Token": API_KEY}

    response = requests.get(url, headers=headers)

    print("Status code:", response.status_code)

    if response.status_code == 200:
        data = response.json()
        print("\nPUUID:")
        print(data["puuid"])
    else:
        print(response.text)

if __name__ == "__main__":
    main()