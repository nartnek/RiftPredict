import argparse
import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv(override=True)


REGION = "americas"


def get_api_key():
    """
    Returns the Riot API key from the environment.
    """
    api_key = os.getenv("RIOT_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing RIOT_API_KEY environment variable."
        )

    return api_key


def get_puuid(game_name, tag_line):
    """
    Converts a Riot ID into its PUUID.
    """
    game_name_encoded = quote(game_name, safe="")
    tag_line_encoded = quote(tag_line, safe="")

    url = (
        f"https://{REGION}.api.riotgames.com/"
        "riot/account/v1/accounts/"
        f"by-riot-id/{game_name_encoded}/{tag_line_encoded}"
    )

    headers = {
        "X-Riot-Token": get_api_key()
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()
    return data["puuid"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game-name",
        required=True,
    )
    parser.add_argument(
        "--tag-line",
        required=True,
    )

    args = parser.parse_args()

    puuid = get_puuid(
        args.game_name,
        args.tag_line,
    )

    print("PUUID:")
    print(puuid)


if __name__ == "__main__":
    main()