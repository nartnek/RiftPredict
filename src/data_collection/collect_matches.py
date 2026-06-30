import os
import time
import argparse
import requests
import pandas as pd


API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise ValueError("Missing RIOT_API_KEY environment variable.")


HEADERS = {
    "X-Riot-Token": API_KEY
}


REGION = "americas"  # For NA, use americas for Match-V5


def riot_get(url, params=None):
    """
    Sends a GET request to Riot API.
    Handles simple rate limiting.
    """
    while True:
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code == 429:
            wait_time = int(response.headers.get("Retry-After", 5))
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue

        if response.status_code != 200:
            print("Error:", response.status_code)
            print(response.text)
            response.raise_for_status()

        return response.json()


def get_match_ids(puuid, count=20, queue=None):
    """
    Gets match IDs for one player by PUUID.
    Riot Match-V5 has the endpoint:
    /lol/match/v5/matches/by-puuid/{puuid}/ids
    """
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"

    params = {
        "start": 0,
        "count": count
    }

    if queue is not None:
        params["queue"] = queue

    return riot_get(url, params=params)


def get_match_data(match_id):
    """
    Gets full match data from one match ID.
    """
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def extract_match_row(match_json):
    """
    Extract useful info from Riot's match JSON.
    Creates one row for the CSV.
    """

    info = match_json["info"]
    metadata = match_json["metadata"]

    row = {
        "match_id": metadata["matchId"],
        "queue_id": info.get("queueId"),
        "game_version": info.get("gameVersion"),
        "game_duration": info.get("gameDuration"),
    }

    # Determine winner
    winner_team = None
    for team in info["teams"]:
        if team["win"] is True:
            winner_team = team["teamId"]

    if winner_team == 100:
        row["winner"] = "blue"
    elif winner_team == 200:
        row["winner"] = "red"
    else:
        row["winner"] = None

    # Default values
    roles = ["top", "jg", "mid", "adc", "sup"]

    for role in roles:
        row[f"blue_{role}"] = None
        row[f"red_{role}"] = None

    position_map = {
        "TOP": "top",
        "JUNGLE": "jg",
        "MIDDLE": "mid",
        "BOTTOM": "adc",
        "UTILITY": "sup"
    }

    for player in info["participants"]:
        team_id = player["teamId"]
        position = player.get("teamPosition")
        champion = player.get("championName")

        if position not in position_map:
            continue

        role = position_map[position]

        if team_id == 100:
            row[f"blue_{role}"] = champion
        elif team_id == 200:
            row[f"red_{role}"] = champion

    return row


def collect_matches(puuids, matches_per_player=20, queue=None):
    """
    Collects matches from multiple PUUIDs.
    Removes duplicate match IDs.
    """
    all_match_ids = set()

    for puuid in puuids:
        print(f"Getting match IDs for PUUID: {puuid}")
        ids = get_match_ids(puuid, count=matches_per_player, queue=queue)
        all_match_ids.update(ids)

    print(f"Total unique matches found: {len(all_match_ids)}")

    rows = []

    for i, match_id in enumerate(sorted(all_match_ids), start=1):
        print(f"[{i}/{len(all_match_ids)}] Downloading {match_id}")
        match_json = get_match_data(match_id)
        row = extract_match_row(match_json)
        rows.append(row)

        # Small delay to avoid hitting rate limits too fast
        time.sleep(1)

    return pd.DataFrame(rows)


def clean_and_split(df):
    """
    Saves raw, clean, train, and test CSVs.
    """

    os.makedirs("data", exist_ok=True)

    df.to_csv("data/raw_matches.csv", index=False)

    needed_columns = [
        "match_id",
        "blue_top", "blue_jg", "blue_mid", "blue_adc", "blue_sup",
        "red_top", "red_jg", "red_mid", "red_adc", "red_sup",
        "winner"
    ]

    clean_df = df.drop_duplicates(subset=["match_id"])
    clean_df = clean_df.dropna(subset=needed_columns)

    clean_df.to_csv("data/clean_matches.csv", index=False)

    train_df = clean_df.sample(frac=0.8, random_state=42)
    test_df = clean_df.drop(train_df.index)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Saved:")
    print("data/raw_matches.csv")
    print("data/clean_matches.csv")
    print("data/train.csv")
    print("data/test.csv")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--puuids",
        nargs="+",
        required=True,
        help="One or more Riot PUUIDs"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of matches per PUUID"
    )

    parser.add_argument(
        "--queue",
        type=int,
        default=None,
        help="Optional queue ID. Example: 420 for ranked solo/duo, 440 for ranked flex."
    )

    args = parser.parse_args()

    df = collect_matches(
        puuids=args.puuids,
        matches_per_player=args.count,
        queue=args.queue
    )

    clean_and_split(df)


if __name__ == "__main__":
    main()