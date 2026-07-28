import os
import time

import pandas as pd
import requests
from collections import deque
from dotenv import load_dotenv

load_dotenv(override=True)


def get_api_key():
    api_key = os.getenv("RIOT_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing RIOT_API_KEY environment variable."
        )

    return api_key


REGION = "americas"  # For NA, use americas for Match-V5


def riot_get(url, params=None, max_retries=5):
    headers = {
        "X-Riot-Token": get_api_key()
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=20,
            )
        except requests.RequestException as error:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Network error: {error}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            wait_time = int(
                response.headers.get("Retry-After", 10)
            )
            print(
                f"Rate limited. Waiting {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue

        if response.status_code in {500, 502, 503, 504}:
            wait_time = 2 ** attempt
            print(
                f"Riot API returned {response.status_code}. "
                f"Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue

        if response.status_code in {401, 403}:
            raise RuntimeError(
                "The Riot API key is missing, invalid, or expired."
            )

        response.raise_for_status()

    raise RuntimeError(
        f"Request failed after {max_retries} attempts: {url}"
    )


def get_match_ids(puuid, count=20, queue=420):
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


def extract_participant_puuids(match_json):
    """
    Returns every participant PUUID contained in a match.
    """
    participants = match_json.get("info", {}).get("participants", [])

    return [
        participant["puuid"]
        for participant in participants
        if participant.get("puuid")
    ]


DRAFT_COLUMNS = [
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


def is_valid_match_row(row, required_queue=420):
    """
    Returns True when a match contains a complete ranked draft
    and a valid winner.
    """
    if row.get("queue_id") != required_queue:
        return False

    if row.get("winner") not in {"blue", "red"}:
        return False

    for column in DRAFT_COLUMNS:
        value = row.get(column)

        if value is None or pd.isna(value):
            return False

        if not str(value).strip():
            return False

    return True



CHECKPOINT_PATH = "data/raw_matches.csv"


def load_checkpoint(path=CHECKPOINT_PATH):
    """
    Loads previously collected matches so collection can resume.
    """
    if not os.path.exists(path):
        return []

    checkpoint_df = pd.read_csv(path)

    if checkpoint_df.empty:
        return []

    if "match_id" not in checkpoint_df.columns:
        raise ValueError(
            f"Checkpoint file {path} has no match_id column."
        )

    rows = checkpoint_df.to_dict(orient="records")

    print(f"Loaded checkpoint: {len(rows)} existing matches")
    return rows


def save_checkpoint(rows, path=CHECKPOINT_PATH):
    """
    Saves collected matches while the collector is still running.
    """
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint_df = pd.DataFrame(rows)
    checkpoint_df = checkpoint_df.drop_duplicates(
        subset=["match_id"]
    )

    checkpoint_df.to_csv(path, index=False)

    print(
        f"Checkpoint saved: "
        f"{len(checkpoint_df)} total matches"
    )


def collect_matches(
    puuids,
    target=1000,
    matches_per_player=20,
    queue=420,
    checkpoint_every=10,
):
    """
    Crawls through match participants until the requested number
    of valid unique matches has been collected.

    Existing checkpoint data is loaded so collection can resume.
    """
    loaded_rows = load_checkpoint()

    # Deduplicate previously saved rows by match ID.
    loaded_by_id = {
        row["match_id"]: row
        for row in loaded_rows
        if row.get("match_id")
    }

    # Remember every previously processed match.
    seen_match_ids = set(loaded_by_id)

    # Only valid matches count toward the target.
    rows = [
        row
        for row in loaded_by_id.values()
        if is_valid_match_row(
            row,
            required_queue=queue,
        )
    ]

    print(
        f"Valid checkpoint matches for queue {queue}: "
        f"{len(rows)}"
    )

    if len(rows) >= target:
        print(
            f"Target already reached: "
            f"{len(rows)}/{target} valid matches"
        )
        return pd.DataFrame(rows)

    puuids_to_visit = deque(puuids)
    queued_puuids = set(puuids)
    visited_puuids = set()

    new_valid_matches = 0

    try:
        while puuids_to_visit and len(rows) < target:
            puuid = puuids_to_visit.popleft()
            queued_puuids.discard(puuid)

            if puuid in visited_puuids:
                continue

            visited_puuids.add(puuid)

            print(
                f"\nChecking player {len(visited_puuids)} | "
                f"Valid matches: {len(rows)}/{target}"
            )

            try:
                match_ids = get_match_ids(
                    puuid,
                    count=matches_per_player,
                    queue=queue,
                )
            except RuntimeError:
                # Stop on an expired/invalid key or exhausted retries.
                raise
            except Exception as error:
                print(
                    f"Could not retrieve matches for player: "
                    f"{error}"
                )
                continue

            for match_id in match_ids:
                is_existing_match = match_id in seen_match_ids

                if is_existing_match:
                    print(
                        f"Reading existing match for participant discovery: "
                        f"{match_id}"
                    )
                else:
                    seen_match_ids.add(match_id)

                    print(
                        f"Downloading {match_id} "
                        f"({len(rows)}/{target} valid)"
                    )

                try:
                    match_json = get_match_data(match_id)
                except RuntimeError:
                    raise
                except Exception as error:
                    print(f"Skipping {match_id}: {error}")
                    continue

                # Discover additional players automatically.
                for participant_puuid in (
                    extract_participant_puuids(match_json)
                ):
                    if (
                        participant_puuid not in visited_puuids
                        and participant_puuid not in queued_puuids
                    ):
                        puuids_to_visit.append(participant_puuid)
                        queued_puuids.add(participant_puuid)

                if is_existing_match:
                    continue

                row = extract_match_row(match_json)

                if not is_valid_match_row(
                    row,
                    required_queue=queue,
                ):
                    print(
                        f"Rejected incomplete match: {match_id}"
                    )
                    continue

                rows.append(row)
                new_valid_matches += 1

                if (
                    new_valid_matches
                    % checkpoint_every
                    == 0
                ):
                    save_checkpoint(rows)

                if len(rows) >= target:
                    break

                # Reduce the likelihood of exceeding rate limits.
                time.sleep(1.3)

    finally:
        # Save progress even when the program stops because of
        # an expired key, network failure, or Ctrl+C.
        save_checkpoint(rows)

    print("\nCollection finished:")
    print(f"Visited players: {len(visited_puuids)}")
    print(f"Processed match IDs: {len(seen_match_ids)}")
    print(f"New valid matches: {new_valid_matches}")
    print(f"Total valid matches: {len(rows)}")

    return pd.DataFrame(rows)




def clean_and_split(df, required_queue=420):
    os.makedirs("data", exist_ok=True)

    if df.empty:
        print("No matches were collected.")
        return pd.DataFrame()

    if "match_id" not in df.columns:
        raise ValueError(
            "Collected data does not contain a match_id column."
        )

    df.to_csv("data/raw_matches.csv", index=False)

    clean_df = df.drop_duplicates(subset=["match_id"]).copy()

    clean_df = clean_df[
        clean_df.apply(
            lambda row: is_valid_match_row(
                row.to_dict(),
                required_queue=required_queue,
            ),
            axis=1,
        )
    ]

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

    return clean_df