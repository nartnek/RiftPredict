import json
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
CRAWLER_STATE_PATH = "data/crawler_state.json"


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


def load_crawler_state(
    queue,
    path=CRAWLER_STATE_PATH,
):
    """
    Loads the crawler's progress through player PUUIDs.

    Returns None when no compatible state exists.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError):
        print("Crawler state could not be loaded.")
        return None

    # Do not reuse a state file from another queue.
    if state.get("queue") != queue:
        print(
            "Crawler state belongs to a different queue. "
            "Starting a new crawler state."
        )
        return None

    return state


def save_crawler_state(
    visited_puuids,
    puuids_to_visit,
    seen_match_ids,
    queue,
    current_puuid=None,
    path=CRAWLER_STATE_PATH,
):
    """
    Saves crawler traversal progress so restarting does not require
    rediscovering participants from old matches.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "queue": queue,
        "visited_puuids": list(visited_puuids),
        "puuids_to_visit": list(puuids_to_visit),
        "seen_match_ids": list(seen_match_ids),
        "current_puuid": current_puuid,
    }

    # Write to a temporary file first so an interrupted write is
    # less likely to corrupt the real state file.
    temporary_path = f"{path}.tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(state, file)

    os.replace(temporary_path, path)


def bootstrap_participants(
    match_ids,
    puuids_to_visit,
    queued_puuids,
    visited_puuids,
    number_of_matches=5,
):
    """
    One-time migration helper.

    Older checkpoints contain match rows but not participant PUUIDs.
    If no crawler_state.json exists yet, read only a few old matches
    to recover enough participant PUUIDs to restart the crawler.

    Future restarts use crawler_state.json instead.
    """
    if not match_ids:
        return

    bootstrap_ids = list(match_ids)[:number_of_matches]

    print(
        "\nNo crawler state was found."
        "\nPerforming one-time participant bootstrap from "
        f"{len(bootstrap_ids)} existing matches..."
    )

    for index, match_id in enumerate(
        bootstrap_ids,
        start=1,
    ):
        print(
            f"Bootstrap match {index}/{len(bootstrap_ids)}: "
            f"{match_id}"
        )

        match_json = get_match_data(match_id)

        participant_puuids = extract_participant_puuids(
            match_json
        )

        for participant_puuid in participant_puuids:
            if (
                participant_puuid not in visited_puuids
                and participant_puuid not in queued_puuids
            ):
                puuids_to_visit.append(participant_puuid)
                queued_puuids.add(participant_puuid)


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

    Match data and crawler traversal state are both checkpointed,
    allowing collection to resume after interruption without
    re-downloading existing matches.
    """

    # ------------------------------------------------------------
    # Load saved matches
    # ------------------------------------------------------------

    loaded_rows = load_checkpoint()

    loaded_by_id = {
        row["match_id"]: row
        for row in loaded_rows
        if row.get("match_id")
    }

    seen_match_ids = set(loaded_by_id)

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

    # ------------------------------------------------------------
    # Load saved crawler traversal state
    # ------------------------------------------------------------

    state = load_crawler_state(queue)

    if state is not None:
        visited_puuids = set(
            state.get("visited_puuids", [])
        )

        puuids_to_visit = deque(
            state.get("puuids_to_visit", [])
        )

        # Include previously processed invalid matches too.
        seen_match_ids.update(
            state.get("seen_match_ids", [])
        )

        queued_puuids = set(puuids_to_visit)

        # If the program stopped while processing one player,
        # put that player back at the front of the queue.
        previous_current_puuid = state.get(
            "current_puuid"
        )

        if (
            previous_current_puuid
            and previous_current_puuid
            not in visited_puuids
            and previous_current_puuid
            not in queued_puuids
        ):
            puuids_to_visit.appendleft(
                previous_current_puuid
            )
            queued_puuids.add(
                previous_current_puuid
            )

        # Also make sure the supplied seed PUUID still exists
        # somewhere in the traversal.
        for seed_puuid in puuids:
            if (
                seed_puuid not in visited_puuids
                and seed_puuid not in queued_puuids
            ):
                puuids_to_visit.append(seed_puuid)
                queued_puuids.add(seed_puuid)

        print(
            "Loaded crawler state:"
            f"\n  Visited players: {len(visited_puuids)}"
            f"\n  Players waiting: {len(puuids_to_visit)}"
            f"\n  Known match IDs: {len(seen_match_ids)}"
        )

    else:
        # No saved crawler traversal exists yet.
        puuids_to_visit = deque(puuids)
        queued_puuids = set(puuids)
        visited_puuids = set()

        # Your old CSV contains match/champion information but
        # does not contain participant PUUIDs.
        #
        # Therefore, the first time you upgrade to this system,
        # retrieve only a handful of existing matches to seed
        # the player queue.
        if loaded_by_id:
            bootstrap_participants(
                match_ids=loaded_by_id.keys(),
                puuids_to_visit=puuids_to_visit,
                queued_puuids=queued_puuids,
                visited_puuids=visited_puuids,
                number_of_matches=5,
            )

        save_crawler_state(
            visited_puuids=visited_puuids,
            puuids_to_visit=puuids_to_visit,
            seen_match_ids=seen_match_ids,
            queue=queue,
        )

    # ------------------------------------------------------------
    # Crawl
    # ------------------------------------------------------------

    new_valid_matches = 0
    current_puuid = None

    try:
        while (
            puuids_to_visit
            and len(rows) < target
        ):
            current_puuid = puuids_to_visit.popleft()
            queued_puuids.discard(current_puuid)

            if current_puuid in visited_puuids:
                current_puuid = None
                continue

            print(
                f"\nChecking player "
                f"{len(visited_puuids) + 1} | "
                f"Valid matches: {len(rows)}/{target}"
            )

            try:
                match_ids = get_match_ids(
                    current_puuid,
                    count=matches_per_player,
                    queue=queue,
                )

            except RuntimeError:
                # Most importantly: do not lose this player if
                # the API key expires.
                if current_puuid not in queued_puuids:
                    puuids_to_visit.appendleft(
                        current_puuid
                    )
                    queued_puuids.add(
                        current_puuid
                    )

                save_crawler_state(
                    visited_puuids=visited_puuids,
                    puuids_to_visit=puuids_to_visit,
                    seen_match_ids=seen_match_ids,
                    queue=queue,
                    current_puuid=current_puuid,
                )

                raise

            except Exception as error:
                print(
                    "Could not retrieve matches for player: "
                    f"{error}"
                )

                # Put it at the back so it can be retried later.
                if current_puuid not in queued_puuids:
                    puuids_to_visit.append(
                        current_puuid
                    )
                    queued_puuids.add(
                        current_puuid
                    )

                save_crawler_state(
                    visited_puuids=visited_puuids,
                    puuids_to_visit=puuids_to_visit,
                    seen_match_ids=seen_match_ids,
                    queue=queue,
                    current_puuid=current_puuid,
                )

                current_puuid = None
                continue

            # ----------------------------------------------------
            # Process this player's matches
            # ----------------------------------------------------

            target_reached = False

            for match_id in match_ids:

                # THIS IS THE IMPORTANT CHANGE:
                #
                # Existing matches are skipped immediately.
                # We do NOT call get_match_data() again.
                if match_id in seen_match_ids:
                    continue

                print(
                    f"Downloading {match_id} "
                    f"({len(rows)}/{target} valid)"
                )

                # Mark it before processing so it is not downloaded
                # twice during the same run.
                seen_match_ids.add(match_id)

                try:
                    match_json = get_match_data(match_id)

                except RuntimeError:
                    # The request never completed successfully,
                    # so allow this match to be retried next run.
                    seen_match_ids.discard(match_id)

                    if current_puuid not in queued_puuids:
                        puuids_to_visit.appendleft(
                            current_puuid
                        )
                        queued_puuids.add(
                            current_puuid
                        )

                    save_crawler_state(
                        visited_puuids=visited_puuids,
                        puuids_to_visit=puuids_to_visit,
                        seen_match_ids=seen_match_ids,
                        queue=queue,
                        current_puuid=current_puuid,
                    )

                    raise

                except Exception as error:
                    print(
                        f"Skipping {match_id}: {error}"
                    )

                    # It may have been a temporary error.
                    seen_match_ids.discard(match_id)
                    continue

                # ------------------------------------------------
                # Discover new players
                # ------------------------------------------------

                participant_puuids = (
                    extract_participant_puuids(
                        match_json
                    )
                )

                for participant_puuid in participant_puuids:
                    if participant_puuid == current_puuid:
                        continue

                    if (
                        participant_puuid
                        not in visited_puuids
                        and participant_puuid
                        not in queued_puuids
                    ):
                        puuids_to_visit.append(
                            participant_puuid
                        )

                        queued_puuids.add(
                            participant_puuid
                        )

                # ------------------------------------------------
                # Extract training row
                # ------------------------------------------------

                row = extract_match_row(match_json)

                if not is_valid_match_row(
                    row,
                    required_queue=queue,
                ):
                    print(
                        f"Rejected incomplete match: "
                        f"{match_id}"
                    )
                    continue

                rows.append(row)
                new_valid_matches += 1

                # Periodically save BOTH the dataset and crawler.
                if (
                    new_valid_matches
                    % checkpoint_every
                    == 0
                ):
                    save_checkpoint(rows)

                    save_crawler_state(
                        visited_puuids=visited_puuids,
                        puuids_to_visit=puuids_to_visit,
                        seen_match_ids=seen_match_ids,
                        queue=queue,
                        current_puuid=current_puuid,
                    )

                if len(rows) >= target:
                    # Do not mark the player as completely visited.
                    # If you later raise the target, the crawler can
                    # query this player again and continue through
                    # remaining unseen match IDs.
                    if current_puuid not in queued_puuids:
                        puuids_to_visit.appendleft(
                            current_puuid
                        )
                        queued_puuids.add(
                            current_puuid
                        )

                    target_reached = True
                    break

                # Reduce the likelihood of exceeding Riot API limits.
                time.sleep(1.3)

            if target_reached:
                break

            # This player's returned match list was fully processed.
            visited_puuids.add(current_puuid)
            current_puuid = None

            # Saving after every completed player is cheap and makes
            # restarts much more reliable.
            save_crawler_state(
                visited_puuids=visited_puuids,
                puuids_to_visit=puuids_to_visit,
                seen_match_ids=seen_match_ids,
                queue=queue,
                current_puuid=None,
            )

    finally:
        # Runs for normal exit, Ctrl+C, expired API key, etc.
        save_checkpoint(rows)

        save_crawler_state(
            visited_puuids=visited_puuids,
            puuids_to_visit=puuids_to_visit,
            seen_match_ids=seen_match_ids,
            queue=queue,
            current_puuid=current_puuid,
        )

    print("\nCollection finished:")
    print(f"Visited players: {len(visited_puuids)}")
    print(f"Known match IDs: {len(seen_match_ids)}")
    print(f"Players waiting: {len(puuids_to_visit)}")
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