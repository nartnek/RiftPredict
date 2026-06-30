import argparse

from src.data_collection.get_puuid import get_puuid
from src.data_collection.collect_matches import collect_matches, clean_and_split


def main():
    parser = argparse.ArgumentParser(
        description="Collect Riot match data and train models."
    )

    parser.add_argument("--game-name", required=True, help="Riot ID game name")
    parser.add_argument("--tag-line", required=True, help="Riot ID tag line")
    parser.add_argument("--count", type=int, default=20, help="Number of matches to collect")
    parser.add_argument(
        "--queue",
        type=int,
        default=None,
        help="Optional queue ID. Example: 420 = ranked solo/duo, 440 = ranked flex."
    )

    args = parser.parse_args()

    print("Getting PUUID...")
    puuid = get_puuid(args.game_name, args.tag_line)
    print(f"PUUID found: {puuid}")

    print("Collecting matches...")
    df = collect_matches(
        puuids=[puuid],
        matches_per_player=args.count,
        queue=args.queue
    )

    print("Saving CSV files...")
    clean_and_split(df)

    print("Training and evaluating models...")

    # Important: import this AFTER the CSV files are created.
    # train_and_evaluate imports X_train, X_test, y_train, y_test from preprocessing.
    from src.models.train_and_evaluate import main as train_models

    train_models()

    print("Full pipeline complete.")


if __name__ == "__main__":
    main()