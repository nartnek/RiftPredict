import argparse

from src.data_collection.get_puuid import get_puuid
from src.data_collection.collect_matches import collect_matches, clean_and_split

# To run:
# python3 main.py \
#   --game-name MunchyPunchyLOL \
#   --tag-line TTV1 \
#   --target 1000 \
#   --matches-per-player 30

def main():
    parser = argparse.ArgumentParser(
        description="Collect Riot match data and train models."
    )

    parser.add_argument("--game-name", required=True, help="Riot ID game name")
    parser.add_argument("--tag-line", required=True, help="Riot ID tag line")
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Total number of valid unique matches to collect.",
    )

    parser.add_argument(
        "--matches-per-player",
        type=int,
        default=50,
        help="Number of recent matches checked for each discovered player.",
    )
    
    parser.add_argument(
        "--queue",
        type=int,
        default=420,
        help="Queue ID. Default: 420 = ranked solo/duo; 440 = ranked flex.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and evaluate models after collecting data.",
    )

    args = parser.parse_args()

    print("Getting PUUID...")
    puuid = get_puuid(args.game_name, args.tag_line)
    print(f"PUUID found: {puuid}")

    print("Collecting matches...")
    df = collect_matches(
        puuids=[puuid],
        target=args.target,
        matches_per_player=args.matches_per_player,
        queue=args.queue,
    )

    print("Saving CSV files...")
    clean_df = clean_and_split(
        df,
        required_queue=args.queue,
    )

    print(f"Raw matches: {len(df)}")
    print(f"Valid matches: {len(clean_df)}")
    print(f"Rejected matches: {len(df) - len(clean_df)}")

    if args.train:
        print("Training and evaluating models...")

        from src.models.train_and_evaluate import main as train_models

        train_models()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()