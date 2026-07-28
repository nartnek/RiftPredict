# unit_tests/data_collection/test_clean_and_split.py

import pandas as pd

import src.data_collection.collect_matches as cm


def make_valid_row(match_id="NA1_TEST"):
    """
    Creates a complete row that should pass validation.
    """
    return {
        "match_id": match_id,
        "queue_id": 420,
        "game_version": "16.13.1",
        "game_duration": 1800,
        "winner": "blue",
        "blue_top": "Garen",
        "blue_jg": "LeeSin",
        "blue_mid": "Ahri",
        "blue_adc": "Jinx",
        "blue_sup": "Thresh",
        "red_top": "Darius",
        "red_jg": "Vi",
        "red_mid": "Syndra",
        "red_adc": "Kaisa",
        "red_sup": "Nautilus",
    }


def test_clean_and_split_removes_invalid_and_duplicate_rows(
    tmp_path,
    monkeypatch,
):
    # Make the test write into a temporary directory rather than
    # overwriting the project's real data files.
    monkeypatch.chdir(tmp_path)

    valid_row = make_valid_row("NA1_VALID")
    duplicate_row = valid_row.copy()

    invalid_row = make_valid_row("NA1_INVALID")
    invalid_row["blue_top"] = None

    input_df = pd.DataFrame(
        [
            valid_row,
            duplicate_row,
            invalid_row,
        ]
    )

    clean_df = cm.clean_and_split(
        input_df,
        required_queue=420,
    )

    assert len(clean_df) == 1
    assert clean_df.iloc[0]["match_id"] == "NA1_VALID"
    assert clean_df["match_id"].duplicated().sum() == 0
    assert clean_df.isna().sum().sum() == 0


def test_clean_and_split_writes_expected_csv_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    input_df = pd.DataFrame(
        [
            make_valid_row("NA1_TEST_1"),
            make_valid_row("NA1_TEST_2"),
        ]
    )

    cm.clean_and_split(
        input_df,
        required_queue=420,
    )

    data_directory = tmp_path / "data"

    assert (data_directory / "raw_matches.csv").exists()
    assert (data_directory / "clean_matches.csv").exists()
    assert (data_directory / "train.csv").exists()
    assert (data_directory / "test.csv").exists()

    raw_df = pd.read_csv(
        data_directory / "raw_matches.csv"
    )
    clean_df = pd.read_csv(
        data_directory / "clean_matches.csv"
    )
    train_df = pd.read_csv(
        data_directory / "train.csv"
    )
    test_df = pd.read_csv(
        data_directory / "test.csv"
    )

    assert len(raw_df) == 2
    assert len(clean_df) == 2

    # Every clean row should appear in exactly one split.
    assert len(train_df) + len(test_df) == len(clean_df)

    combined_ids = set(train_df["match_id"]) | set(
        test_df["match_id"]
    )

    assert combined_ids == {
        "NA1_TEST_1",
        "NA1_TEST_2",
    }