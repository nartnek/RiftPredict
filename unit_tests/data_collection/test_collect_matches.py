# unit_tests/test_collect_matches.py

import pandas as pd
import pytest

import src.data_collection.collect_matches as cm


POSITIONS = [
    "TOP",
    "JUNGLE",
    "MIDDLE",
    "BOTTOM",
    "UTILITY",
]

BLUE_CHAMPIONS = [
    "Garen",
    "LeeSin",
    "Ahri",
    "Jinx",
    "Thresh",
]

RED_CHAMPIONS = [
    "Darius",
    "Vi",
    "Syndra",
    "Kaisa",
    "Nautilus",
]


def make_match(
    match_id="NA1_TEST_1",
    blue_win=True,
    participant_puuids=None,
):
    """Creates fake Riot match JSON for unit tests."""
    if participant_puuids is None:
        participant_puuids = [
            f"puuid_{index}"
            for index in range(10)
        ]

    participants = []

    for index, position in enumerate(POSITIONS):
        participants.append(
            {
                "teamId": 100,
                "teamPosition": position,
                "championName": BLUE_CHAMPIONS[index],
                "puuid": participant_puuids[index],
            }
        )

    for index, position in enumerate(POSITIONS):
        participants.append(
            {
                "teamId": 200,
                "teamPosition": position,
                "championName": RED_CHAMPIONS[index],
                "puuid": participant_puuids[index + 5],
            }
        )

    return {
        "metadata": {
            "matchId": match_id,
        },
        "info": {
            "queueId": 420,
            "gameVersion": "16.13.1",
            "gameDuration": 1800,
            "teams": [
                {
                    "teamId": 100,
                    "win": blue_win,
                },
                {
                    "teamId": 200,
                    "win": not blue_win,
                },
            ],
            "participants": participants,
        },
    }


def make_valid_row(match_id="NA1_TEST_1"):
    return cm.extract_match_row(
        make_match(match_id=match_id)
    )


def test_extract_match_row_assigns_roles_and_winner():
    row = cm.extract_match_row(make_match())

    assert row["match_id"] == "NA1_TEST_1"
    assert row["queue_id"] == 420
    assert row["winner"] == "blue"

    assert row["blue_top"] == "Garen"
    assert row["blue_jg"] == "LeeSin"
    assert row["blue_mid"] == "Ahri"
    assert row["blue_adc"] == "Jinx"
    assert row["blue_sup"] == "Thresh"

    assert row["red_top"] == "Darius"
    assert row["red_jg"] == "Vi"
    assert row["red_mid"] == "Syndra"
    assert row["red_adc"] == "Kaisa"
    assert row["red_sup"] == "Nautilus"


def test_extract_match_row_detects_red_winner():
    row = cm.extract_match_row(
        make_match(blue_win=False)
    )

    assert row["winner"] == "red"


def test_extract_participant_puuids():
    match = make_match()

    puuids = cm.extract_participant_puuids(match)

    assert len(puuids) == 10
    assert puuids[0] == "puuid_0"
    assert puuids[-1] == "puuid_9"


def test_valid_match_row_is_accepted():
    row = make_valid_row()

    assert cm.is_valid_match_row(
        row,
        required_queue=420,
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("queue_id", 440),
        ("winner", "unknown"),
        ("blue_top", None),
        ("blue_jg", float("nan")),
        ("blue_mid", ""),
        ("red_sup", "   "),
    ],
)
def test_invalid_match_rows_are_rejected(
    field,
    invalid_value,
):
    row = make_valid_row()
    row[field] = invalid_value

    assert not cm.is_valid_match_row(
        row,
        required_queue=420,
    )


def test_checkpoint_round_trip_removes_duplicates(
    tmp_path,
):
    checkpoint = tmp_path / "raw_matches.csv"

    rows = [
        make_valid_row("NA1_1"),
        make_valid_row("NA1_1"),
        make_valid_row("NA1_2"),
    ]

    cm.save_checkpoint(
        rows,
        path=str(checkpoint),
    )

    loaded_rows = cm.load_checkpoint(
        path=str(checkpoint),
    )

    loaded_ids = {
        row["match_id"]
        for row in loaded_rows
    }

    assert loaded_ids == {"NA1_1", "NA1_2"}
    assert len(loaded_rows) == 2


def test_load_checkpoint_returns_empty_when_missing(
    tmp_path,
):
    checkpoint = tmp_path / "missing.csv"

    assert cm.load_checkpoint(
        path=str(checkpoint)
    ) == []


def test_collect_matches_discovers_another_player(
    monkeypatch,
):
    match_one = make_match(
        match_id="NA1_1",
        participant_puuids=[
            "second_player",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "p7",
            "p8",
            "p9",
            "p10",
        ],
    )

    match_two = make_match(
        match_id="NA1_2",
        participant_puuids=[
            f"other_{index}"
            for index in range(10)
        ],
    )

    def fake_get_match_ids(
        puuid,
        count,
        queue,
    ):
        if puuid == "starting_player":
            return ["NA1_1"]

        if puuid == "second_player":
            return ["NA1_2"]

        return []

    matches = {
        "NA1_1": match_one,
        "NA1_2": match_two,
    }

    monkeypatch.setattr(
        cm,
        "load_checkpoint",
        lambda: [],
    )

    monkeypatch.setattr(
        cm,
        "save_checkpoint",
        lambda rows: None,
    )

    monkeypatch.setattr(
        cm,
        "get_match_ids",
        fake_get_match_ids,
    )

    monkeypatch.setattr(
        cm,
        "get_match_data",
        lambda match_id: matches[match_id],
    )

    monkeypatch.setattr(
        cm.time,
        "sleep",
        lambda seconds: None,
    )

    result = cm.collect_matches(
        puuids=["starting_player"],
        target=2,
        matches_per_player=5,
        queue=420,
        checkpoint_every=1,
    )

    assert len(result) == 2
    assert set(result["match_id"]) == {
        "NA1_1",
        "NA1_2",
    }


def test_collect_matches_stops_if_checkpoint_reaches_target(
    monkeypatch,
):
    checkpoint_rows = [
        make_valid_row("NA1_1"),
        make_valid_row("NA1_2"),
    ]

    monkeypatch.setattr(
        cm,
        "load_checkpoint",
        lambda: checkpoint_rows,
    )

    def unexpected_api_call(*args, **kwargs):
        pytest.fail(
            "The Riot API should not be called "
            "when the target is already reached."
        )

    monkeypatch.setattr(
        cm,
        "get_match_ids",
        unexpected_api_call,
    )

    result = cm.collect_matches(
        puuids=["starting_player"],
        target=2,
        queue=420,
    )

    assert len(result) == 2