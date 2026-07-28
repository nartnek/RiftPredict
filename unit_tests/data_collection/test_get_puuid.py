from unittest.mock import Mock

import pytest
import requests

import src.data_collection.get_puuid as gp


def test_get_puuid_returns_puuid(monkeypatch):
    fake_response = Mock()
    fake_response.json.return_value = {
        "puuid": "test-puuid-123"
    }

    fake_get = Mock(return_value=fake_response)

    monkeypatch.setattr(
        gp,
        "get_api_key",
        lambda: "RGAPI-test-key",
    )
    monkeypatch.setattr(
        gp.requests,
        "get",
        fake_get,
    )

    result = gp.get_puuid(
        "Example Player",
        "NA1",
    )

    assert result == "test-puuid-123"
    fake_response.raise_for_status.assert_called_once_with()

    requested_url = fake_get.call_args.args[0]
    assert requested_url.endswith(
        "/by-riot-id/Example%20Player/NA1"
    )


def test_get_puuid_raises_for_http_error(monkeypatch):
    fake_response = Mock()
    fake_response.raise_for_status.side_effect = (
        requests.HTTPError("401 Unauthorized")
    )

    monkeypatch.setattr(
        gp,
        "get_api_key",
        lambda: "invalid-key",
    )
    monkeypatch.setattr(
        gp.requests,
        "get",
        lambda *args, **kwargs: fake_response,
    )

    with pytest.raises(
        requests.HTTPError,
        match="401 Unauthorized",
    ):
        gp.get_puuid("ExamplePlayer", "NA1")


def test_get_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv(
        "RIOT_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="Missing RIOT_API_KEY",
    ):
        gp.get_api_key()