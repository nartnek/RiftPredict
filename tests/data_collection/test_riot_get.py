import pytest

import src.data_collection.collect_matches as cm


class FakeResponse:
    def __init__(
        self,
        status_code,
        json_data=None,
        headers=None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        raise RuntimeError(
            f"HTTP status {self.status_code}"
        )


def test_riot_get_retries_after_rate_limit(
    monkeypatch,
):
    responses = [
        FakeResponse(
            status_code=429,
            headers={"Retry-After": "2"},
        ),
        FakeResponse(
            status_code=200,
            json_data={"success": True},
        ),
    ]

    sleep_calls = []

    monkeypatch.setattr(
        cm,
        "get_api_key",
        lambda: "fake-test-key",
    )

    monkeypatch.setattr(
        cm.requests,
        "get",
        lambda *args, **kwargs: responses.pop(0),
    )

    monkeypatch.setattr(
        cm.time,
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    result = cm.riot_get(
        "https://example.test"
    )

    assert result == {"success": True}
    assert sleep_calls == [2]


@pytest.mark.parametrize(
    "status_code",
    [401, 403],
)
def test_riot_get_rejects_invalid_api_key(
    monkeypatch,
    status_code,
):
    monkeypatch.setattr(
        cm,
        "get_api_key",
        lambda: "invalid-key",
    )

    monkeypatch.setattr(
        cm.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            status_code=status_code
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="invalid, or expired",
    ):
        cm.riot_get("https://example.test")