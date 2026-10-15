import pytest
from fastapi.testclient import TestClient

from race_dashboard.backend.api import create_app


@pytest.fixture
def client(replay):
    with TestClient(create_app(replay)) as c:
        yield c


def test_race_metadata(client):
    body = client.get("/race").json()
    assert body["event"] == "Test GP" and body["total_laps"] == 5
    assert [d["code"] for d in body["drivers"]] == ["A", "B", "C", "D"]
    assert len(body["outline"]) == 200
    assert body["best_lap"] == {"driver": "A", "time_s": 19.5}


def test_frames_window(client):
    body = client.get("/frames", params={"t0": 0, "t1": 10}).json()
    assert body["frames"] == 40
    assert set(body["drivers"]) == {"A", "B", "C", "D"}


@pytest.mark.parametrize("params", [{"t0": 10, "t1": 10}, {"t0": 20, "t1": 10}, {"t0": -1, "t1": 5}])
def test_frames_rejects_bad_ranges(client, params):
    assert client.get("/frames", params=params).status_code == 422


@pytest.mark.parametrize("params", [
    {"t0": "nan", "t1": 10}, {"t0": "inf", "t1": 10},
    {"t0": 0, "t1": "nan"}, {"t0": 0, "t1": "inf"},
])
def test_frames_rejects_non_finite(client, params):
    assert client.get("/frames", params=params).status_code == 422


@pytest.mark.parametrize("params", [{"t0": 1e308, "t1": 1.5e308}, {"t0": 0, "t1": 1e308}])
def test_frames_rejects_huge_finite(client, params):
    assert client.get("/frames", params=params).status_code == 422


def test_frames_missing_params_422(client):
    assert client.get("/frames").status_code == 422


def test_frames_window_length_is_capped(client, monkeypatch):
    monkeypatch.setattr("race_dashboard.backend.api.MAX_WINDOW_S", 20.0)
    assert client.get("/frames", params={"t0": 0, "t1": 100}).json()["frames"] == 80


def test_frames_past_end_is_empty_not_error(client):
    r = client.get("/frames", params={"t0": 500, "t1": 560})
    assert r.status_code == 200 and r.json()["frames"] == 0


def test_insights_served_when_present(replay):
    data = [{"lap": 1, "ranked_by": "jev", "insights": []}]
    with TestClient(create_app(replay, insights=data)) as c:
        assert c.get("/insights").json() == data


def test_insights_404_when_not_built(client):
    r = client.get("/insights")
    assert r.status_code == 404 and "insights" in r.json()["detail"]
