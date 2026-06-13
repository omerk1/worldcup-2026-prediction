from pathlib import Path

import pandas as pd

from src.utils.history import append_history, lock_prematch


def test_append_history_creates_file_with_generated_at(tmp_path):
    df = pd.DataFrame([{"team": "Brazil", "champion": 0.15}])
    path = tmp_path / "history.csv"

    append_history(df, path, key_cols=["team"], generated_at=pd.Timestamp("2026-06-11"))

    saved = pd.read_csv(path)
    assert saved.iloc[0]["generated_at"] == "2026-06-11"
    assert saved.iloc[0]["team"] == "Brazil"


def test_append_history_accumulates_across_dates(tmp_path):
    path = tmp_path / "history.csv"
    df1 = pd.DataFrame([{"team": "Brazil", "champion": 0.15}])
    df2 = pd.DataFrame([{"team": "Brazil", "champion": 0.16}])

    append_history(df1, path, key_cols=["team"], generated_at=pd.Timestamp("2026-06-11"))
    append_history(df2, path, key_cols=["team"], generated_at=pd.Timestamp("2026-06-12"))

    saved = pd.read_csv(path)
    assert len(saved) == 2
    assert list(saved["generated_at"]) == ["2026-06-11", "2026-06-12"]


def test_append_history_with_existing_generated_at_column(tmp_path):
    df = pd.DataFrame([{"generated_at": "stale", "team": "Brazil", "champion": 0.15}])
    path = tmp_path / "history.csv"

    append_history(df, path, key_cols=["team"], generated_at=pd.Timestamp("2026-06-11"))

    saved = pd.read_csv(path)
    assert list(saved.columns) == ["generated_at", "team", "champion"]
    assert saved.iloc[0]["generated_at"] == "2026-06-11"


def test_append_history_replaces_same_day_rerun(tmp_path):
    path = tmp_path / "history.csv"
    df1 = pd.DataFrame([{"team": "Brazil", "champion": 0.15}])
    df2 = pd.DataFrame([{"team": "Brazil", "champion": 0.20}])

    append_history(df1, path, key_cols=["team"], generated_at=pd.Timestamp("2026-06-11"))
    append_history(df2, path, key_cols=["team"], generated_at=pd.Timestamp("2026-06-11"))

    saved = pd.read_csv(path)
    assert len(saved) == 1
    assert saved.iloc[0]["champion"] == 0.20


def _predictions_history(tmp_path) -> Path:
    path = tmp_path / "predictions_history.csv"
    pd.DataFrame([
        {"generated_at": "2026-06-11", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "1-0"},
        {"generated_at": "2026-06-12", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "1-0"},
    ]).to_csv(path, index=False)
    return path


def test_lock_prematch_locks_earliest_snapshot(tmp_path):
    history_path = _predictions_history(tmp_path)
    lock_path = tmp_path / "prematch_predictions.csv"

    locked = lock_prematch(
        history_path, lock_path, date="2026-06-11", home_team="Mexico", away_team="South Africa",
        actual_home_score=2, actual_away_score=0,
    )

    assert locked is True
    saved = pd.read_csv(lock_path)
    assert len(saved) == 1
    assert saved.iloc[0]["predicted_at"] == "2026-06-11"
    assert saved.iloc[0]["predicted_score"] == "1-0"
    assert saved.iloc[0]["actual_home_score"] == 2
    assert saved.iloc[0]["actual_away_score"] == 0


def test_lock_prematch_is_idempotent(tmp_path):
    history_path = _predictions_history(tmp_path)
    lock_path = tmp_path / "prematch_predictions.csv"

    first = lock_prematch(
        history_path, lock_path, date="2026-06-11", home_team="Mexico", away_team="South Africa",
        actual_home_score=2, actual_away_score=0,
    )
    second = lock_prematch(
        history_path, lock_path, date="2026-06-11", home_team="Mexico", away_team="South Africa",
        actual_home_score=2, actual_away_score=0,
    )

    assert first is True
    assert second is False
    assert len(pd.read_csv(lock_path)) == 1


def test_lock_prematch_no_matching_prediction_is_noop(tmp_path):
    history_path = _predictions_history(tmp_path)
    lock_path = tmp_path / "prematch_predictions.csv"

    locked = lock_prematch(
        history_path, lock_path, date="2026-06-20", home_team="Brazil", away_team="Morocco",
        actual_home_score=1, actual_away_score=1,
    )

    assert locked is False
    assert not lock_path.exists()
