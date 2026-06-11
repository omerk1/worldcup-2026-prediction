import pandas as pd

from src.utils.history import append_history


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
