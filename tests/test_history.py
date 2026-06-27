import pandas as pd

from src.utils.history import lock_first_snapshot, record_actual_result, update_predictions


def test_lock_first_snapshot_creates_file(tmp_path):
    df = pd.DataFrame([{"team": "Brazil", "champion_prob": 0.15}])
    lock_path = tmp_path / "pretournament_simulation.csv"

    locked = lock_first_snapshot(df, lock_path, generated_at=pd.Timestamp("2026-06-11"))

    assert locked is True
    saved = pd.read_csv(lock_path)
    assert saved.iloc[0]["generated_at"] == "2026-06-11"
    assert saved.iloc[0]["team"] == "Brazil"


def test_lock_first_snapshot_skips_if_already_exists(tmp_path):
    lock_path = tmp_path / "pretournament_simulation.csv"
    df1 = pd.DataFrame([{"team": "Brazil", "champion_prob": 0.15}])
    df2 = pd.DataFrame([{"team": "Brazil", "champion_prob": 0.20}])

    first = lock_first_snapshot(df1, lock_path, generated_at=pd.Timestamp("2026-06-11"))
    second = lock_first_snapshot(df2, lock_path, generated_at=pd.Timestamp("2026-06-12"))

    assert first is True
    assert second is False
    saved = pd.read_csv(lock_path)
    assert len(saved) == 1
    assert saved.iloc[0]["champion_prob"] == 0.15
    assert saved.iloc[0]["generated_at"] == "2026-06-11"


def test_lock_first_snapshot_overwrites_existing_generated_at_column(tmp_path):
    df = pd.DataFrame([{"generated_at": "stale", "team": "Brazil", "champion_prob": 0.15}])
    lock_path = tmp_path / "pretournament_simulation.csv"

    lock_first_snapshot(df, lock_path, generated_at=pd.Timestamp("2026-06-11"))

    saved = pd.read_csv(lock_path)
    assert list(saved.columns) == ["generated_at", "team", "champion_prob"]
    assert saved.iloc[0]["generated_at"] == "2026-06-11"


def test_update_predictions_creates_file_on_first_run(tmp_path):
    lock_path = tmp_path / "prematch_predictions.csv"
    df = pd.DataFrame([
        {"generated_at": "2026-06-11", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "1-0"},
    ])

    newly_seen = update_predictions(df, lock_path, key_cols=["date", "home_team", "away_team"])

    assert newly_seen == 1
    saved = pd.read_csv(lock_path, float_precision="round_trip")
    assert list(saved.columns) == [
        "predicted_at", "date", "home_team", "away_team", "predicted_score",
        "actual_home_score", "actual_away_score",
    ]
    assert saved.iloc[0]["predicted_at"] == "2026-06-11"
    assert pd.isna(saved.iloc[0]["actual_home_score"])


def test_update_predictions_refreshes_unplayed_fixture(tmp_path):
    lock_path = tmp_path / "prematch_predictions.csv"
    df1 = pd.DataFrame([
        {"generated_at": "2026-06-11", "date": "2026-06-12", "home_team": "Canada",
         "away_team": "Qatar", "predicted_score": "1-0"},
    ])
    update_predictions(df1, lock_path, key_cols=["date", "home_team", "away_team"])

    # Canada's first game has since been played, retraining shifts this prediction
    df2 = pd.DataFrame([
        {"generated_at": "2026-06-12", "date": "2026-06-12", "home_team": "Canada",
         "away_team": "Qatar", "predicted_score": "2-0"},
    ])
    newly_seen = update_predictions(df2, lock_path, key_cols=["date", "home_team", "away_team"])

    assert newly_seen == 0
    saved = pd.read_csv(lock_path, float_precision="round_trip")
    assert len(saved) == 1
    assert saved.iloc[0]["predicted_at"] == "2026-06-12"
    assert saved.iloc[0]["predicted_score"] == "2-0"


def test_update_predictions_freezes_played_fixture(tmp_path):
    lock_path = tmp_path / "prematch_predictions.csv"
    df1 = pd.DataFrame([
        {"generated_at": "2026-06-11", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "1-0"},
    ])
    update_predictions(df1, lock_path, key_cols=["date", "home_team", "away_team"])

    record_actual_result(
        lock_path, date="2026-06-11", home_team="Mexico", away_team="South Africa",
        actual_home_score=2, actual_away_score=0,
    )

    # A later run's retrained model would predict this differently, but the
    # match has already been played, so the row should stay frozen.
    df2 = pd.DataFrame([
        {"generated_at": "2026-06-12", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "2-1"},
    ])
    newly_seen = update_predictions(df2, lock_path, key_cols=["date", "home_team", "away_team"])

    assert newly_seen == 0
    saved = pd.read_csv(lock_path, float_precision="round_trip")
    assert len(saved) == 1
    assert saved.iloc[0]["predicted_at"] == "2026-06-11"
    assert saved.iloc[0]["predicted_score"] == "1-0"
    assert saved.iloc[0]["actual_home_score"] == 2
    assert saved.iloc[0]["actual_away_score"] == 0


def test_update_predictions_appends_newly_seen_fixture_and_keeps_existing(tmp_path):
    lock_path = tmp_path / "prematch_predictions.csv"
    df1 = pd.DataFrame([
        {"generated_at": "2026-06-11", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "1-0"},
    ])
    update_predictions(df1, lock_path, key_cols=["date", "home_team", "away_team"])

    df2 = pd.DataFrame([
        {"generated_at": "2026-06-20", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "2-0"},
        {"generated_at": "2026-06-20", "date": "2026-06-30", "home_team": "Brazil",
         "away_team": "Argentina", "predicted_score": "1-1"},
    ])
    newly_seen = update_predictions(df2, lock_path, key_cols=["date", "home_team", "away_team"])

    assert newly_seen == 1
    saved = pd.read_csv(lock_path, float_precision="round_trip")
    assert len(saved) == 2
    assert saved.iloc[0]["predicted_score"] == "2-0"
    assert saved.iloc[1]["predicted_at"] == "2026-06-20"
    assert saved.iloc[1]["home_team"] == "Brazil"


def test_update_predictions_backfills_stage_for_existing_rows_without_stage(tmp_path):
    lock_path = tmp_path / "prematch_predictions.csv"
    # Simulate a pre-migration CSV that has no stage column.
    old_df = pd.DataFrame([
        {"predicted_at": "2026-06-11", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "1-0",
         "actual_home_score": 2, "actual_away_score": 0},
    ])
    old_df = old_df.astype({"actual_home_score": "Int64", "actual_away_score": "Int64"})
    old_df.to_csv(lock_path, index=False)

    # New run with stage in snapshot; the frozen row should be backfilled.
    df_new = pd.DataFrame([
        {"generated_at": "2026-06-12", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "2-1", "stage": "group_stage"},
    ])
    update_predictions(df_new, lock_path, key_cols=["date", "home_team", "away_team"])

    saved = pd.read_csv(lock_path, float_precision="round_trip")
    assert "stage" in saved.columns
    assert saved.iloc[0]["stage"] == "group_stage"
    # The row is frozen (has actual scores), so original prediction is kept.
    assert saved.iloc[0]["predicted_score"] == "1-0"


def _locked_predictions(tmp_path):
    path = tmp_path / "prematch_predictions.csv"
    df = pd.DataFrame([
        {"predicted_at": "2026-06-11", "date": "2026-06-11", "home_team": "Mexico",
         "away_team": "South Africa", "predicted_score": "1-0",
         "actual_home_score": pd.NA, "actual_away_score": pd.NA},
    ])
    df = df.astype({"actual_home_score": "Int64", "actual_away_score": "Int64"})
    df.to_csv(path, index=False)
    return path


def test_record_actual_result_fills_in_score(tmp_path):
    lock_path = _locked_predictions(tmp_path)

    recorded = record_actual_result(
        lock_path, date="2026-06-11", home_team="Mexico", away_team="South Africa",
        actual_home_score=2, actual_away_score=0,
    )

    assert recorded is True
    saved = pd.read_csv(lock_path, float_precision="round_trip")
    assert saved.iloc[0]["actual_home_score"] == 2
    assert saved.iloc[0]["actual_away_score"] == 0


def test_record_actual_result_no_matching_fixture_is_noop(tmp_path):
    lock_path = _locked_predictions(tmp_path)

    recorded = record_actual_result(
        lock_path, date="2026-06-20", home_team="Brazil", away_team="Morocco",
        actual_home_score=1, actual_away_score=1,
    )

    assert recorded is False
    saved = pd.read_csv(lock_path, float_precision="round_trip")
    assert pd.isna(saved.iloc[0]["actual_home_score"])


def test_record_actual_result_missing_file_is_noop(tmp_path):
    lock_path = tmp_path / "prematch_predictions.csv"

    recorded = record_actual_result(
        lock_path, date="2026-06-11", home_team="Mexico", away_team="South Africa",
        actual_home_score=2, actual_away_score=0,
    )

    assert recorded is False
    assert not lock_path.exists()
