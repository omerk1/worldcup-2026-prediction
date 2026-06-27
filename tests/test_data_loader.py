import numpy as np
import pandas as pd
import pytest

from src.data_processing.data_loader import get_played_matches, infer_stage, latest_fifa_points


def _results_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Unplayed fixture - score should come from live_results_df.
        {"date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa",
         "home_score": np.nan, "away_score": np.nan, "tournament": "FIFA World Cup", "neutral": False},
        # Already-played match, no live result - included via the normal path.
        {"date": "2025-01-01", "home_team": "Brazil", "away_team": "Argentina",
         "home_score": 2.0, "away_score": 1.0, "tournament": "Friendly", "neutral": True},
        # Already-played match that ALSO has a (now redundant) live result recorded.
        {"date": "2026-06-12", "home_team": "France", "away_team": "Germany",
         "home_score": 3.0, "away_score": 0.0, "tournament": "FIFA World Cup", "neutral": False},
        # Outside the lookback window.
        {"date": "2010-01-01", "home_team": "OldTeam", "away_team": "OtherTeam",
         "home_score": 1.0, "away_score": 1.0, "tournament": "Friendly", "neutral": True},
    ]).assign(date=lambda d: pd.to_datetime(d["date"]))


def _live_results_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa",
         "home_score": 2.0, "away_score": 0.0},
        # Same match as the France/Germany row above, scored independently via record_result.py.
        {"date": "2026-06-12", "home_team": "France", "away_team": "Germany",
         "home_score": 3.0, "away_score": 0.0},
    ]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_live_result_fills_in_unplayed_fixture():
    as_of = pd.Timestamp("2026-06-15")
    played = get_played_matches(_results_df(), as_of, lookback_years=12, live_results_df=_live_results_df())

    mex = played[(played["home_team"] == "Mexico") & (played["away_team"] == "South Africa")]
    assert len(mex) == 1
    assert mex.iloc[0]["home_score"] == 2.0
    assert mex.iloc[0]["away_score"] == 0.0


def test_no_double_count_when_results_df_already_has_score():
    as_of = pd.Timestamp("2026-06-15")
    played = get_played_matches(_results_df(), as_of, lookback_years=12, live_results_df=_live_results_df())

    fra = played[(played["home_team"] == "France") & (played["away_team"] == "Germany")]
    assert len(fra) == 1


def test_lookback_window_and_total_count():
    as_of = pd.Timestamp("2026-06-15")
    played = get_played_matches(_results_df(), as_of, lookback_years=12, live_results_df=_live_results_df())

    assert len(played) == 3
    assert not ((played["home_team"] == "OldTeam").any())


def test_works_without_live_results_df():
    as_of = pd.Timestamp("2026-06-15")
    played = get_played_matches(_results_df(), as_of, lookback_years=12, live_results_df=None)

    # Unplayed fixture stays unplayed; old match still excluded by lookback.
    assert len(played) == 2
    assert set(played["home_team"]) == {"Brazil", "France"}


def _fifa_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2024-09-19", "team": "Strong", "total_points": 1900.0},
        {"date": "2024-09-19", "team": "Mid", "total_points": 1500.0},
        {"date": "2024-09-19", "team": "Weak", "total_points": 1000.0},
        {"date": "2024-09-19", "team": "Riser", "total_points": 1100.0},
    ]).assign(date=lambda d: pd.to_datetime(d["date"]))


def _knockout_fixtures_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp("2026-06-28"), "home_team": "South Africa", "away_team": "Canada",
         "stage": "round_of_32", "city": "Inglewood", "neutral": True},
    ])


def test_infer_stage_group_stage_by_date():
    knockouts = _knockout_fixtures_df()
    assert infer_stage(pd.Timestamp("2026-06-25"), "Japan", "Sweden", knockouts) == "group_stage"


def test_infer_stage_knockout_found():
    knockouts = _knockout_fixtures_df()
    stage = infer_stage(pd.Timestamp("2026-06-28"), "South Africa", "Canada", knockouts)
    assert stage == "round_of_32"


def test_infer_stage_knockout_not_found_raises():
    knockouts = _knockout_fixtures_df()
    with pytest.raises(ValueError, match="No knockout fixture found"):
        infer_stage(pd.Timestamp("2026-06-28"), "France", "Brazil", knockouts)


def test_latest_fifa_points_without_override():
    points = latest_fifa_points(_fifa_df())
    assert points["Riser"] == 1100.0


def test_latest_fifa_points_applies_current_rank_override():
    # "Riser" has since climbed to the #1 rank - it should now get the points
    # that the #1-ranked team ("Strong") had in the snapshot.
    current_rankings = pd.DataFrame([{"team": "Riser", "rank": 1}])
    points = latest_fifa_points(_fifa_df(), current_rankings)

    assert points["Riser"] == 1900.0
    # Untouched teams keep their snapshot points.
    assert points["Strong"] == 1900.0
    assert points["Mid"] == 1500.0


def test_latest_fifa_points_override_for_team_missing_from_snapshot():
    current_rankings = pd.DataFrame([{"team": "Newcomer", "rank": 2}])
    points = latest_fifa_points(_fifa_df(), current_rankings)

    assert points["Newcomer"] == 1500.0
