import numpy as np
import pandas as pd

from src.data_processing.data_loader import get_played_matches


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
