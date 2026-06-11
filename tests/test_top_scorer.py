import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel
from src.models.top_scorer import expected_team_goals, player_goal_shares, predict_top_scorers


def _make_goalscorers() -> pd.DataFrame:
    rows = [
        {"date": "2025-06-01", "team": "Atlantis", "scorer": "A. Striker", "own_goal": False, "penalty": False},
        {"date": "2025-06-01", "team": "Atlantis", "scorer": "A. Striker", "own_goal": False, "penalty": False},
        {"date": "2025-06-01", "team": "Atlantis", "scorer": "B. Winger", "own_goal": False, "penalty": False},
        {"date": "2025-06-01", "team": "Atlantis", "scorer": "C. Defender", "own_goal": True, "penalty": False},
        {"date": "2018-06-01", "team": "Atlantis", "scorer": "Old Legend", "own_goal": False, "penalty": False},
    ]
    return pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d["date"]))


def _make_model() -> DixonColesModel:
    return DixonColesModel(
        teams=["Atlantis"], mu=0.1, rho=-0.05, gamma=0.25,
        attack=np.array([0.3]), defense=np.array([0.0]),
    )


def test_player_goal_shares_excludes_own_goals_and_decays_old_goals():
    as_of = pd.Timestamp("2026-06-11")
    shares = player_goal_shares(_make_goalscorers(), {"Atlantis"}, as_of, half_life_days=730)

    assert "C. Defender" not in shares["scorer"].values

    striker_share = shares.loc[shares["scorer"] == "A. Striker", "share"].iloc[0]
    legend_share = shares.loc[shares["scorer"] == "Old Legend", "share"].iloc[0]
    assert striker_share > legend_share
    assert np.isclose(shares["share"].sum(), 1.0)


def test_predict_top_scorers_ranks_by_expected_goals():
    as_of = pd.Timestamp("2026-06-11")
    model = _make_model()
    simulation = pd.DataFrame([{
        "team": "Atlantis", "round_of_32": 1.0, "round_of_16": 0.5,
        "quarterfinals": 0.25, "semifinals": 0.1, "final": 0.05, "champion": 0.02,
    }])

    df = predict_top_scorers(_make_goalscorers(), model, simulation, {"Atlantis"}, as_of, half_life_days=730)

    assert list(df["player"])[0] == "A. Striker"
    expected_total = expected_team_goals(model, "Atlantis", expected_matches=3 + 1.0 + 0.5 + 0.25 + 0.1 + 0.05)
    assert np.isclose(df["expected_goals"].sum(), expected_total)
