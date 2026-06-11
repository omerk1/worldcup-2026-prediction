import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel
from src.simulation.tournament import derive_groups, run_simulations


def _make_fixtures(n_groups: int = 12) -> pd.DataFrame:
    rows = []
    date = pd.Timestamp("2026-06-11")
    for g in range(n_groups):
        teams = [f"G{g}T{i}" for i in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append({
                    "date": date,
                    "home_team": teams[i],
                    "away_team": teams[j],
                    "neutral": True,
                })
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _make_model(teams: list[str]) -> DixonColesModel:
    return DixonColesModel(
        teams=teams,
        mu=0.3,
        rho=-0.05,
        gamma=0.25,
        attack=np.zeros(len(teams)),
        defense=np.zeros(len(teams)),
    )


def test_derive_groups_partitions_all_teams():
    fixtures = _make_fixtures()
    groups = derive_groups(fixtures)
    assert len(groups) == 12
    all_teams = [t for teams in groups.values() for t in teams]
    assert len(all_teams) == 48
    assert len(set(all_teams)) == 48
    assert all(len(teams) == 4 for teams in groups.values())


def test_run_simulations_probabilities_are_consistent():
    fixtures = _make_fixtures()
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    model = _make_model(teams)

    df, _ = run_simulations(model, fixtures, host_nations=set(), n_simulations=20, seed=0)

    assert len(df) == 48
    assert np.isclose(df["round_of_32"].sum(), 32.0)
    assert np.isclose(df["champion"].sum(), 1.0)
    assert (df["round_of_32"] >= df["round_of_16"]).all()
    assert (df["round_of_16"] >= df["quarterfinals"]).all()
    assert (df["quarterfinals"] >= df["semifinals"]).all()
    assert (df["semifinals"] >= df["final"]).all()
    assert (df["final"] >= df["champion"]).all()
