"""Estimate Golden Boot contenders from historical scoring shares.

For each player, we compute a recency-weighted share of their national
team's goals (own goals excluded, penalties included) over the lookback
window. Each team's expected total tournament goals is estimated as its
average expected goals per match (Dixon-Coles attack rating vs a
league-average defense, neutral venue) times its expected number of
matches (3 group-stage matches plus expected knockout matches from the
Monte Carlo simulation). A player's expected tournament goals is then
team_expected_goals * player_share.

Caveats: this is a "recent form" proxy, not a squad-list-aware model -
injuries, retirements, or squad omissions aren't accounted for. It also
assumes a player's share of team goals stays constant regardless of
opponent or match importance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel, time_weights

KNOCKOUT_STAGE_COLUMNS = ["round_of_32", "round_of_16", "quarterfinals", "semifinals", "final"]


def player_goal_shares(
    goalscorers: pd.DataFrame, teams: set[str], as_of: pd.Timestamp, half_life_days: float
) -> pd.DataFrame:
    """Recency-weighted share of each team's goals scored by each player."""
    df = goalscorers[
        ~goalscorers["own_goal"]
        & goalscorers["team"].isin(teams)
        & goalscorers["scorer"].notna()
        & (goalscorers["date"] <= as_of)
    ].copy()
    df["weight"] = time_weights(df["date"], as_of, half_life_days)

    player_w = df.groupby(["team", "scorer"])["weight"].sum().reset_index(name="player_weight")
    team_w = df.groupby("team")["weight"].sum().reset_index(name="team_weight")
    merged = player_w.merge(team_w, on="team")
    merged["share"] = merged["player_weight"] / merged["team_weight"]
    return merged


def expected_team_goals(model: DixonColesModel, team: str, expected_matches: float) -> float:
    """Expected goals/match for `team` against a league-average defense (neutral), times expected matches."""
    attack, _ = model.ratings(team)
    goals_per_match = np.exp(model.mu + attack)
    return goals_per_match * expected_matches


def predict_top_scorers(
    goalscorers: pd.DataFrame,
    model: DixonColesModel,
    simulation: pd.DataFrame,
    teams: set[str],
    as_of: pd.Timestamp,
    half_life_days: float,
    group_stage_matches: int = 3,
) -> pd.DataFrame:
    shares = player_goal_shares(goalscorers, teams, as_of, half_life_days)

    sim = simulation.set_index("team")
    expected_knockout_matches = sim[KNOCKOUT_STAGE_COLUMNS].sum(axis=1)
    expected_matches = group_stage_matches + expected_knockout_matches

    rows = []
    for row in shares.itertuples(index=False):
        if row.team not in expected_matches.index:
            continue
        team_goals = expected_team_goals(model, row.team, expected_matches.loc[row.team])
        rows.append(
            {
                "player": row.scorer,
                "team": row.team,
                "goal_share": row.share,
                "expected_matches": expected_matches.loc[row.team],
                "expected_goals": team_goals * row.share,
            }
        )

    return pd.DataFrame(rows).sort_values("expected_goals", ascending=False).reset_index(drop=True)
