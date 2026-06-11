"""Elo ratings computed by chronologically replaying historical results.

Uses the standard "World Football Elo" formulation: a goal-difference
multiplier scales each match's impact, and the K-factor is further scaled by
`tournament_weight` so World Cup results move ratings the most and friendlies
the least (using the same weights as the Dixon-Coles fit).
"""
from __future__ import annotations

import pandas as pd

from src.data_processing.data_loader import tournament_weight


def _goal_diff_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def compute_elo_ratings(
    results: pd.DataFrame,
    config: dict,
    as_of: pd.Timestamp,
) -> dict[str, float]:
    """Replay all played matches up to `as_of` and return final Elo ratings."""
    matches = results[
        results["home_score"].notna() & (results["date"] <= as_of)
    ].sort_values("date")

    elo_config = config["elo"]
    weights = config["tournament_weights"]
    initial_rating = elo_config["initial_rating"]
    base_k = elo_config["base_k_factor"]
    home_advantage = elo_config["home_advantage"]

    ratings: dict[str, float] = {}

    for row in matches.itertuples():
        r_home = ratings.setdefault(row.home_team, initial_rating)
        r_away = ratings.setdefault(row.away_team, initial_rating)

        rating_diff = r_home - r_away
        if not row.neutral:
            rating_diff += home_advantage

        expected_home = 1.0 / (10 ** (-rating_diff / 400.0) + 1.0)

        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score == row.away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        goal_diff = abs(row.home_score - row.away_score)
        k = base_k * tournament_weight(row.tournament, weights) * _goal_diff_multiplier(goal_diff)

        delta = k * (actual_home - expected_home)
        ratings[row.home_team] = r_home + delta
        ratings[row.away_team] = r_away - delta

    return ratings
