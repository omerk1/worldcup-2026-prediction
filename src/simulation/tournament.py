"""Monte Carlo simulation of the 2026 World Cup (group stage + knockout bracket).

Groups are derived from the 72 group-stage fixtures (each group is a set of 4
teams that all play each other). The knockout bracket is a simplified 32-team
single-elimination seeding: the 32 qualifiers (12 group winners, 12 runners-up,
8 best third-placed teams) are ranked by overall model rating (attack +
defense) and placed into a standard tournament seeding order, so the
strongest teams are spread across different bracket halves/quarters (seed 1
and seed 2 can only meet in the final, etc.) rather than colliding early just
because their groups happen to be alphabetically adjacent. A swap step avoids
two teams from the same group meeting in the round of 32. This is *not*
FIFA's official bracket-assignment table (which depends on exactly which
groups' third-placed teams qualify), just a reasonable approximation for
simulation purposes.

Knockout matches are treated as neutral-venue (no host advantage), and drawn
matches are decided by a 50/50 coin flip (proxy for extra time + penalties).
"""
from __future__ import annotations

import string
from collections import defaultdict

import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel

STAGE_LABELS = ["round_of_32", "round_of_16", "quarterfinals", "semifinals", "final", "champion"]


def derive_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Group the 48 teams into 12 groups of 4 based on who plays whom."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in fixtures.itertuples():
        union(row.home_team, row.away_team)

    members: dict[str, list[str]] = defaultdict(list)
    for team in parent:
        members[find(team)].append(team)

    # Order groups by each group's earliest fixture date, label A, B, C, ...
    group_order: list[str] = []
    seen: set[str] = set()
    for row in fixtures.sort_values("date").itertuples():
        root = find(row.home_team)
        if root not in seen:
            seen.add(root)
            group_order.append(root)

    labels = string.ascii_uppercase
    return {labels[i]: sorted(members[root]) for i, root in enumerate(group_order)}


class ScoreMatrixCache:
    """Caches Dixon-Coles score-matrix CDFs so repeated matchups across many
    simulations don't re-run the (cheap but non-trivial) Poisson math."""

    def __init__(self, model: DixonColesModel, max_goals: int = 8):
        self.model = model
        self.max_goals = max_goals
        self._cdf: dict[tuple[str, str, bool], np.ndarray] = {}

    def _cdf_for(self, home: str, away: str, host: bool) -> np.ndarray:
        key = (home, away, host)
        cdf = self._cdf.get(key)
        if cdf is None:
            matrix = self.model.score_matrix(home, away, host=host, max_goals=self.max_goals)
            cdf = np.cumsum(matrix.flatten())
            self._cdf[key] = cdf
        return cdf

    def sample(self, home: str, away: str, host: bool, rng: np.random.Generator) -> tuple[int, int]:
        cdf = self._cdf_for(home, away, host)
        idx = int(np.searchsorted(cdf, rng.random(), side="right"))
        idx = min(idx, len(cdf) - 1)
        return divmod(idx, self.max_goals + 1)


def _new_standings(teams: list[str]) -> dict[str, dict[str, int]]:
    return {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}


def _apply_result(standings: dict[str, dict[str, int]], home: str, away: str, hg: int, ag: int) -> None:
    standings[home]["gf"] += hg
    standings[home]["ga"] += ag
    standings[away]["gf"] += ag
    standings[away]["ga"] += hg
    standings[home]["gd"] = standings[home]["gf"] - standings[home]["ga"]
    standings[away]["gd"] = standings[away]["gf"] - standings[away]["ga"]
    if hg > ag:
        standings[home]["pts"] += 3
    elif hg < ag:
        standings[away]["pts"] += 3
    else:
        standings[home]["pts"] += 1
        standings[away]["pts"] += 1


def _rank_group(standings: dict[str, dict[str, int]], rng: np.random.Generator) -> list[str]:
    teams = list(standings.keys())
    tiebreak = {t: rng.random() for t in teams}
    teams.sort(key=lambda t: (-standings[t]["pts"], -standings[t]["gd"], -standings[t]["gf"], tiebreak[t]))
    return teams


def _seed_order(n: int) -> list[int]:
    """Standard single-elimination seeding order (1-indexed) for n slots.

    Ensures seed 1 and seed 2 can only meet in the final, seeds 1-4 can only
    meet from the semifinals onward, etc.
    """
    order = [1]
    size = 1
    while size < n:
        order = [s for seed in order for s in (seed, 2 * size + 1 - seed)]
        size *= 2
    return order


def _build_r32_bracket(
    group_rankings: dict[str, list[str]],
    team_to_group: dict[str, str],
    qualifying_thirds: list[str],
    team_strength: dict[str, float],
) -> list[tuple[str, str]]:
    winners = [ranked[0] for ranked in group_rankings.values()]
    runners_up = [ranked[1] for ranked in group_rankings.values()]
    qualifiers = winners + runners_up + qualifying_thirds  # 12 + 12 + 8 = 32

    # Rank the 32 qualifiers by overall model rating: seed 1 = strongest.
    ranked_by_strength = sorted(qualifiers, key=lambda t: team_strength.get(t, 0.0), reverse=True)
    slots = [ranked_by_strength[s - 1] for s in _seed_order(32)]
    pairs = [(slots[2 * k], slots[2 * k + 1]) for k in range(16)]

    def same_group(a: str, b: str) -> bool:
        return team_to_group[a] == team_to_group[b]

    for i, (a, b) in enumerate(pairs):
        if not same_group(a, b):
            continue
        for j in range(i + 1, len(pairs)):
            c, d = pairs[j]
            if not same_group(a, d) and not same_group(c, b):
                pairs[i] = (a, d)
                pairs[j] = (c, b)
                break

    return pairs


def _simulate_knockout(
    pairs: list[tuple[str, str]], cache: ScoreMatrixCache, rng: np.random.Generator
) -> dict[str, str]:
    reached: dict[str, str] = {}
    for home, away in pairs:
        reached[home] = "round_of_32"
        reached[away] = "round_of_32"

    current = pairs
    for stage in ["round_of_16", "quarterfinals", "semifinals", "final", "champion"]:
        winners = []
        for home, away in current:
            hg, ag = cache.sample(home, away, host=False, rng=rng)
            if hg == ag:
                winner = home if rng.random() < 0.5 else away
            else:
                winner = home if hg > ag else away
            winners.append(winner)

        if stage == "champion":
            reached[winners[0]] = "champion"
            break

        for w in winners:
            reached[w] = stage
        current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    return reached


def simulate_once(
    fixtures: pd.DataFrame,
    groups: dict[str, list[str]],
    host_nations: set[str],
    team_strength: dict[str, float],
    cache: ScoreMatrixCache,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Run a single simulated tournament and return team -> furthest stage reached."""
    standings = {g: _new_standings(teams) for g, teams in groups.items()}
    team_to_group = {t: g for g, teams in groups.items() for t in teams}

    for row in fixtures.itertuples():
        host = (not row.neutral) and (row.home_team in host_nations)
        hg, ag = cache.sample(row.home_team, row.away_team, host, rng)
        _apply_result(standings[team_to_group[row.home_team]], row.home_team, row.away_team, hg, ag)

    rankings = {g: _rank_group(standings[g], rng) for g in groups}

    thirds = [(g, rankings[g][2]) for g in rankings]
    thirds.sort(
        key=lambda gt: (
            -standings[gt[0]][gt[1]]["pts"],
            -standings[gt[0]][gt[1]]["gd"],
            -standings[gt[0]][gt[1]]["gf"],
            rng.random(),
        )
    )
    qualifying_thirds = [team for _, team in thirds[:8]]

    bracket = _build_r32_bracket(rankings, team_to_group, qualifying_thirds, team_strength)
    return _simulate_knockout(bracket, cache, rng)


def run_simulations(
    model: DixonColesModel,
    fixtures: pd.DataFrame,
    host_nations: set[str],
    n_simulations: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    groups = derive_groups(fixtures)
    cache = ScoreMatrixCache(model)

    all_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    team_strength = {t: sum(model.ratings(t)) for t in all_teams}
    counts = {t: {label: 0 for label in STAGE_LABELS} for t in all_teams}

    for _ in range(n_simulations):
        reached = simulate_once(fixtures, groups, host_nations, team_strength, cache, rng)
        for team, stage in reached.items():
            stage_idx = STAGE_LABELS.index(stage)
            for label in STAGE_LABELS[: stage_idx + 1]:
                counts[team][label] += 1

    rows = []
    for team in all_teams:
        row = {"team": team}
        for label in STAGE_LABELS:
            row[label] = counts[team][label] / n_simulations
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
    return df, groups
