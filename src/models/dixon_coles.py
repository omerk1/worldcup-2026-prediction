"""Dixon-Coles Poisson model with time-decay weighting and a FIFA-ranking prior.

Each team i gets an attack rating (alpha_i) and defense rating (beta_i).
Higher beta_i means a STRONGER defense (it subtracts from opponents'
expected goals). Expected goals for a match home vs away:

    lambda_home = exp(mu + alpha_home - beta_away + gamma * host_advantage)
    lambda_away = exp(mu + alpha_away - beta_home)

Scores are Poisson(lambda) with the Dixon-Coles low-score correlation
correction (tau) for (0,0), (1,0), (0,1), (1,1).

Parameters are fit by maximizing a time-weighted log-likelihood plus an L2
penalty pulling each team's (alpha, beta) toward a prior derived from its
FIFA ranking points. This keeps data-rich teams driven by results while
sparse-data teams fall back to a sensible "tier" based on FIFA ranking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from src.data_processing.data_loader import FIFA_NAME_MAP


def time_weights(dates: pd.Series, as_of: pd.Timestamp, half_life_days: float) -> np.ndarray:
    days_ago = (as_of - dates).dt.days.values.astype(float)
    decay_rate = np.log(2) / half_life_days
    return np.exp(-decay_rate * days_ago)


def _tau_vectorized(x: np.ndarray, y: np.ndarray, lam_home: np.ndarray, lam_away: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam_home)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam_home[m00] * lam_away[m00] * rho
    tau[m01] = 1 + lam_home[m01] * rho
    tau[m10] = 1 + lam_away[m10] * rho
    tau[m11] = 1 - rho
    return tau


def _tau_scalar(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam_home * lam_away * rho
    if x == 0 and y == 1:
        return 1 + lam_home * rho
    if x == 1 and y == 0:
        return 1 + lam_away * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


@dataclass
class DixonColesModel:
    teams: list[str]
    mu: float
    rho: float
    gamma: float
    attack: np.ndarray
    defense: np.ndarray
    team_idx: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.team_idx = {t: i for i, t in enumerate(self.teams)}

    def ratings(self, team: str) -> tuple[float, float]:
        idx = self.team_idx.get(team)
        if idx is None:
            return 0.0, 0.0
        return float(self.attack[idx]), float(self.defense[idx])

    def expected_goals(self, home_team: str, away_team: str, host: bool = False) -> tuple[float, float]:
        a_h, d_h = self.ratings(home_team)
        a_a, d_a = self.ratings(away_team)
        lam_home = np.exp(self.mu + a_h - d_a + (self.gamma if host else 0.0))
        lam_away = np.exp(self.mu + a_a - d_h)
        return float(lam_home), float(lam_away)

    def score_matrix(self, home_team: str, away_team: str, host: bool = False, max_goals: int = 8) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(home_team, away_team, host)
        goals = np.arange(0, max_goals + 1)
        p_home = poisson.pmf(goals, lam_h)
        p_away = poisson.pmf(goals, lam_a)
        matrix = np.outer(p_home, p_away)
        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            matrix[x, y] *= _tau_scalar(x, y, lam_h, lam_a, self.rho)
        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()
        return matrix

    def predict(self, home_team: str, away_team: str, host: bool = False, max_goals: int = 8, top_n: int = 5) -> dict:
        matrix = self.score_matrix(home_team, away_team, host, max_goals)
        lam_h, lam_a = self.expected_goals(home_team, away_team, host)

        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())

        flat = [
            (matrix[h, a], h, a)
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
        ]
        flat.sort(reverse=True)
        top_scores = [
            {"home_goals": int(h), "away_goals": int(a), "probability": float(p)}
            for p, h, a in flat[:top_n]
        ]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "host": host,
            "expected_goals_home": lam_h,
            "expected_goals_away": lam_a,
            "home_win_prob": home_win,
            "draw_prob": draw,
            "away_win_prob": away_win,
            "most_likely_score": f"{top_scores[0]['home_goals']}-{top_scores[0]['away_goals']}",
            "top_scores": top_scores,
        }

    def to_dict(self) -> dict:
        return {
            "teams": self.teams,
            "mu": self.mu,
            "rho": self.rho,
            "gamma": self.gamma,
            "attack": self.attack.tolist(),
            "defense": self.defense.tolist(),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DixonColesModel":
        with open(path) as f:
            d = json.load(f)
        return cls(
            teams=d["teams"],
            mu=d["mu"],
            rho=d["rho"],
            gamma=d["gamma"],
            attack=np.array(d["attack"]),
            defense=np.array(d["defense"]),
        )


def _fifa_priors(teams: list[str], fifa_points: pd.Series, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Map FIFA ranking points to (attack, defense) priors.

    Given lambda_home = exp(mu + attack_home - defense_away + ...), a HIGH
    defense_i value reduces opponents' expected goals, i.e. high defense_i =
    a strong defense. Good teams (high FIFA points / z-score) should get
    both a high attack prior and a high defense prior.
    """
    mapped_names = [FIFA_NAME_MAP.get(t, t) for t in teams]
    points = np.array([fifa_points.get(name, np.nan) for name in mapped_names], dtype=float)
    valid = ~np.isnan(points)
    mean_p, std_p = points[valid].mean(), points[valid].std()
    if std_p == 0:
        z = np.zeros_like(points)
    else:
        z = np.where(valid, (points - mean_p) / std_p, 0.0)
    return scale * z, scale * z


def fit_dixon_coles(
    matches: pd.DataFrame,
    fifa_points: pd.Series,
    config: dict,
    as_of: pd.Timestamp,
) -> DixonColesModel:
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = matches["home_team"].map(team_idx).values
    away_idx = matches["away_team"].map(team_idx).values
    home_goals = matches["home_score"].values.astype(float)
    away_goals = matches["away_score"].values.astype(float)
    host_adv = (~matches["neutral"].astype(bool)).values.astype(float)

    weights = time_weights(matches["date"], as_of, config["decay_half_life_days"])

    prior_attack, prior_defense = _fifa_priors(teams, fifa_points, config["fifa_prior_scale"])
    prior_weight = config["fifa_prior_weight"]

    avg_goals = (home_goals.mean() + away_goals.mean()) / 2
    mu0 = float(np.log(avg_goals))
    x0 = np.concatenate([
        [mu0, config["rho_init"], 0.1],
        prior_attack.copy(),
        prior_defense.copy(),
    ])

    def unpack(params):
        mu, rho, gamma = params[0], params[1], params[2]
        attack = params[3:3 + n]
        defense = params[3 + n:3 + 2 * n]
        return mu, rho, gamma, attack, defense

    def neg_log_posterior(params):
        mu, rho, gamma, attack, defense = unpack(params)
        lam_home = np.exp(mu + attack[home_idx] - defense[away_idx] + gamma * host_adv)
        lam_away = np.exp(mu + attack[away_idx] - defense[home_idx])

        ll = (
            home_goals * np.log(lam_home) - lam_home - gammaln(home_goals + 1)
            + away_goals * np.log(lam_away) - lam_away - gammaln(away_goals + 1)
        )
        tau = _tau_vectorized(home_goals, away_goals, lam_home, lam_away, rho)
        tau = np.clip(tau, 1e-10, None)
        ll += np.log(tau)

        weighted_ll = np.sum(weights * ll)
        penalty = prior_weight * (
            np.sum((attack - prior_attack) ** 2) + np.sum((defense - prior_defense) ** 2)
        )
        return -weighted_ll + penalty

    bounds = (
        [(None, None), (-0.3, 0.3), (0.0, 1.0)]
        + [(None, None)] * n
        + [(None, None)] * n
    )

    result = minimize(neg_log_posterior, x0, method="L-BFGS-B", bounds=bounds)
    mu, rho, gamma, attack, defense = unpack(result.x)

    return DixonColesModel(teams=teams, mu=float(mu), rho=float(rho), gamma=float(gamma), attack=attack, defense=defense)
