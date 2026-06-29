"""Dixon-Coles Poisson model with time-decay weighting and a strength prior.

Each team i gets an attack rating (alpha_i) and defense rating (beta_i).
Higher beta_i means a STRONGER defense (it subtracts from opponents'
expected goals). Expected goals for a match home vs away:

    lambda_home = exp(mu + alpha_home - beta_away + gamma * host_advantage)
    lambda_away = exp(mu + alpha_away - beta_home)

Scores are Poisson(lambda) with the Dixon-Coles low-score correlation
correction (tau) for (0,0), (1,0), (0,1), (1,1).

Parameters are fit by maximizing a log-likelihood over historical matches,
weighted by both recency (time-decay) and tournament importance (World Cup >
continental > qualifiers > friendlies), plus an L2 penalty pulling each
team's (alpha, beta) toward a strength prior blended from FIFA ranking points
and self-computed Elo ratings. This keeps data-rich teams driven by results
while sparse-data teams fall back to a sensible "tier" based on global
standing. Matches where the weaker side is below `min_fifa_points_threshold`
are excluded from fitting to avoid "blowout vs minnow" distortion.
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

from src.data_processing.data_loader import FIFA_NAME_MAP, tournament_weight


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

    def score_matrix_120(
        self,
        home_team: str,
        away_team: str,
        host: bool = False,
        max_goals: int = 8,
    ) -> np.ndarray:
        """120-minute score matrix for knockout games.

        Decisive 90-min outcomes carry over unchanged — the game ends at 90,
        no ET is played. Each draw outcome (k-k at 90) is spread through the
        ET goal distribution: a 1-1 draw can become 1-1 at 120 (ET 0-0 →
        pens), 2-1 at 120 (ET 1-0), 2-2 at 120 (ET 1-1 → pens), etc.
        """
        m90 = self.score_matrix(home_team, away_team, host=host, max_goals=max_goals)
        lam_h, lam_a = self.expected_goals(home_team, away_team, host=False)
        n = max_goals + 1
        goals = np.arange(n)
        m_et = np.outer(poisson.pmf(goals, lam_h / 3), poisson.pmf(goals, lam_a / 3))
        m_et /= m_et.sum()

        m120 = m90 * (1 - np.eye(n))  # copy decisive cells; draw cells start at 0
        for k, p in enumerate(np.diag(m90)):
            m120[k:, k:] += p * m_et[:n - k, :n - k]
        m120 /= m120.sum()
        return m120

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


def _zscore(values: np.ndarray) -> np.ndarray:
    """Z-score `values`, leaving NaNs (missing data) as NaN."""
    valid = ~np.isnan(values)
    if not valid.any():
        return np.full_like(values, np.nan)
    mean_v, std_v = values[valid].mean(), values[valid].std()
    z = np.full_like(values, np.nan)
    if std_v == 0:
        z[valid] = 0.0
    else:
        z[valid] = (values[valid] - mean_v) / std_v
    return z


def _strength_priors(
    teams: list[str],
    fifa_points: pd.Series,
    elo_ratings: dict[str, float],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map FIFA ranking points + Elo ratings to (attack, defense) priors.

    Both signals are z-scored across `teams` and averaged. A team missing one
    signal falls back to the other; a team missing both gets z=0 (average).
    Given lambda_home = exp(mu + attack_home - defense_away + ...), a HIGH
    defense_i value reduces opponents' expected goals, i.e. high defense_i =
    a strong defense. Good teams (high combined z-score) should get both a
    high attack prior and a high defense prior.
    """
    mapped_names = [FIFA_NAME_MAP.get(t, t) for t in teams]
    fifa_pts = np.array([fifa_points.get(name, np.nan) for name in mapped_names], dtype=float)
    elo_pts = np.array([elo_ratings.get(t, np.nan) for t in teams], dtype=float)

    fifa_z = _zscore(fifa_pts)
    elo_z = _zscore(elo_pts)

    stacked = np.stack([fifa_z, elo_z])
    valid = ~np.isnan(stacked)
    counts = np.clip(valid.sum(axis=0), 1, None)
    combined = np.where(valid, stacked, 0.0).sum(axis=0) / counts

    return scale * combined, scale * combined


def _filter_minnow_matches(matches: pd.DataFrame, fifa_points: pd.Series, threshold: float) -> pd.DataFrame:
    """Drop matches where the weaker side's FIFA points are below `threshold`.

    Missing FIFA points are treated as below threshold, so matches involving
    teams we have no ranking data for are also excluded.
    """
    def team_points(team: str) -> float:
        return fifa_points.get(FIFA_NAME_MAP.get(team, team), np.nan)

    home_points = matches["home_team"].map(team_points).fillna(-np.inf)
    away_points = matches["away_team"].map(team_points).fillna(-np.inf)
    weaker_points = np.minimum(home_points, away_points)
    return matches[weaker_points >= threshold]


def fit_dixon_coles(
    matches: pd.DataFrame,
    fifa_points: pd.Series,
    elo_ratings: dict[str, float],
    config: dict,
    as_of: pd.Timestamp,
) -> DixonColesModel:
    model_cfg = config["model"]
    weights_cfg = config["tournament_weights"]

    matches = _filter_minnow_matches(matches, fifa_points, model_cfg["min_fifa_points_threshold"])

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = matches["home_team"].map(team_idx).values
    away_idx = matches["away_team"].map(team_idx).values
    home_goals = matches["home_score"].values.astype(float)
    away_goals = matches["away_score"].values.astype(float)
    host_adv = (~matches["neutral"].astype(bool)).values.astype(float)

    weights = time_weights(matches["date"], as_of, model_cfg["decay_half_life_days"])
    importance = matches["tournament"].apply(lambda t: tournament_weight(t, weights_cfg)).values
    weights = weights * importance

    prior_attack, prior_defense = _strength_priors(teams, fifa_points, elo_ratings, model_cfg["strength_prior_scale"])
    prior_weight = model_cfg["strength_prior_weight"]

    avg_goals = (home_goals.mean() + away_goals.mean()) / 2
    mu0 = float(np.log(avg_goals))
    x0 = np.concatenate([
        [mu0, model_cfg["rho_init"], 0.1],
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
