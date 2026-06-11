import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel, _strength_priors, fit_dixon_coles


def make_model():
    return DixonColesModel(
        teams=["A", "B"],
        mu=0.2,
        rho=-0.05,
        gamma=0.25,
        attack=np.array([0.3, -0.1]),
        defense=np.array([0.2, -0.2]),
    )


def test_score_matrix_sums_to_one():
    model = make_model()
    matrix = model.score_matrix("A", "B", host=False)
    assert np.isclose(matrix.sum(), 1.0)


def test_predict_outcome_probs_sum_to_one():
    model = make_model()
    pred = model.predict("A", "B", host=False)
    total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
    assert np.isclose(total, 1.0, atol=1e-6)


def test_host_advantage_increases_home_expected_goals():
    model = make_model()
    lam_home_neutral, _ = model.expected_goals("A", "B", host=False)
    lam_home_host, _ = model.expected_goals("A", "B", host=True)
    assert lam_home_host > lam_home_neutral


def test_strength_prior_ranks_strong_team_above_weak_team():
    teams = ["Strong", "Weak"]
    fifa_points = pd.Series({"Strong": 1900.0, "Weak": 900.0})
    elo_ratings = {"Strong": 1700.0, "Weak": 1300.0}
    attack_prior, defense_prior = _strength_priors(teams, fifa_points, elo_ratings, scale=0.35)
    assert attack_prior[0] > attack_prior[1]
    assert defense_prior[0] > defense_prior[1]


def test_strength_prior_handles_missing_data():
    teams = ["Strong", "Weak", "Unknown"]
    fifa_points = pd.Series({"Strong": 1900.0, "Weak": 900.0})
    elo_ratings = {"Strong": 1700.0, "Weak": 1300.0}
    attack_prior, defense_prior = _strength_priors(teams, fifa_points, elo_ratings, scale=0.35)
    assert np.isfinite(attack_prior).all()
    assert np.isfinite(defense_prior).all()
    assert attack_prior[2] == 0.0


def test_fit_dixon_coles_recovers_reasonable_params():
    rng = np.random.default_rng(0)
    n_matches = 200
    dates = pd.date_range("2024-01-01", periods=n_matches, freq="3D")
    teams = ["A", "B", "C", "D"]
    home = rng.choice(teams, n_matches)
    away = []
    for h in home:
        choices = [t for t in teams if t != h]
        away.append(rng.choice(choices))
    home_score = rng.poisson(1.4, n_matches)
    away_score = rng.poisson(1.0, n_matches)

    matches = pd.DataFrame({
        "date": dates,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "neutral": False,
        "tournament": "Friendly",
    })

    fifa_points = pd.Series({t: 1500.0 for t in teams})
    elo_ratings = {t: 1500.0 for t in teams}
    config = {
        "model": {
            "decay_half_life_days": 730,
            "strength_prior_scale": 0.35,
            "strength_prior_weight": 5.0,
            "rho_init": -0.05,
            "min_fifa_points_threshold": 1100,
        },
        "tournament_weights": {
            "world_cup": 2.5,
            "continental": 1.5,
            "qualifier": 1.0,
            "friendly": 0.3,
            "other": 0.5,
            "world_cup_names": ["FIFA World Cup"],
            "continental_names": [],
            "qualifier_keywords": ["qualification", "Nations League"],
        },
    }
    model = fit_dixon_coles(matches, fifa_points, elo_ratings, config, as_of=dates.max())
    assert set(model.teams) == set(teams)
    assert model.gamma >= 0
