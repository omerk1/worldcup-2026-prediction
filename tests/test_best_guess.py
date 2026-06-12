import numpy as np

from src.models.best_guess import best_guess


def test_best_guess_can_diverge_from_most_likely_score():
    # Argmax of the whole matrix is (1,1)=0.20 (a draw), but the best
    # home-win cell (1,0)=0.15 combined with home_win_prob=0.40 yields higher
    # expected points under "3 for exact, 1 for direction only".
    matrix = np.array([
        [0.03, 0.15, 0.10],
        [0.15, 0.20, 0.10],
        [0.13, 0.12, 0.02],
    ])

    guess = best_guess(matrix)

    assert guess["best_guess_score"] == "1-0"
    assert guess["direction"] == "home_win"
    assert round(guess["expected_points"], 2) == 0.70


def test_best_guess_picks_highest_probability_cell_when_direction_dominates():
    matrix = np.array([
        [0.40, 0.05],
        [0.05, 0.50],
    ])

    guess = best_guess(matrix)

    assert guess["best_guess_score"] == "1-1"
    assert guess["direction"] == "draw"
    assert round(guess["expected_points"], 2) == 1.90
