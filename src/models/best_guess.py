"""Pick the scoreline that maximizes expected points in a prediction game
scored "3 points for the exact score, 1 point for the correct direction
(home win / draw / away win) only".

Under that rule, the expected points of guessing scoreline (h, a) is:

    E(h, a) = 3 * P(exact score = h-a) + 1 * P(direction correct, not exact)
            = 2 * P(score = h-a) + P(direction of h-a)

which is not always maximized by the single most likely cell in the score
matrix - a direction whose probability mass is concentrated in one cell can
beat a slightly more probable cell whose direction's overall probability is
lower.
"""
from __future__ import annotations

import numpy as np

_DIRECTIONS = {
    "home_win": lambda h, a: h > a,
    "draw": lambda h, a: h == a,
    "away_win": lambda h, a: h < a,
}


def best_guess(matrix: np.ndarray) -> dict:
    max_goals = matrix.shape[0] - 1

    candidates = []
    for direction, in_region in _DIRECTIONS.items():
        direction_prob = float(sum(
            matrix[h, a]
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
            if in_region(h, a)
        ))
        score_prob, h, a = max(
            (matrix[h, a], h, a)
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
            if in_region(h, a)
        )
        expected_points = 2 * score_prob + direction_prob
        candidates.append((expected_points, direction, h, a, float(score_prob), direction_prob))

    expected_points, direction, h, a, score_prob, direction_prob = max(candidates, key=lambda c: c[0])

    return {
        "best_guess_score": f"{h}-{a}",
        "direction": direction,
        "expected_points": expected_points,
        "score_prob": score_prob,
        "direction_prob": direction_prob,
    }
