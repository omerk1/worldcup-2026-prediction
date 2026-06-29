"""Pick the scoreline that maximizes expected points in a prediction game
scored "`exact` points for the exact score, `direction` points for the
correct direction (home win / draw / away win) only" (not stacked).

Under that rule, the expected points of guessing scoreline (h, a) is:

    E(h, a) = exact * P(score = h-a) + direction * P(direction correct, not exact)
            = (exact - direction) * P(score = h-a) + direction * P(direction of h-a)

which is not always maximized by the single most likely cell in the score
matrix - a direction whose probability mass is concentrated in one cell can
beat a slightly more probable cell whose direction's overall probability is
lower.

Pass a 120-minute matrix (DixonColesModel.score_matrix_120) for knockout
rounds so that draw probabilities already reflect ET resolution.
"""
from __future__ import annotations

import numpy as np

_DIRECTIONS = {
    "home_win": lambda h, a: h > a,
    "draw": lambda h, a: h == a,
    "away_win": lambda h, a: h < a,
}


def best_guess(
    matrix: np.ndarray,
    direction_points: float = 1,
    exact_points: float = 3,
) -> dict:
    max_goals = matrix.shape[0] - 1

    p_home = float(sum(
        matrix[h, a]
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
        if h > a
    ))
    p_draw = float(sum(
        matrix[h, a]
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
        if h == a
    ))
    p_away = float(sum(
        matrix[h, a]
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
        if h < a
    ))

    direction_probs = {"home_win": p_home, "draw": p_draw, "away_win": p_away}

    candidates = []
    for direction, in_region in _DIRECTIONS.items():
        direction_prob = direction_probs[direction]
        score_prob, h, a = max(
            (matrix[h, a], h, a)
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
            if in_region(h, a)
        )
        expected_points = (exact_points - direction_points) * score_prob + direction_points * direction_prob
        candidates.append((expected_points, direction, h, a, float(score_prob), direction_prob))

    expected_points, direction, h, a, score_prob, direction_prob = max(candidates, key=lambda c: c[0])

    return {
        "best_guess_score": f"{h}-{a}",
        "direction": direction,
        "expected_points": expected_points,
        "score_prob": score_prob,
        "direction_prob": direction_prob,
    }
