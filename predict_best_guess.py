"""Pick the scoreline that maximizes expected points in a prediction game
scored "3 points for exact score, 1 point for correct direction only".

This is a different objective than `predict_fixtures.py`'s `predicted_score`
(the single most likely scoreline overall): it picks, for each match, the
scoreline that maximizes 2*P(score) + P(direction), which can favor a less
likely scoreline whose direction has much higher overall probability.

Example:
    python predict_best_guess.py
"""
import argparse

import pandas as pd

from src.data_processing.data_loader import get_worldcup_2026_fixtures, load_results
from src.models.best_guess import best_guess
from src.models.dixon_coles import DixonColesModel
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.history import append_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/processed/team_ratings.json")
    parser.add_argument("--out", default="outputs/worldcup_2026_best_guess.csv")
    parser.add_argument("--from-date", default=None, help="Only show fixtures on/after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config()
    model = DixonColesModel.load(PROJECT_ROOT / args.ratings)

    results = load_results()
    fixtures = get_worldcup_2026_fixtures(results)
    if args.from_date:
        fixtures = fixtures[fixtures["date"] >= args.from_date]

    host_nations = set(config["model"]["host_nations"])
    scoring = config["scoring"]["group_stage"]

    rows = []
    for _, row in fixtures.iterrows():
        host = (not row["neutral"]) and (row["home_team"] in host_nations)
        matrix = model.score_matrix(row["home_team"], row["away_team"], host=host)
        pred = model.predict(row["home_team"], row["away_team"], host=host)
        guess = best_guess(matrix, direction_points=scoring["direction"], exact_points=scoring["exact"])
        rows.append({
            "date": row["date"].date(),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "best_guess_score": guess["best_guess_score"],
            "direction": guess["direction"],
            "expected_points": round(guess["expected_points"], 3),
            "most_likely_score": pred["most_likely_score"],
        })

    generated_at = pd.Timestamp.today().normalize()

    out_df = pd.DataFrame(rows)
    out_df.insert(0, "generated_at", generated_at.date().isoformat())
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} best-guess picks to {out_path}\n")
    print(out_df.to_string(index=False))

    history_path = PROJECT_ROOT / "outputs" / "history" / "best_guess_history.csv"
    append_history(out_df, history_path, key_cols=["date", "home_team", "away_team"], generated_at=generated_at)
    print(f"\nAppended snapshot to {history_path}")


if __name__ == "__main__":
    main()
