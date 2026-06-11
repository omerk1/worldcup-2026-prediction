"""Predict scores for all 72 World Cup 2026 group-stage fixtures."""
import argparse

import pandas as pd

from src.data_processing.data_loader import get_worldcup_2026_fixtures, load_results
from src.models.dixon_coles import DixonColesModel
from src.utils.config_loader import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/processed/team_ratings.json")
    parser.add_argument("--out", default="outputs/worldcup_2026_predictions.csv")
    parser.add_argument("--from-date", default=None, help="Only show fixtures on/after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config()
    model = DixonColesModel.load(PROJECT_ROOT / args.ratings)

    results = load_results()
    fixtures = get_worldcup_2026_fixtures(results)
    if args.from_date:
        fixtures = fixtures[fixtures["date"] >= args.from_date]

    host_nations = set(config["model"]["host_nations"])

    rows = []
    for _, row in fixtures.iterrows():
        host = (not row["neutral"]) and (row["home_team"] in host_nations)
        pred = model.predict(row["home_team"], row["away_team"], host=host)
        rows.append({
            "date": row["date"].date(),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "city": row["city"],
            "host_advantage": host,
            "expected_goals_home": round(pred["expected_goals_home"], 2),
            "expected_goals_away": round(pred["expected_goals_away"], 2),
            "predicted_score": pred["most_likely_score"],
            "home_win_prob": round(pred["home_win_prob"], 3),
            "draw_prob": round(pred["draw_prob"], 3),
            "away_win_prob": round(pred["away_win_prob"], 3),
        })

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} predictions to {out_path}\n")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
