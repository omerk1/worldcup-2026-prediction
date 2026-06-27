"""Predict scores for all World Cup 2026 fixtures (group stage + knockouts)."""
import argparse

import pandas as pd

from src.data_processing.data_loader import get_worldcup_2026_fixtures, load_knockout_fixtures, load_results
from src.models.dixon_coles import DixonColesModel
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.history import update_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/processed/team_ratings.json")
    parser.add_argument("--out", default="outputs/worldcup_2026_predictions.csv")
    parser.add_argument("--from-date", default=None, help="Only show fixtures on/after this date (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    config = load_config()
    model = DixonColesModel.load(PROJECT_ROOT / args.ratings)

    results = load_results()
    group_fixtures = get_worldcup_2026_fixtures(results)
    group_fixtures = group_fixtures.assign(stage="group_stage")
    knockout_fixtures = load_knockout_fixtures()
    fixtures = pd.concat(
        [group_fixtures[["date", "home_team", "away_team", "stage", "city", "neutral"]],
         knockout_fixtures[["date", "home_team", "away_team", "stage", "city", "neutral"]]],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)

    host_nations = set(config["model"]["host_nations"])

    rows = []
    for _, row in fixtures.iterrows():
        host = (not row["neutral"]) and (row["home_team"] in host_nations)
        pred = model.predict(row["home_team"], row["away_team"], host=host)
        rows.append({
            "date": row["date"].date(),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "stage": row["stage"],
            "city": row["city"],
            "host_advantage": host,
            "expected_goals_home": round(pred["expected_goals_home"], 2),
            "expected_goals_away": round(pred["expected_goals_away"], 2),
            "predicted_score": pred["most_likely_score"],
            "home_win_prob": round(pred["home_win_prob"], 3),
            "draw_prob": round(pred["draw_prob"], 3),
            "away_win_prob": round(pred["away_win_prob"], 3),
        })

    generated_at = pd.Timestamp.today().normalize()

    out_df = pd.DataFrame(rows)
    out_df.insert(0, "generated_at", generated_at.date().isoformat())

    lock_path = PROJECT_ROOT / "outputs" / "history" / "prematch_predictions.csv"
    newly_seen = update_predictions(out_df, lock_path, key_cols=["date", "home_team", "away_team"])

    from_date = args.from_date or generated_at.date().isoformat()
    display_df = out_df[out_df["date"].astype(str) >= from_date]

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display_df.to_csv(out_path, index=False)
    print(f"Wrote {len(display_df)} predictions to {out_path}\n")
    print(display_df.to_string(index=False))

    print(f"\nUpdated pre-match predictions for unplayed fixtures in {lock_path}")
    if newly_seen:
        print(f"({newly_seen} fixture(s) seen for the first time)")


if __name__ == "__main__":
    main()
