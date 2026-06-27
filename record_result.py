"""Record an actual 2026 World Cup result so future training runs can use it.

Usage:
    python record_result.py --date 2026-06-11 --home Mexico --away "South Africa" \
        --home-score 2 --away-score 0
"""
import argparse

import pandas as pd

from src.data_processing.data_loader import LIVE_RESULTS_COLUMNS, load_knockout_fixtures, load_live_results, infer_stage
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.history import record_actual_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Match date (YYYY-MM-DD)")
    parser.add_argument("--home", required=True, help="Home team name (as in results.csv)")
    parser.add_argument("--away", required=True, help="Away team name (as in results.csv)")
    parser.add_argument("--home-score", required=True, type=int)
    parser.add_argument("--away-score", required=True, type=int)
    args = parser.parse_args()

    config = load_config()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "wc_2026_live_results.csv"

    live = load_live_results(processed_dir)

    date = pd.Timestamp(args.date)
    knockout_fixtures = load_knockout_fixtures()
    stage = infer_stage(date, args.home, args.away, knockout_fixtures)

    mask = (
        (live["date"] == date)
        & (live["home_team"] == args.home)
        & (live["away_team"] == args.away)
    )

    row = {
        "date": date,
        "home_team": args.home,
        "away_team": args.away,
        "home_score": args.home_score,
        "away_score": args.away_score,
        "stage": stage,
    }

    if mask.any():
        live.loc[mask, ["home_score", "away_score", "stage"]] = [args.home_score, args.away_score, stage]
        print(f"Updated existing result: {args.home} {args.home_score}-{args.away_score} {args.away}")
    else:
        live = pd.concat([live, pd.DataFrame([row])], ignore_index=True)
        print(f"Recorded result: {args.home} {args.home_score}-{args.away_score} {args.away}")

    live = live[LIVE_RESULTS_COLUMNS].sort_values("date")
    live.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    history_dir = PROJECT_ROOT / "outputs" / "history"
    recorded_pred = record_actual_result(
        history_dir / "prematch_predictions.csv",
        date=date.date().isoformat(),
        home_team=args.home,
        away_team=args.away,
        actual_home_score=args.home_score,
        actual_away_score=args.away_score,
    )
    recorded_bg = record_actual_result(
        history_dir / "prematch_best_guess.csv",
        date=date.date().isoformat(),
        home_team=args.home,
        away_team=args.away,
        actual_home_score=args.home_score,
        actual_away_score=args.away_score,
    )
    if recorded_pred or recorded_bg:
        print("Recorded actual result on locked pre-match prediction(s)")


if __name__ == "__main__":
    main()
