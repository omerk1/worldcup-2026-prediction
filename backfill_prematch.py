"""Fill in actual scores on outputs/history/prematch_predictions.csv and
outputs/history/prematch_best_guess.csv for every result already recorded in
wc_2026_live_results.csv.

Safe to rerun - re-recording an already-filled-in actual is a no-op write of
the same value. Useful as a one-off safety net if a result was added to
wc_2026_live_results.csv without going through record_result.py.

Example:
    python backfill_prematch.py
"""
import pandas as pd

from src.data_processing.data_loader import load_live_results
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.history import record_actual_result


def main():
    config = load_config()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    live = load_live_results(processed_dir)

    history_dir = PROJECT_ROOT / "outputs" / "history"
    recorded = 0
    for _, row in live.iterrows():
        date = pd.Timestamp(row["date"]).date().isoformat()
        for lock_name in ["prematch_predictions.csv", "prematch_best_guess.csv"]:
            if record_actual_result(
                history_dir / lock_name,
                date=date,
                home_team=row["home_team"],
                away_team=row["away_team"],
                actual_home_score=int(row["home_score"]),
                actual_away_score=int(row["away_score"]),
            ):
                recorded += 1

    print(f"Recorded {recorded} actual result(s) on locked pre-match predictions.")


if __name__ == "__main__":
    main()
