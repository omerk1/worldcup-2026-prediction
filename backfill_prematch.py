"""Backfill outputs/history/prematch_predictions.csv and
outputs/history/prematch_best_guess.csv for results already recorded in
wc_2026_live_results.csv (e.g. results recorded before the lock-in mechanism
in record_result.py existed).

Idempotent - skips matches already locked.

Example:
    python backfill_prematch.py
"""
import pandas as pd

from src.data_processing.data_loader import load_live_results
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.history import lock_prematch


def main():
    config = load_config()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    live = load_live_results(processed_dir)

    history_dir = PROJECT_ROOT / "outputs" / "history"
    locked = 0
    for _, row in live.iterrows():
        date = pd.Timestamp(row["date"]).date().isoformat()
        for history_name, lock_name in [
            ("predictions_history.csv", "prematch_predictions.csv"),
            ("best_guess_history.csv", "prematch_best_guess.csv"),
        ]:
            if lock_prematch(
                history_dir / history_name,
                history_dir / lock_name,
                date=date,
                home_team=row["home_team"],
                away_team=row["away_team"],
                actual_home_score=int(row["home_score"]),
                actual_away_score=int(row["away_score"]),
            ):
                locked += 1

    print(f"Locked {locked} pre-match prediction rows.")


if __name__ == "__main__":
    main()
