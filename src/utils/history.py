"""Append timestamped prediction snapshots to a running history CSV.

Lets us look back at how predictions evolved over the tournament (e.g. to
compare a team's win probability before/after each of its matches), and
later compare past predictions against actual results.
"""
from pathlib import Path

import pandas as pd


def append_history(df: pd.DataFrame, history_path: Path, key_cols: list[str], generated_at: pd.Timestamp) -> None:
    """Append `df` to `history_path`, tagged with `generated_at`.

    Re-running on the same date replaces that date's previous snapshot for the
    same `key_cols` rather than duplicating it.
    """
    snapshot = df.copy()
    if "generated_at" not in snapshot.columns:
        snapshot.insert(0, "generated_at", generated_at.date().isoformat())
    else:
        snapshot["generated_at"] = generated_at.date().isoformat()

    history_path = Path(history_path)
    if history_path.exists():
        existing = pd.read_csv(history_path)
        is_rerun = (existing["generated_at"] == snapshot["generated_at"].iloc[0]) & existing[key_cols].apply(tuple, axis=1).isin(
            set(snapshot[key_cols].apply(tuple, axis=1))
        )
        existing = existing[~is_rerun]
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)
