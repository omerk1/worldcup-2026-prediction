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
        existing = pd.read_csv(history_path, float_precision="round_trip")
        is_rerun = (existing["generated_at"] == snapshot["generated_at"].iloc[0]) & existing[key_cols].apply(tuple, axis=1).isin(
            set(snapshot[key_cols].apply(tuple, axis=1))
        )
        existing = existing[~is_rerun]
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)


def lock_first_snapshot(df: pd.DataFrame, lock_path: Path, generated_at: pd.Timestamp) -> bool:
    """Write `df` to `lock_path`, tagged with `generated_at`, but only if
    `lock_path` doesn't already exist.

    Captures a one-time "pre-tournament" snapshot (e.g. tournament-winner or
    Golden Boot odds) for later comparison against the actual outcome, without
    being overwritten by later re-runs. Returns whether the snapshot was
    written.
    """
    lock_path = Path(lock_path)
    if lock_path.exists():
        return False

    snapshot = df.copy()
    if "generated_at" not in snapshot.columns:
        snapshot.insert(0, "generated_at", generated_at.date().isoformat())
    else:
        snapshot["generated_at"] = generated_at.date().isoformat()

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(lock_path, index=False)
    return True


def lock_prematch(
    history_path: Path,
    lock_path: Path,
    date: str,
    home_team: str,
    away_team: str,
    actual_home_score: int,
    actual_away_score: int,
) -> bool:
    """Snapshot the earliest prediction for (date, home_team, away_team) from
    `history_path` into `lock_path`, alongside the actual result.

    This captures "what the model predicted before we knew the outcome",
    immune to later re-runs (which retrain on the result itself). No-op if
    `history_path` has no matching prediction, or this match is already
    locked. Returns whether a row was added.
    """
    lock_path = Path(lock_path)
    locked = None
    if lock_path.exists():
        locked = pd.read_csv(lock_path, float_precision="round_trip")
        already = (
            (locked["date"].astype(str) == str(date))
            & (locked["home_team"] == home_team)
            & (locked["away_team"] == away_team)
        ).any()
        if already:
            return False

    history_path = Path(history_path)
    if not history_path.exists():
        return False

    history = pd.read_csv(history_path, float_precision="round_trip")
    match_rows = history[
        (history["date"].astype(str) == str(date))
        & (history["home_team"] == home_team)
        & (history["away_team"] == away_team)
    ]
    if match_rows.empty:
        return False

    earliest = match_rows.sort_values("generated_at").iloc[0].rename({"generated_at": "predicted_at"})
    earliest["actual_home_score"] = actual_home_score
    earliest["actual_away_score"] = actual_away_score

    row_df = pd.DataFrame([earliest])
    combined = pd.concat([locked, row_df], ignore_index=True) if locked is not None else row_df

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(lock_path, index=False)
    return True
