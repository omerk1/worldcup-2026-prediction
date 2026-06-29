"""Track pre-match/pre-tournament prediction snapshots for later evaluation
against actual results.
"""
from pathlib import Path

import pandas as pd


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


def update_predictions(df: pd.DataFrame, lock_path: Path, key_cols: list[str]) -> int:
    """Write `df`'s predictions for not-yet-played fixtures to `lock_path`,
    keyed by `key_cols`, leaving already-played fixtures untouched.

    A fixture already in `lock_path` with an actual result recorded is
    frozen there, capturing the prediction made right before that match was
    played. Every other fixture in `df` overwrites (or adds) the
    corresponding row in `lock_path`, so an unplayed fixture's pre-match
    prediction stays up to date with all results known so far (e.g. a
    team's second group game reflects its first game's result).

    `df` must contain the full current set of fixtures and have a
    `generated_at` column, which is renamed to `predicted_at`. Returns the
    number of fixtures in `df` not previously present in `lock_path` at all
    (e.g. newly-revealed knockout fixtures).
    """
    lock_path = Path(lock_path)
    snapshot = df.rename(columns={"generated_at": "predicted_at"}).copy()
    for col in key_cols:
        snapshot[col] = snapshot[col].astype(str)
    snapshot["actual_home_score"] = pd.array([None] * len(snapshot), dtype="Int64")
    snapshot["actual_away_score"] = pd.array([None] * len(snapshot), dtype="Int64")
    snapshot["actual_et_home_score"] = pd.array([None] * len(snapshot), dtype="Int64")
    snapshot["actual_et_away_score"] = pd.array([None] * len(snapshot), dtype="Int64")

    if lock_path.exists():
        existing = pd.read_csv(lock_path, float_precision="round_trip")
        existing["actual_home_score"] = existing["actual_home_score"].astype("Int64")
        existing["actual_away_score"] = existing["actual_away_score"].astype("Int64")
        if "stage" not in existing.columns and "stage" in snapshot.columns:
            existing["stage"] = "group_stage"
        if "actual_et_home_score" not in existing.columns:
            existing["actual_et_home_score"] = pd.array([None] * len(existing), dtype="Int64")
            existing["actual_et_away_score"] = pd.array([None] * len(existing), dtype="Int64")
        else:
            existing["actual_et_home_score"] = existing["actual_et_home_score"].astype("Int64")
            existing["actual_et_away_score"] = existing["actual_et_away_score"].astype("Int64")

        existing_keys = existing[key_cols].apply(tuple, axis=1)
        snapshot_keys = snapshot[key_cols].apply(tuple, axis=1)

        played_keys = set(existing_keys[existing["actual_home_score"].notna()])
        frozen = existing[existing_keys.isin(played_keys)]
        live = snapshot[~snapshot_keys.isin(played_keys)]

        newly_seen = int((~snapshot_keys.isin(set(existing_keys))).sum())
        combined = pd.concat([frozen, live], ignore_index=True)
    else:
        combined = snapshot
        newly_seen = len(combined)

    combined = combined.sort_values(key_cols).reset_index(drop=True)

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(lock_path, index=False)
    return newly_seen


def record_actual_result(
    lock_path: Path,
    date: str,
    home_team: str,
    away_team: str,
    actual_home_score: int,
    actual_away_score: int,
    actual_et_home_score: int | None = None,
    actual_et_away_score: int | None = None,
) -> bool:
    """Fill in the actual result for the locked pre-match prediction matching
    (date, home_team, away_team) in `lock_path`.

    For knockout games that go to extra time, pass the 120-minute score via
    `actual_et_home_score` / `actual_et_away_score`.

    No-op if `lock_path` doesn't exist or has no matching row. Returns whether
    a row was updated.
    """
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return False

    locked = pd.read_csv(lock_path, float_precision="round_trip")
    mask = (
        (locked["date"].astype(str) == str(date))
        & (locked["home_team"] == home_team)
        & (locked["away_team"] == away_team)
    )
    if not mask.any():
        return False

    locked["actual_home_score"] = locked["actual_home_score"].astype("Int64")
    locked["actual_away_score"] = locked["actual_away_score"].astype("Int64")
    if "actual_et_home_score" not in locked.columns:
        locked["actual_et_home_score"] = pd.array([None] * len(locked), dtype="Int64")
        locked["actual_et_away_score"] = pd.array([None] * len(locked), dtype="Int64")
    else:
        locked["actual_et_home_score"] = locked["actual_et_home_score"].astype("Int64")
        locked["actual_et_away_score"] = locked["actual_et_away_score"].astype("Int64")

    locked.loc[mask, "actual_home_score"] = actual_home_score
    locked.loc[mask, "actual_away_score"] = actual_away_score
    if actual_et_home_score is not None:
        locked.loc[mask, "actual_et_home_score"] = actual_et_home_score
        locked.loc[mask, "actual_et_away_score"] = actual_et_away_score
    locked.to_csv(lock_path, index=False)
    return True
