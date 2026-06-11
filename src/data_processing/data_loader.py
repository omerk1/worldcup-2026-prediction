"""Load historical results, FIFA rankings, and 2026 World Cup fixtures."""
from pathlib import Path

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT

# results.csv (martj42) uses "current team name" conventions that sometimes
# differ from the names used in the FIFA ranking dataset. Mapping is only
# needed for the 2026 World Cup teams where the names diverge.
FIFA_NAME_MAP = {
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "New Zealand": "Aotearoa New Zealand",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "United States": "USA",
    "Cape Verde": "Cabo Verde",
}


def load_results(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or (PROJECT_ROOT / "data" / "raw")
    df = pd.read_csv(raw_dir / "results.csv", parse_dates=["date"])
    return df


def load_fifa_ranking(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or (PROJECT_ROOT / "data" / "raw")
    df = pd.read_csv(raw_dir / "fifa_ranking.csv", parse_dates=["date"])
    return df


def latest_fifa_points(fifa_df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping team name -> latest known FIFA ranking points."""
    latest_date = fifa_df["date"].max()
    latest = fifa_df[fifa_df["date"] == latest_date]
    return latest.set_index("team")["total_points"]


def tournament_weight(tournament: str, weights: dict) -> float:
    """Map a tournament name to an importance weight (config['tournament_weights'])."""
    if tournament == "Friendly":
        return weights["friendly"]
    if tournament in weights["world_cup_names"]:
        return weights["world_cup"]
    if tournament in weights["continental_names"]:
        return weights["continental"]
    if any(kw in tournament for kw in weights["qualifier_keywords"]):
        return weights["qualifier"]
    return weights["other"]


LIVE_RESULTS_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score"]


def load_live_results(processed_dir: Path | None = None) -> pd.DataFrame:
    """Actual scores recorded for 2026 World Cup matches as they're played (record_result.py)."""
    processed_dir = processed_dir or (PROJECT_ROOT / "data" / "processed")
    path = processed_dir / "wc_2026_live_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=LIVE_RESULTS_COLUMNS)
    return pd.read_csv(path, parse_dates=["date"])


def get_played_matches(
    results_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_years: int,
    live_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Matches with known scores within the lookback window, up to `as_of`.

    `live_results_df` (date, home_team, away_team, home_score, away_score) overrides
    scores for fixtures that were unplayed in `results_df` (e.g. 2026 World Cup
    matches recorded via record_result.py as the tournament progresses).
    """
    cutoff = as_of - pd.DateOffset(years=lookback_years)
    played = results_df[
        results_df["home_score"].notna()
        & (results_df["date"] >= cutoff)
        & (results_df["date"] <= as_of)
    ].copy()

    if live_results_df is not None and len(live_results_df):
        merged = results_df.merge(
            live_results_df, on=["date", "home_team", "away_team"], suffixes=("", "_live")
        )
        merged = merged[merged["home_score_live"].notna()]
        merged["home_score"] = merged["home_score_live"]
        merged["away_score"] = merged["away_score_live"]
        merged = merged.drop(columns=["home_score_live", "away_score_live"])
        merged = merged[(merged["date"] >= cutoff) & (merged["date"] <= as_of)]
        played = pd.concat([played, merged], ignore_index=True)

    return played


def get_worldcup_2026_fixtures(results_df: pd.DataFrame) -> pd.DataFrame:
    """The 72 group-stage fixtures for the 2026 World Cup (scores not yet played)."""
    fixtures = results_df[
        (results_df["tournament"] == "FIFA World Cup")
        & (results_df["date"] >= "2026-06-11")
        & (results_df["date"] <= "2026-06-27")
    ].copy()
    return fixtures.sort_values("date").reset_index(drop=True)
