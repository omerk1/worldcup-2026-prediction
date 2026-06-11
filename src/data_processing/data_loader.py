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


def get_played_matches(results_df: pd.DataFrame, as_of: pd.Timestamp, lookback_years: int) -> pd.DataFrame:
    """Matches with known scores within the lookback window, up to `as_of`."""
    cutoff = as_of - pd.DateOffset(years=lookback_years)
    played = results_df[
        results_df["home_score"].notna()
        & (results_df["date"] >= cutoff)
        & (results_df["date"] <= as_of)
    ].copy()
    return played


def get_worldcup_2026_fixtures(results_df: pd.DataFrame) -> pd.DataFrame:
    """The 72 group-stage fixtures for the 2026 World Cup (scores not yet played)."""
    fixtures = results_df[
        (results_df["tournament"] == "FIFA World Cup")
        & (results_df["date"] >= "2026-06-11")
        & (results_df["date"] <= "2026-06-27")
    ].copy()
    return fixtures.sort_values("date").reset_index(drop=True)
