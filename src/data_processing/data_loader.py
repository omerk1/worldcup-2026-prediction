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


def latest_fifa_points(fifa_df: pd.DataFrame, current_rankings: pd.DataFrame | None = None) -> pd.Series:
    """Return a Series mapping team name -> FIFA ranking points.

    `fifa_ranking.csv` is a periodically re-fetched snapshot that can lag the
    real FIFA rankings by a long time. If `current_rankings` (columns: team,
    rank) is given, those teams' points are re-estimated from their current
    rank position, mapped onto the points distribution of the latest snapshot
    (i.e. "team X is now ranked Nth, so give it the points the Nth-ranked team
    had in our snapshot"). This refreshes relative tiering for teams whose
    rank has moved a lot since the snapshot, without needing fresh point
    totals for the whole world.
    """
    latest_date = fifa_df["date"].max()
    latest = fifa_df[fifa_df["date"] == latest_date]
    points = latest.set_index("team")["total_points"].copy()

    if current_rankings is not None and len(current_rankings):
        ranked = points.dropna().sort_values(ascending=False).reset_index(drop=True)
        for _, row in current_rankings.iterrows():
            rank = int(row["rank"])
            if rank <= len(ranked):
                points.loc[row["team"]] = ranked.iloc[rank - 1]

    return points


def load_current_rankings(config_dir: Path | None = None) -> pd.DataFrame:
    """Manually-curated current FIFA rank positions (team, rank).

    Used by `latest_fifa_points` to refresh the points-based prior for teams
    where `fifa_ranking.csv` is stale. See configs/fifa_ranking_current.csv
    for the source and date of this snapshot.
    """
    config_dir = config_dir or (PROJECT_ROOT / "configs")
    path = config_dir / "fifa_ranking_current.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team", "rank"])
    return pd.read_csv(path)


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


LIVE_RESULTS_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "stage"]

GROUP_STAGE_CUTOFF = pd.Timestamp("2026-06-27")


def load_live_results(processed_dir: Path | None = None) -> pd.DataFrame:
    """Actual scores recorded for 2026 World Cup matches as they're played (record_result.py)."""
    processed_dir = processed_dir or (PROJECT_ROOT / "data" / "processed")
    path = processed_dir / "wc_2026_live_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=LIVE_RESULTS_COLUMNS)
    df = pd.read_csv(path, parse_dates=["date"])
    if "stage" not in df.columns:
        df["stage"] = "group_stage"
    return df


def load_knockout_fixtures(fixtures_dir: Path | None = None) -> pd.DataFrame:
    """Knockout-stage fixtures from data/fixtures/worldcup_2026_knockouts.csv."""
    fixtures_dir = fixtures_dir or (PROJECT_ROOT / "data" / "fixtures")
    path = fixtures_dir / "worldcup_2026_knockouts.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "home_team", "away_team", "stage", "city", "neutral"])
    return pd.read_csv(path, parse_dates=["date"])


def infer_stage(
    date: pd.Timestamp,
    home_team: str,
    away_team: str,
    knockout_fixtures: pd.DataFrame,
) -> str:
    """Return the competition stage for a match.

    Group-stage matches (on or before GROUP_STAGE_CUTOFF) always return
    "group_stage". For later dates, the match must appear in `knockout_fixtures`
    or a ValueError is raised — there is no silent fallback.
    """
    if date <= GROUP_STAGE_CUTOFF:
        return "group_stage"
    match = knockout_fixtures[
        (knockout_fixtures["date"] == date)
        & (knockout_fixtures["home_team"] == home_team)
        & (knockout_fixtures["away_team"] == away_team)
    ]
    if len(match) == 0:
        raise ValueError(
            f"No knockout fixture found for {home_team} vs {away_team} on "
            f"{date.date()} — add it to worldcup_2026_knockouts.csv first"
        )
    return str(match.iloc[0]["stage"])


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
        unplayed = results_df[results_df["home_score"].isna()]
        merged = unplayed.merge(
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
