"""Estimate 2026 World Cup Golden Boot contenders.

Combines each player's recency-weighted share of their national team's
historical goals with each team's expected total tournament goals
(attack rating x expected number of matches, the latter from
simulate_tournament.py's stage-progression probabilities).

Note: this is a "recent form" proxy, not squad-aware - it doesn't know
about injuries, retirements, or final squad selections.

Example:
    python predict_top_scorers.py --top 25
"""
import argparse

import pandas as pd

from src.data_processing.data_loader import get_worldcup_2026_fixtures, load_results
from src.models.dixon_coles import DixonColesModel
from src.models.top_scorer import predict_top_scorers
from src.simulation.tournament import derive_groups
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.history import append_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/processed/team_ratings.json")
    parser.add_argument("--simulation", default="outputs/worldcup_2026_simulation.csv")
    parser.add_argument("--goalscorers", default="data/raw/goalscorers.csv")
    parser.add_argument("--out", default="outputs/worldcup_2026_top_scorers.csv")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD, defaults to today")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    config = load_config()
    model = DixonColesModel.load(PROJECT_ROOT / args.ratings)
    results = load_results()
    goalscorers = pd.read_csv(PROJECT_ROOT / args.goalscorers, parse_dates=["date"])
    simulation = pd.read_csv(PROJECT_ROOT / args.simulation)

    fixtures = get_worldcup_2026_fixtures(results)
    groups = derive_groups(fixtures)
    teams = {t for ts in groups.values() for t in ts}

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
    half_life = config["model"]["decay_half_life_days"]

    df = predict_top_scorers(goalscorers, model, simulation, teams, as_of, half_life)
    df.insert(0, "generated_at", as_of.date().isoformat())

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} player projections to {out_path}\n")

    pretty = df.head(args.top).copy()
    pretty["goal_share"] = (pretty["goal_share"] * 100).round(1)
    pretty["expected_matches"] = pretty["expected_matches"].round(2)
    pretty["expected_goals"] = pretty["expected_goals"].round(2)
    print(pretty.to_string(index=False))

    history_path = PROJECT_ROOT / "outputs" / "history" / "top_scorers_history.csv"
    append_history(df, history_path, key_cols=["player", "team"], generated_at=as_of)
    print(f"\nAppended snapshot to {history_path}")


if __name__ == "__main__":
    main()
