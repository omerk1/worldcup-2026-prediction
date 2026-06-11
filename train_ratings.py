"""Fit Dixon-Coles team ratings from historical results + FIFA/Elo strength prior."""
import argparse

import pandas as pd

from src.data_processing.data_loader import (
    get_played_matches,
    latest_fifa_points,
    load_current_rankings,
    load_fifa_ranking,
    load_live_results,
    load_results,
)
from src.models.dixon_coles import fit_dixon_coles
from src.models.elo import compute_elo_ratings
from src.utils.config_loader import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None, help="Fit using matches up to this date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--out", default="data/processed/team_ratings.json")
    args = parser.parse_args()

    config = load_config()
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()

    results = load_results()
    fifa = load_fifa_ranking()
    current_rankings = load_current_rankings()
    fifa_points = latest_fifa_points(fifa, current_rankings)
    live_results = load_live_results()

    matches = get_played_matches(results, as_of, config["model"]["lookback_years"], live_results)
    print(f"Fitting on {len(matches)} matches between {matches['date'].min().date()} and {matches['date'].max().date()}")

    elo_ratings = compute_elo_ratings(matches, config, as_of)

    model = fit_dixon_coles(matches, fifa_points, elo_ratings, config, as_of)
    print(f"mu={model.mu:.3f}  rho={model.rho:.3f}  gamma (host advantage)={model.gamma:.3f}")

    out_path = PROJECT_ROOT / args.out
    model.save(out_path)
    print(f"Saved ratings for {len(model.teams)} teams to {out_path}")

    # Quick sanity check: print attack/defense for the 2026 World Cup hosts and a few favorites
    sample = ["Mexico", "United States", "Canada", "Brazil", "Argentina", "France", "Senegal", "Ecuador"]
    print("\nteam            attack  defense  elo")
    for t in sample:
        if t in model.team_idx:
            a, d = model.ratings(t)
            print(f"{t:<15} {a:+.3f}  {d:+.3f}  {elo_ratings.get(t, 1500.0):.0f}")


if __name__ == "__main__":
    main()
