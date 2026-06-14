"""Monte Carlo simulation of the 2026 World Cup: tournament winner odds.

Simulates the 72 group-stage fixtures, derives groups/standings, picks the 8
best third-placed teams, builds a simplified knockout bracket, and plays it
out to a champion - repeated many times to estimate each team's probability
of reaching each stage.

Example:
    python simulate_tournament.py --simulations 2000
"""
import argparse

import pandas as pd

from src.data_processing.data_loader import get_worldcup_2026_fixtures, load_results
from src.models.dixon_coles import DixonColesModel
from src.simulation.tournament import run_simulations
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.history import lock_first_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/processed/team_ratings.json")
    parser.add_argument("--out", default="outputs/worldcup_2026_simulation.csv")
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    model = DixonColesModel.load(PROJECT_ROOT / args.ratings)

    results = load_results()
    fixtures = get_worldcup_2026_fixtures(results)
    host_nations = set(config["model"]["host_nations"])

    print(f"Running {args.simulations} simulated tournaments...")
    df, groups = run_simulations(model, fixtures, host_nations, args.simulations, seed=args.seed)

    generated_at = pd.Timestamp.today().normalize()
    df.insert(0, "generated_at", generated_at.date().isoformat())

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote results for {len(df)} teams to {out_path}\n")

    pct = df.copy()
    for col in df.columns[2:]:
        pct[col] = (pct[col] * 100).round(1)
    print(pct.head(20).to_string(index=False))

    lock_path = PROJECT_ROOT / "outputs" / "history" / "pretournament_simulation.csv"
    if lock_first_snapshot(df, lock_path, generated_at=generated_at):
        print(f"\nLocked pre-tournament snapshot to {lock_path}")


if __name__ == "__main__":
    main()
