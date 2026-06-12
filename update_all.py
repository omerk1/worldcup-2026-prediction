"""Run the full daily update pipeline in one go.

Steps: fetch latest data -> refit team ratings -> regenerate fixture
predictions -> regenerate best-guess picks -> regenerate tournament
simulation -> regenerate top scorer projections.

Example:
    python update_all.py
    python update_all.py --simulations 5000 --seed 42
    python update_all.py --skip-fetch
"""
import argparse
import subprocess
import sys

from src.utils.config_loader import PROJECT_ROOT


def run(*args: str) -> None:
    print(f"\n=== {' '.join(args)} ===")
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true", help="Skip re-downloading source data")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_fetch:
        run("-m", "src.data_processing.fetch_data")

    run("train_ratings.py")
    run("predict_fixtures.py")
    run("predict_best_guess.py")

    sim_args = ["simulate_tournament.py", "--simulations", str(args.simulations)]
    if args.seed is not None:
        sim_args += ["--seed", str(args.seed)]
    run(*sim_args)

    run("predict_top_scorers.py")

    print("\nAll outputs refreshed.")


if __name__ == "__main__":
    main()
