"""Predict the score of a single match.

Example:
    python predict_match.py --home Mexico --away "South Africa" --host
    python predict_match.py --home "South Korea" --away "Czech Republic"
"""
import argparse

from src.models.dixon_coles import DixonColesModel
from src.utils.config_loader import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--host", action="store_true", help="Home team is playing in its own host country (gets home advantage)")
    parser.add_argument("--ratings", default="data/processed/team_ratings.json")
    args = parser.parse_args()

    load_config()  # validates config exists
    model = DixonColesModel.load(PROJECT_ROOT / args.ratings)

    for team in (args.home, args.away):
        if team not in model.team_idx:
            print(f"Warning: '{team}' not found in fitted ratings - using neutral (average) rating.")

    pred = model.predict(args.home, args.away, host=args.host)

    print(f"\n{pred['home_team']} vs {pred['away_team']}" + (" (host)" if args.host else " (neutral venue)"))
    print(f"Expected goals: {pred['expected_goals_home']:.2f} - {pred['expected_goals_away']:.2f}")
    print(f"Most likely score: {pred['most_likely_score']}")
    print(f"Win/Draw/Loss: {pred['home_win_prob']:.1%} / {pred['draw_prob']:.1%} / {pred['away_win_prob']:.1%}")
    print("\nTop scorelines:")
    for s in pred["top_scores"]:
        print(f"  {s['home_goals']}-{s['away_goals']}  {s['probability']:.1%}")


if __name__ == "__main__":
    main()
