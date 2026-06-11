# 2026 World Cup Prediction

Predicting 2026 FIFA World Cup match scores (and derived win/draw/loss probabilities) using a
Dixon-Coles Poisson model, with a FIFA-ranking + Elo prior to sensibly tier teams that have
little recent international history.

## Approach

1. **Historical data** — ~12 years of international results (martj42 dataset), including
   recent 2025-26 friendlies/qualifiers.
2. **Dixon-Coles model** — each team gets an `attack` and `defense` rating. Expected goals:

   ```
   lambda_home = exp(mu + attack_home - defense_away + gamma * host_advantage)
   lambda_away = exp(mu + attack_away - defense_home)
   ```

   Scores follow Poisson(lambda) with the Dixon-Coles correction for low-scoring outcomes
   (0-0, 1-0, 0-1, 1-1).
3. **Time decay** — matches are weighted by recency (`exp(-decay_rate * days_ago)`, ~2 year
   half-life), so recent World Cups / qualifiers matter most without hard cutoffs.
4. **Tournament-importance weighting** — on top of time decay, each match is weighted by
   competition type (World Cup > continental championships > qualifiers/Nations League >
   friendlies), so a friendly played with weakened squads counts for much less than a World
   Cup match. The same weights scale the K-factor in the Elo ratings (point 6).
5. **Minnow filter** — historical matches where the weaker side's FIFA ranking points are below
   `min_fifa_points_threshold` are excluded from fitting, to avoid "blowout vs minnow" results
   distorting attack/defense ratings for World Cup-level matchups.
6. **FIFA ranking + Elo prior** — each team's rating is regularized toward a value blending its
   latest FIFA ranking points with a self-computed Elo rating (replayed chronologically from
   history, with a goal-difference multiplier and tournament-importance-scaled K-factor). This
   anchors the "tier" of teams with sparse data (e.g. first-time qualifiers like Curaçao or
   Cape Verde) while letting data-rich teams (Brazil, France, ...) be driven by actual results.
7. **Host advantage** — only applied to Mexico/Canada/USA matches played in their own country
   (per `neutral` flag in the source data); all other 2026 matches are treated as neutral.
8. **Live updates** — as real 2026 World Cup results come in, record them with
   `record_result.py`. They're merged into the training data (`get_played_matches`) so
   re-running `train_ratings.py` produces updated ratings/predictions for the rest of the
   tournament.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download historical results, goalscorers, FIFA rankings
python -m src.data_processing.fetch_data

# Fit team ratings (writes data/processed/team_ratings.json)
python train_ratings.py

# Predict a single match
python predict_match.py --home Mexico --away "South Africa" --host
python predict_match.py --home "South Korea" --away "Czech Republic"

# Predict all 72 group-stage fixtures (writes outputs/worldcup_2026_predictions.csv)
python predict_fixtures.py

# Record an actual result as the tournament progresses, then re-fit
python record_result.py --date 2026-06-11 --home Mexico --away "South Africa" \
    --home-score 2 --away-score 0
python train_ratings.py
```

Note: team names follow the source dataset's conventions, e.g. `Czech Republic` (not
Czechia), `South Korea`, `Ivory Coast`, `United States`, `Curaçao`, `DR Congo`, `Cape Verde`.

## Project Structure

```
worldcup-2026-prediction/
├── src/
│   ├── data_processing/   # fetch_data.py, data_loader.py
│   ├── models/             # dixon_coles.py - core model, elo.py - Elo ratings
│   └── utils/              # config loading
├── configs/config.yaml     # data URLs + model hyperparameters
├── data/
│   ├── raw/                 # downloaded CSVs (not in git)
│   └── processed/           # fitted team_ratings.json, wc_2026_live_results.csv (not in git)
├── train_ratings.py         # fit ratings from history
├── predict_match.py         # predict a single match
├── predict_fixtures.py      # predict all 72 group-stage matches
├── record_result.py         # record an actual 2026 WC result for future training
├── outputs/                  # prediction CSVs
└── tests/
```

## Configuration

All tunables are in `configs/config.yaml`:

- `lookback_years` / `decay_half_life_days` — how much history to use and how fast it decays
- `strength_prior_scale` / `strength_prior_weight` — how strongly the FIFA+Elo blend shapes
  team tiers
- `min_fifa_points_threshold` — matches where the weaker side is below this FIFA points
  threshold are excluded from fitting
- `host_nations` — teams that get a home-advantage boost in their own country
- `tournament_weights` — importance multipliers by competition type (World Cup, continental,
  qualifier, friendly, other), applied to both the Dixon-Coles fit and the Elo K-factor
- `elo` — initial rating, base K-factor, and home-advantage offset for the Elo replay

## Status / Next Steps

- [x] Per-match score prediction (group stage)
- [ ] Knockout-stage / tiebreaker simulation (Monte Carlo tournament winner odds)
- [ ] Top scorer prediction (player-level goal rates from `goalscorers.csv`)
