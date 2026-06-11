# 2026 World Cup Prediction

Predicting 2026 FIFA World Cup match scores (and derived win/draw/loss probabilities) using a
Dixon-Coles Poisson model, with a FIFA-ranking-based prior to sensibly tier teams that have
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
4. **FIFA-ranking prior** — each team's rating is regularized toward a value implied by its
   latest FIFA ranking points. This anchors the "tier" of teams with sparse data (e.g.
   first-time qualifiers like Curaçao or Cape Verde) while letting data-rich teams (Brazil,
   France, ...) be driven by actual results.
5. **Host advantage** — only applied to Mexico/Canada/USA matches played in their own country
   (per `neutral` flag in the source data); all other 2026 matches are treated as neutral.

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
```

Note: team names follow the source dataset's conventions, e.g. `Czech Republic` (not
Czechia), `South Korea`, `Ivory Coast`, `United States`, `Curaçao`, `DR Congo`, `Cape Verde`.

## Project Structure

```
worldcup-2026-prediction/
├── src/
│   ├── data_processing/   # fetch_data.py, data_loader.py
│   ├── models/             # dixon_coles.py - core model
│   └── utils/              # config loading
├── configs/config.yaml     # data URLs + model hyperparameters
├── data/
│   ├── raw/                 # downloaded CSVs (not in git)
│   └── processed/           # fitted team_ratings.json (not in git)
├── train_ratings.py         # fit ratings from history
├── predict_match.py         # predict a single match
├── predict_fixtures.py      # predict all 72 group-stage matches
├── outputs/                  # prediction CSVs
└── tests/
```

## Configuration

All tunables are in `configs/config.yaml`:

- `lookback_years` / `decay_half_life_days` — how much history to use and how fast it decays
- `fifa_prior_scale` / `fifa_prior_weight` — how strongly FIFA ranking shapes team tiers
- `host_nations` — teams that get a home-advantage boost in their own country

## Status / Next Steps

- [x] Per-match score prediction (group stage)
- [ ] Knockout-stage / tiebreaker simulation (Monte Carlo tournament winner odds)
- [ ] Top scorer prediction (player-level goal rates from `goalscorers.csv`)
