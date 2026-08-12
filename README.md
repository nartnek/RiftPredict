# RiftPredict

Predicts the winner of a League of Legends match from the champion draft alone, using supervised classification (KNN, Decision Tree, Random Forest, and a soft-voting Ensemble) trained on historical ranked match data pulled from the Riot Games Match-V5 API.

CMPT 310 — D100, Group 4
Ken Tran, Oliver Ancheta, Mark Cao, Eugene Park

## Requirements

- Python 3.14
- A Riot Games API key — **only needed if you want to re-run the crawler yourself.** Not required if you're just retraining on the included/cached match data or running the prediction interface.

If you do need it, create a `.env` file in the project root:
```
RIOT_API_KEY=your_riot_api_key_here
```

## Setup

```bash
git clone https://github.com/nartnek/RiftPredict.git
cd RiftPredict

python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Windows note:** use `python` instead of `python3` throughout this guide.

## Quick Start — run predictions right away
 
This repo already includes a trained model, so there's no need to collect data or train anything to try it out. As long as the files below are present, just run:
 
```bash
python -m src.ui.user_interface
```
 
Enter all 10 champion picks (5 per team, Top/Jungle/Mid/ADC/Support order) when prompted. The interface rejects invalid champion names and duplicate picks across the match, then prints a predicted winner, win probability, team strengths/weaknesses, and per-lane matchup reasoning.
 
**Files this needs** (all included in the repo):
 
| File | Purpose |
|---|---|
| `data/champions.json` | Static champion metadata (tags, stats) from Riot Data Dragon |
| `data/feature_matrices.json` | Precomputed AP ratio/variance, role frequency, and lane counter-matchup data |
| `saved_models/best_model.joblib` | The trained Ensemble model |
| `data/champion_winrates.joblib` | Champion win rates computed during training — must match what the saved model was trained on |
 
You do **not** need `data/raw_matches.csv` for this — it's already baked into the saved `.joblib` artifacts above.

## Running Tests

To run the full test suite from the project root, use:

```bash
python -m pytest tests/ -v
```

## Optional: collect your own data and retrain
 
Everything below is only necessary if you want to gather a fresh set of matches and train the models yourself, rather than using the ones already included.
 
### Collecting match data
 
This is a two-step manual process — there's no single "run the crawler" command, since `collect_matches()` has no built-in entry point.
 
**Step A — get a seed PUUID from a Riot ID:**
```bash
python get_puuid.py --game-name "SomeSummoner" --tag-line "NA1"
```
This prints a PUUID to the console — copy it for Step B.
 
**Step B — run the crawler with that seed PUUID:**
```python
from crawler import collect_matches, clean_and_split  # adjust import path to match your file location
 
df = collect_matches(["<puuid-from-step-A>"], target=1000)
clean_and_split(df)
```
`collect_matches()` starts from your seed player, automatically discovers new PUUIDs from every match it downloads (co-players), and keeps crawling until it hits `target` valid matches. It checkpoints progress to `data/raw_matches.csv` every 10 matches (`checkpoint_every=10`), and resumes automatically from that checkpoint if interrupted — safe to `Ctrl+C` and restart. Requires a `RIOT_API_KEY` set in a `.env` file in the project root (both scripts load it via `python-dotenv`).
 
**Note:** `clean_and_split()` also writes `data/clean_matches.csv`, `data/train.csv`, and `data/test.csv` — but the current pipeline (`encode_champions.py`) reads directly from `data/raw_matches.csv` and does its own train/test split, so `train.csv`/`test.csv` aren't currently used downstream. Worth deciding whether to keep that step or remove it.
 
Retraining additionally needs:
 
| File | Purpose |
|---|---|
| `data/raw_matches.csv` | Historical match data (champion picks + outcome per game) — used to compute win rates and train models |
 
### Retraining steps
 
Run these **in order** from the project root. Each step depends on the output of the previous one.
 
**1. Build features and compute champion win rates**
 
```bash
python -m src.preprocessing.encode_champions
```
 
This splits `raw_matches.csv` into train/test, computes Bayesian-smoothed champion win rates from the training split only (to avoid leakage), and saves them to `data/champion_winrates.joblib`, overwriting the one included in the repo.
 
Expected output (abbreviated):
```
Loaded 15000 raw matches, XXXX remain after dropping duplicates/remakes.
```
 
**2. Train and evaluate the models**
 
```bash
python -m src.models.train_and_evaluate
```
 
Trains KNN, Decision Tree, Random Forest, and a soft-voting Ensemble; saves each model to `saved_models/`, saves the Ensemble as `saved_models/best_model.joblib` (overwriting the included one), and writes evaluation metrics/charts to `results/`.
 
Expected output (abbreviated):
```
Training samples: ...
Testing samples: ...
--- Model Evaluation Results ---
   Model  Accuracy  Precision  Recall  Weighted F1
     KNN    0.5124     0.5085  0.5124       0.5077
      DT    0.5195     0.5198  0.5195       0.5196
      RF    0.5367     0.5321  0.5367       0.5250
Ensemble    0.5195     0.5198  0.5195       0.5196
Saved best overall model (Ensemble) to 'saved_models/best_model.joblib'
```
 
Check `results/metrics.csv`, `results/model_comparison.png`, `results/rf_feature_importance.png`, and `results/confusion_matrix_*.png` afterward.
 
**3. Try your retrained model**
 
```bash
python -m src.ui.user_interface
```



## Project structure

```
RiftPredict/
├── data/                          # champions.json, raw_matches.csv, feature_matrices.json
├── src/
│   ├── preprocessing/
│   │   └── encode_champions.py    # train/test split, win-rate computation
│   ├── feature_engineering/
│   │   ├── composition_features.py
│   │   └── feature_builder.py
│   ├── models/
│   │   ├── train_and_evaluate.py
│   │   └── analyze_team.py        # human-readable prediction explanations
│   └── ui/
│       └── user_interface.py      # interactive CLI
├── saved_models/                  # generated by train_and_evaluate.py
├── results/                       # generated by train_and_evaluate.py
├── tests/                         # unit tests
└── requirements.txt
```

