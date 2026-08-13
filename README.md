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
 
This repo already includes a trained model, so there's no need to collect data or train anything to try it out. You can just run:
 
```bash
python -m src.ui.user_interface
```
 
Enter all 10 champion picks (5 per team, Top/Jungle/Mid/ADC/Support order) when prompted. The interface rejects invalid champion names and duplicate picks across the match, then prints a predicted winner, win probability, team strengths/weaknesses, and per-lane matchup reasoning.
 

## Running Tests

To run the full test suite from the project root, use:

```bash
python -m pytest tests/ -v
```

## Optional: collect your own data and retrain
 
Everything below is only necessary if you want to gather a fresh set of matches and train the models yourself, rather than using the ones already included.
 
### Collecting match data (and optionally retraining) in one command
 
```bash
python main.py \
  --game-name "SomeSummoner" \
  --tag-line "NA1" \
  --target 1000 \
  --matches-per-player 30
```
 
This looks up the PUUID for the given Riot ID, then crawls outward from that player — automatically discovering new players from every match it downloads — until it collects `--target` valid matches. It checkpoints progress to `data/raw_matches.csv` every 10 matches and resumes automatically if interrupted (safe to `Ctrl+C` and restart). Requires `RIOT_API_KEY` set in `.env`.
 
Add `--train` to also run the full retraining pipeline (feature building + all four models) immediately after collection finishes, in the same command:
 
```bash
python main.py --game-name "SomeSummoner" --tag-line "NA1" --target 1000 --train
```
 
Other flags: `--matches-per-player` (default 50) controls how many recent matches are checked per discovered player; `--queue` (default 420, ranked solo/duo — use 440 for ranked flex).
 
**Note:** `clean_and_split()` (called internally by `main.py`) also writes `data/clean_matches.csv`, `data/train.csv`, and `data/test.csv` — but the current pipeline (`encode_champions.py`) reads directly from `data/raw_matches.csv` and does its own train/test split, so `train.csv`/`test.csv` aren't currently used downstream. Worth deciding whether to keep that step or remove it.

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
├── main.py                        # entry point: collect data (+ optionally retrain)
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

