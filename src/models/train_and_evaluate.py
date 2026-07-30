import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np            
import seaborn as sns         
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from src.preprocessing.encode_champions import X_train, X_test, y_train, y_test

# Project root (.../RiftPredict), independent of the working directory
# the script is launched from. This is where the trained model is saved
# so that predict.py can always find it again.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

def plot_feature_importance(model, feature_names, top_n=10, save_path='feature_importance.png'):
    """
    Generates and saves a bar chart of the most important features.
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model does not support feature importances.")
            return

        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, palette="viridis")
        plt.title('Top Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Successfully saved {save_path}")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

def plot_model_comparison(model_scores, save_path='model_comparison.png'):
    """
    Generates and saves a bar chart comparing different models.
    """
    try:
        models = list(model_scores.keys())
        scores = list(model_scores.values())

        plt.figure(figsize=(8, 5))
        sns.barplot(x=models, y=scores, palette="mako")
        plt.title('Model Accuracy Comparison')
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy')
        
        for i, score in enumerate(scores):
            plt.text(i, score + 0.02, f"{score:.2f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Successfully saved {save_path}")
    except Exception as e:
        print(f"Could not generate model comparison plot: {e}")

def plot_win_probability(probability_array, team_names=['Blue Team', 'Red Team'], save_path='win_probability_bar.png'):
    """
    Generates a horizontal stacked bar for a single match's win probability.
    """
    try:
        blue_prob, red_prob = probability_array[0], probability_array[1]

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(0, blue_prob, color='#1f77b4', label=f"{team_names[0]} ({blue_prob*100:.1f}%)")
        ax.barh(0, red_prob, left=blue_prob, color='#d62728', label=f"{team_names[1]} ({red_prob*100:.1f}%)")

        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title("Predicted Win Probability (Sample Match)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Successfully saved {save_path}")
    except Exception as e:
        print(f"Could not generate win probability plot: {e}")
        
def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training labels: {sorted(y_train.unique())}")
    print(f"Testing labels: {sorted(y_test.unique())}")

    if len(X_train) < 5 or len(X_test) < 2:
        print("Not enough data to train/evaluate models reliably.")
        print("Try running with --count 20 or higher.")
        return

    if y_train.nunique() < 2:
        print("Training data only has one class.")
        print("The model needs both blue wins and red wins to train properly.")
        print("Try running with more matches, like --count 50.")
        return

    n_train = len(X_train)
    knn_neighbors = min(5, n_train)

    if knn_neighbors % 2 == 0 and knn_neighbors > 1:
        knn_neighbors -= 1

    print(f"Using KNN with n_neighbors={knn_neighbors}")

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=knn_neighbors),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(random_state=42),
    }

    metrics_list = []
    print("Training models...")

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        metrics_list.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Weighted F1": round(f1, 4),
        })

        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            labels=[0, 1],
            cmap="Blues",
            display_labels=["Loss (0)", "Win (1)"],
        )

        disp.ax_.set_title(f"{model_name} Confusion Matrix")
        plt.savefig(f"{results_dir}/confusion_matrix_{model_name.lower()}.png")
        plt.close()
        print(f"[{model_name}] Evaluated and confusion matrix saved.")

    metrics_df = pd.DataFrame(metrics_list)
    print("\n--- Model Evaluation Results ---")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(f"{results_dir}/metrics.csv", index=False)

    # --- SAVE BEST MODEL ---
    # "Best" = highest test-set accuracy (the same metric already driving
    # the model comparison chart below). Ties keep the first model in
    # dict order (KNN, DT, RF).
    best_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
    best_model_name = best_row["Model"]
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name} (Accuracy={best_row['Accuracy']})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            # Column order the model was trained on. predict.py re-applies
            # this order so a mismatch raises an error instead of silently
            # feeding the model the wrong feature in the wrong slot.
            "feature_names": list(X_train.columns),
            "metrics": best_row.to_dict(),
        },
        model_path,
    )

    print(f"Saved best model to {model_path}")

    # --- NEW VISUALIZATION CALLS ---

    # 1. Model Comparison (Replaces your old pandas plot with the seaborn one)
    # Convert our dataframe columns into a dictionary for the function: {'KNN': 0.85, 'DT': 0.78, ...}
    accuracy_dict = dict(zip(metrics_df["Model"], metrics_df["Accuracy"]))
    plot_model_comparison(accuracy_dict, save_path=f"{results_dir}/model_comparison.png")

    # We will use the Random Forest model for the next two plots as it performs best 
    # and naturally supports feature importances and probability predictions.
    rf_model = models["RF"]

    # 2. Feature Importance
    plot_feature_importance(rf_model, X_train.columns, save_path=f"{results_dir}/rf_feature_importance.png")

    # 3. Win Probability
    # Let's pull a single match from the test set to simulate a live prediction
    if hasattr(rf_model, "predict_proba"):
        # We use .iloc[[0]] to keep it as a DataFrame so sklearn doesn't throw a feature names warning
        sample_probs = rf_model.predict_proba(X_test.iloc[[0]])[0] 
        plot_win_probability(sample_probs, team_names=['Blue Team', 'Red Team'], save_path=f"{results_dir}/sample_match_probability.png")

    print(f"\nAll tasks complete! Check the '{results_dir}' folder for graphs and metrics CSV.")

if __name__ == "__main__":
    main()