import os
import pandas as pd
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

    # Prefer an odd number for KNN if possible
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

    metrics_df.plot(
        x="Model",
        y=["Accuracy", "Weighted F1"],
        kind="bar",
        figsize=(8, 6),
    )

    plt.title("Baseline Models: Accuracy vs F1 Score")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(f"{results_dir}/model_comparison.png")
    plt.close()

    print(f"\nAll tasks complete! Check the '{results_dir}' folder for graphs and metrics CSV.")


if __name__ == "__main__":
    main()