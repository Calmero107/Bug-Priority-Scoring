from pathlib import Path

import pandas as pd

from src.bug_priority.model import save_model, train_model


DATA_PATH = Path("data") / "bugs.csv"
MODEL_PATH = Path("models") / "bug_priority_pipeline.joblib"


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    result = train_model(df)
    save_model(result["pipeline"], MODEL_PATH)

    metrics = result["metrics"]
    print("Model saved to:", MODEL_PATH)
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC AUC:", round(metrics["roc_auc"], 4))
    print("Confusion Matrix:", metrics["confusion_matrix"])
    print("\nClassification Report:\n")
    print(metrics["classification_report"])
