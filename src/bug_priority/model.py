from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "high_priority"
NUMERIC_FEATURES = [
    "affected_users_count",
    "frequency_last_24h",
    "is_crash",
    "is_payment_related",
    "has_workaround",
    "days_since_release",
    "sla_breach_risk",
]
CATEGORICAL_FEATURES = ["environment", "module", "customer_tier"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    return pipeline


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, digits=3),
    }

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_model(pipeline: Pipeline, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)


def predict_bug(model: Pipeline, bug_input: pd.DataFrame) -> dict[str, Any]:
    probability = float(model.predict_proba(bug_input)[0, 1])
    prediction = int(model.predict(bug_input)[0])
    label = "High Priority" if prediction == 1 else "Normal Priority"
    return {
        "prediction": prediction,
        "label": label,
        "high_priority_probability": probability,
    }


def rank_bugs(model: Pipeline, bugs_df: pd.DataFrame) -> pd.DataFrame:
    ranked = bugs_df.copy()
    ranked["high_priority_probability"] = model.predict_proba(ranked[FEATURES])[:, 1]
    ranked["predicted_priority"] = ranked["high_priority_probability"].apply(
        lambda p: "High Priority" if p >= 0.5 else "Normal Priority"
    )
    ranked = ranked.sort_values(by="high_priority_probability", ascending=False).reset_index(drop=True)
    return ranked
