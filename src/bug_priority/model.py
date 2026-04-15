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
    "days_since_release",
    "is_crash",
    "is_payment_related",
    "is_auth_related",
    "is_db_error",
    "is_timeout",
    "is_rate_limit",
    "is_oom",
    "is_exception",
    "is_failed",
    "has_customer_impact",
    "has_workaround",
    "sla_breach_risk",
    "repeat_count_same_pattern",
]
CATEGORICAL_FEATURES = ["environment", "level", "module", "customer_tier", "service_name", "error_code"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, NUMERIC_FEATURES), ("cat", categorical_transformer, CATEGORICAL_FEATURES)]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced"))]
    )


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
    return {"pipeline": pipeline, "metrics": metrics, "X_test": X_test, "y_test": y_test}


def save_model(pipeline: Pipeline, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)


def predict_log_features(model: Pipeline, log_features_df: pd.DataFrame) -> pd.DataFrame:
    result = log_features_df.copy()
    result["high_priority_probability"] = model.predict_proba(result[FEATURES])[:, 1]
    result["predicted_priority"] = result["high_priority_probability"].apply(
        lambda p: "High Priority" if p >= 0.5 else "Normal Priority"
    )
    return result


def predict_single_log(model: Pipeline, log_features_df: pd.DataFrame) -> dict[str, Any]:
    result = predict_log_features(model, log_features_df)
    row = result.iloc[0]
    return {
        "prediction": int(row["high_priority_probability"] >= 0.5),
        "label": row["predicted_priority"],
        "high_priority_probability": float(row["high_priority_probability"]),
        "row": row.to_dict(),
    }


def rank_logs(model: Pipeline, logs_df: pd.DataFrame) -> pd.DataFrame:
    ranked = predict_log_features(model, logs_df)
    return ranked.sort_values(by="high_priority_probability", ascending=False).reset_index(drop=True)
