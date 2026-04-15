from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.bug_priority.log_parser import add_pattern_counts, parse_logs_to_dataframe

RANDOM_SEED = 42

SERVICES = [
    ("payment-service", "payments"),
    ("auth-service", "auth"),
    ("search-service", "search"),
    ("frontend-service", "ui"),
    ("reporting-service", "reporting"),
    ("notification-service", "notifications"),
    ("integrations-service", "integrations"),
]

LOG_TEMPLATES = {
    "payments": [
        "{ts} {level} payment-service Charge failed order_id={id} user={user} error={error} env={env} {signal}",
        "{ts} {level} checkout-service Checkout request rejected for user={user} error={error} env={env} {signal}",
    ],
    "auth": [
        "{ts} {level} auth-service Login failed for user={user} reason={error} env={env} {signal}",
        "{ts} {level} identity-service Session validation failed for user={user} reason={error} env={env} {signal}",
    ],
    "search": [
        "{ts} {level} search-service Search query failed user={user} error={error} env={env} {signal}",
    ],
    "ui": [
        "{ts} {level} frontend-service Checkout button not responding for browser=safari env={env} {signal}",
        "{ts} {level} ui-service Frontend exception during checkout env={env} {signal}",
    ],
    "reporting": [
        "{ts} {level} reporting-service Scheduled export failed tenant={tenant} error={error} env={env} {signal}",
    ],
    "notifications": [
        "{ts} {level} notification-service SMTP timeout for customer={user} env={env} {signal}",
    ],
    "integrations": [
        "{ts} {level} integrations-service Partner webhook failed partner={tenant} error={error} env={env} {signal}",
    ],
}

ERROR_CODES = {
    "payments": ["DB_TIMEOUT", "PAYMENT_GATEWAY_DOWN", "CARD_TOKEN_ERROR", "RATE_LIMIT"],
    "auth": ["TOKEN_EXPIRED", "SESSION_INVALID", "DB_TIMEOUT", "OUT_OF_MEMORY"],
    "search": ["SEARCH_TIMEOUT", "INDEX_UNAVAILABLE"],
    "ui": ["JS_EXCEPTION", "RENDER_TIMEOUT"],
    "reporting": ["EXPORT_TIMEOUT", "DB_TIMEOUT"],
    "notifications": ["SMTP_TIMEOUT", "RATE_LIMIT"],
    "integrations": ["WEBHOOK_TIMEOUT", "PARTNER_API_DOWN"],
}

SIGNALS = [
    "enterprise customer affected",
    "vip tenant impacted",
    "all users impacted",
    "repeated spike after deploy",
    "fallback available workaround",
    "sla breach risk",
    "post-deploy regression",
    "single request failure",
    "every request failing",
    "urgent sev1",
    "manual retry possible",
    "customer complaint opened",
]


@dataclass(frozen=True)
class GenerationConfig:
    rows: int = 1800
    seed: int = RANDOM_SEED


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_log_row(rng: np.random.Generator, idx: int) -> str:
    service_name, module = SERVICES[rng.integers(0, len(SERVICES))]
    env = rng.choice(["prod", "staging", "dev"], p=[0.55, 0.25, 0.20])
    level = rng.choice(["INFO", "WARN", "ERROR", "FATAL"], p=[0.08, 0.30, 0.50, 0.12])
    error = rng.choice(ERROR_CODES[module])
    template = rng.choice(LOG_TEMPLATES[module])

    chosen_signals = rng.choice(SIGNALS, size=rng.integers(1, 4), replace=False)
    signal = " ".join(chosen_signals.tolist())

    if module == "payments" and env == "prod":
        signal += " checkout impacted"
    if level == "FATAL":
        signal += " crash detected"
    if error in {"DB_TIMEOUT", "OUT_OF_MEMORY", "PAYMENT_GATEWAY_DOWN", "PARTNER_API_DOWN"}:
        signal += " exception raised"

    return template.format(
        ts=f"2026-04-15T10:{idx % 60:02d}:{(idx * 7) % 60:02d}Z",
        level=level,
        id=10000 + idx,
        user=2000 + idx,
        tenant=f"tenant-{idx % 40}",
        env=env,
        error=error,
        signal=signal.strip(),
    )


def build_dataset(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    logs = [_build_log_row(rng, idx) for idx in range(config.rows)]

    df = parse_logs_to_dataframe(logs)
    df = add_pattern_counts(df)

    score = (
        np.where(df["environment"] == "prod", 2.4, np.where(df["environment"] == "staging", 0.8, -1.0))
        + np.where(df["level"] == "FATAL", 2.5, np.where(df["level"] == "ERROR", 1.4, np.where(df["level"] == "WARN", 0.2, -1.0)))
        + np.where(df["module"] == "payments", 1.8, 0.0)
        + np.where(df["module"] == "auth", 1.2, 0.0)
        + np.where(df["customer_tier"] == "enterprise", 1.4, np.where(df["customer_tier"] == "pro", 0.5, 0.0))
        + 1.4 * df["is_crash"]
        + 1.2 * df["is_payment_related"]
        + 0.8 * df["is_db_error"]
        + 0.7 * df["is_timeout"]
        + 1.0 * df["is_oom"]
        + 0.9 * df["is_exception"]
        + 1.1 * df["sla_breach_risk"]
        - 1.2 * df["has_workaround"]
        + 0.45 * np.log1p(df["affected_users_count"])
        + 0.30 * np.log1p(df["frequency_last_24h"])
        + 0.35 * np.log1p(df["repeat_count_same_pattern"])
        - 0.03 * df["days_since_release"]
    )

    score += np.where((df["environment"] == "prod") & (df["module"] == "payments"), 1.0, 0.0)
    score += np.where((df["environment"] == "prod") & (df["level"] == "FATAL"), 1.1, 0.0)
    score += np.where((df["module"] == "ui") & (df["level"] == "WARN"), -1.0, 0.0)
    score += np.where((df["environment"] == "dev") & (df["has_workaround"] == 1), -1.2, 0.0)

    probability = _sigmoid(score + rng.normal(0, 1.0, size=config.rows) - 5.2)
    df["high_priority"] = rng.binomial(1, probability)
    return df


def save_dataset(output_path: str | Path, config: GenerationConfig = GenerationConfig()) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_dataset(config).to_csv(output_path, index=False)
    return output_path
