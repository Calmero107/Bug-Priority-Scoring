from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

ENVIRONMENTS = ["prod", "staging", "dev"]
MODULES = ["auth", "payments", "search", "ui", "reporting", "notifications", "integrations"]
CUSTOMER_TIERS = ["free", "pro", "enterprise"]


@dataclass(frozen=True)
class GenerationConfig:
    rows: int = 1200
    seed: int = RANDOM_SEED


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_dataset(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)

    environment = rng.choice(ENVIRONMENTS, size=config.rows, p=[0.5, 0.25, 0.25])
    module = rng.choice(MODULES, size=config.rows, p=[0.18, 0.18, 0.14, 0.18, 0.1, 0.12, 0.1])
    customer_tier = rng.choice(CUSTOMER_TIERS, size=config.rows, p=[0.55, 0.3, 0.15])

    affected_users = np.clip(rng.lognormal(mean=3.2, sigma=1.0, size=config.rows).astype(int), 1, 5000)
    frequency_last_24h = np.clip((affected_users * rng.uniform(0.4, 2.8, size=config.rows)).astype(int), 1, 10000)
    days_since_release = rng.integers(0, 90, size=config.rows)

    is_crash = rng.binomial(1, 0.18, size=config.rows)
    is_payment_related = np.where(module == "payments", rng.binomial(1, 0.75, size=config.rows), rng.binomial(1, 0.08, size=config.rows))
    has_workaround = rng.binomial(1, 0.45, size=config.rows)
    sla_breach_risk = rng.binomial(1, 0.2, size=config.rows)

    score = (
        np.where(environment == "prod", 2.0, np.where(environment == "staging", 0.5, -1.2))
        + np.where(module == "payments", 1.6, 0.0)
        + np.where(module == "auth", 1.0, 0.0)
        + np.where(customer_tier == "enterprise", 1.5, np.where(customer_tier == "pro", 0.5, 0.0))
        + 1.9 * is_crash
        + 1.7 * is_payment_related
        - 1.4 * has_workaround
        + 1.5 * sla_breach_risk
        + 0.55 * np.log1p(affected_users)
        + 0.38 * np.log1p(frequency_last_24h)
        - 0.025 * days_since_release
    )

    # Interactions make the dataset less rule-like and more realistic for ML.
    score += np.where((environment == "prod") & (customer_tier == "enterprise"), 1.0, 0.0)
    score += np.where((module == "payments") & (has_workaround == 0), 1.1, 0.0)
    score += np.where((environment == "dev") & (has_workaround == 1), -1.0, 0.0)
    score += np.where((is_crash == 1) & (affected_users > 200), 1.0, 0.0)
    score += np.where((days_since_release <= 3) & (environment == "prod"), 0.8, 0.0)

    noise = rng.normal(0, 1.0, size=config.rows)
    probability = _sigmoid(score + noise - 5.2)
    high_priority = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "environment": environment,
            "module": module,
            "affected_users_count": affected_users,
            "frequency_last_24h": frequency_last_24h,
            "is_crash": is_crash,
            "is_payment_related": is_payment_related,
            "has_workaround": has_workaround,
            "customer_tier": customer_tier,
            "days_since_release": days_since_release,
            "sla_breach_risk": sla_breach_risk,
            "high_priority": high_priority,
        }
    )
    return df


def save_dataset(output_path: str | Path, config: GenerationConfig = GenerationConfig()) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset(config)
    df.to_csv(output_path, index=False)
    return output_path
