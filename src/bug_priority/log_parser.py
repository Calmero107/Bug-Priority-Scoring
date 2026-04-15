from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

ENV_PATTERN = re.compile(r"env=(prod|staging|dev)", re.IGNORECASE)
LEVEL_PATTERN = re.compile(r"\b(INFO|WARN|ERROR|FATAL)\b", re.IGNORECASE)
SERVICE_PATTERN = re.compile(r"\b([a-zA-Z0-9-]+-service)\b")
ERROR_CODE_PATTERN = re.compile(r"(?:error|reason)=([A-Z_]+)")

KEYWORD_FLAGS = {
    "is_payment_related": ["payment", "charge", "checkout", "invoice", "billing", "refund"],
    "is_auth_related": ["auth", "login", "token", "session", "oauth", "credential"],
    "is_db_error": ["db_timeout", "db_error", "database", "sql", "deadlock", "connection pool exhausted"],
    "is_timeout": ["timeout", "timed out"],
    "is_rate_limit": ["rate_limit", "429", "too many requests", "throttle"],
    "is_oom": ["outofmemory", "oom", "memoryerror"],
    "is_exception": ["exception", "traceback", "stacktrace", "panic"],
    "is_failed": ["failed", "failure", "unavailable", "rejected"],
    "has_customer_impact": ["customer", "user", "order", "checkout", "login"],
    "has_workaround": ["workaround", "retry manually", "fallback available"],
    "sla_breach_risk": ["sla", "sev1", "sev2", "urgent", "breach"],
}

SERVICE_MODULE_MAP = {
    "payment-service": "payments",
    "billing-service": "payments",
    "checkout-service": "payments",
    "auth-service": "auth",
    "identity-service": "auth",
    "search-service": "search",
    "frontend-service": "ui",
    "ui-service": "ui",
    "reporting-service": "reporting",
    "notification-service": "notifications",
    "notifications-service": "notifications",
    "integrations-service": "integrations",
}


def _contains_any(text: str, patterns: Iterable[str]) -> int:
    return int(any(pattern in text for pattern in patterns))


def _extract_level(log_text: str) -> str:
    match = LEVEL_PATTERN.search(log_text)
    return match.group(1).upper() if match else "INFO"


def _extract_environment(log_text: str) -> str:
    match = ENV_PATTERN.search(log_text)
    return match.group(1).lower() if match else "prod"


def _extract_service(log_text: str) -> str:
    match = SERVICE_PATTERN.search(log_text)
    return match.group(1).lower() if match else "unknown-service"


def _extract_module(service_name: str, normalized_text: str) -> str:
    if service_name in SERVICE_MODULE_MAP:
        return SERVICE_MODULE_MAP[service_name]

    if "payment" in normalized_text or "checkout" in normalized_text:
        return "payments"
    if "auth" in normalized_text or "login" in normalized_text or "token" in normalized_text:
        return "auth"
    if "search" in normalized_text:
        return "search"
    if "ui" in normalized_text or "frontend" in normalized_text or "button" in normalized_text:
        return "ui"
    if "report" in normalized_text or "export" in normalized_text:
        return "reporting"
    if "notify" in normalized_text or "email" in normalized_text or "smtp" in normalized_text:
        return "notifications"
    if "integration" in normalized_text or "webhook" in normalized_text or "partner" in normalized_text:
        return "integrations"
    return "unknown"


def _extract_error_code(log_text: str) -> str:
    match = ERROR_CODE_PATTERN.search(log_text)
    return match.group(1).upper() if match else "UNKNOWN"


def _estimate_affected_users(normalized_text: str, level: str, environment: str) -> int:
    count = 5
    if "enterprise" in normalized_text or "vip" in normalized_text:
        count += 150
    if "all users" in normalized_text or "all customers" in normalized_text:
        count += 500
    if "order" in normalized_text or "checkout" in normalized_text:
        count += 120
    if "login" in normalized_text:
        count += 80
    if level == "ERROR":
        count += 50
    if level == "FATAL":
        count += 120
    if environment == "prod":
        count += 100
    if environment == "staging":
        count += 20
    return count


def _estimate_frequency(normalized_text: str, level: str) -> int:
    frequency = 1
    if "repeated" in normalized_text or "spike" in normalized_text:
        frequency += 50
    if "every request" in normalized_text or "flood" in normalized_text:
        frequency += 100
    if "timeout" in normalized_text:
        frequency += 20
    if "failed" in normalized_text:
        frequency += 15
    if level == "ERROR":
        frequency += 10
    if level == "FATAL":
        frequency += 30
    return frequency


def _estimate_days_since_release(normalized_text: str) -> int:
    if "new release" in normalized_text or "post-deploy" in normalized_text or "after deploy" in normalized_text:
        return 1
    if "regression" in normalized_text:
        return 3
    return 21


def parse_log(log_text: str) -> dict:
    normalized_text = log_text.lower().strip()
    level = _extract_level(log_text)
    environment = _extract_environment(log_text)
    service_name = _extract_service(log_text)
    module = _extract_module(service_name, normalized_text)
    error_code = _extract_error_code(log_text)

    features = {
        "raw_log": log_text.strip(),
        "environment": environment,
        "level": level,
        "service_name": service_name,
        "module": module,
        "error_code": error_code,
        "affected_users_count": _estimate_affected_users(normalized_text, level, environment),
        "frequency_last_24h": _estimate_frequency(normalized_text, level),
        "days_since_release": _estimate_days_since_release(normalized_text),
    }

    for feature_name, patterns in KEYWORD_FLAGS.items():
        features[feature_name] = _contains_any(normalized_text, patterns)

    features["is_crash"] = int(level == "FATAL" or "crash" in normalized_text or "panic" in normalized_text)
    features["customer_tier"] = (
        "enterprise" if "enterprise" in normalized_text or "vip" in normalized_text else "pro" if "pro" in normalized_text else "free"
    )
    return features


def parse_logs_to_dataframe(log_lines: list[str]) -> pd.DataFrame:
    rows = [parse_log(line) for line in log_lines if line.strip()]
    return pd.DataFrame(rows)


def add_pattern_counts(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    pattern_counts = Counter(zip(enriched["service_name"], enriched["error_code"], enriched["environment"]))
    enriched["repeat_count_same_pattern"] = [
        pattern_counts[(row.service_name, row.error_code, row.environment)] for row in enriched.itertuples()
    ]
    return enriched
