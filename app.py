from pathlib import Path

import pandas as pd
import streamlit as st

from src.bug_priority.data_generation import GenerationConfig, save_dataset
from src.bug_priority.model import load_model, predict_bug, train_model, save_model

DATA_PATH = Path("data") / "bugs.csv"
MODEL_PATH = Path("models") / "bug_priority_pipeline.joblib"

st.set_page_config(page_title="Bug Priority Scoring", page_icon="🐞", layout="centered")
st.title("🐞 Bug Priority Scoring Demo")
st.caption("Predict whether a bug should be treated as high priority.")


@st.cache_resource
def get_model():
    if not DATA_PATH.exists():
        save_dataset(DATA_PATH, GenerationConfig(rows=1500))

    if not MODEL_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        result = train_model(df)
        save_model(result["pipeline"], MODEL_PATH)

    return load_model(MODEL_PATH)


model = get_model()

with st.form("bug_form"):
    environment = st.selectbox("Environment", ["prod", "staging", "dev"])
    module = st.selectbox(
        "Module",
        ["auth", "payments", "search", "ui", "reporting", "notifications", "integrations"],
    )
    customer_tier = st.selectbox("Customer Tier", ["free", "pro", "enterprise"])

    affected_users_count = st.number_input("Affected Users Count", min_value=1, value=120, step=1)
    frequency_last_24h = st.number_input("Frequency in Last 24h", min_value=1, value=350, step=1)
    days_since_release = st.number_input("Days Since Release", min_value=0, value=3, step=1)

    is_crash = st.checkbox("Crash / Service Unavailable", value=True)
    is_payment_related = st.checkbox("Payment Related", value=False)
    has_workaround = st.checkbox("Workaround Available", value=False)
    sla_breach_risk = st.checkbox("SLA Breach Risk", value=False)

    submitted = st.form_submit_button("Predict")

if submitted:
    bug_input = pd.DataFrame(
        [
            {
                "environment": environment,
                "module": module,
                "customer_tier": customer_tier,
                "affected_users_count": int(affected_users_count),
                "frequency_last_24h": int(frequency_last_24h),
                "days_since_release": int(days_since_release),
                "is_crash": int(is_crash),
                "is_payment_related": int(is_payment_related),
                "has_workaround": int(has_workaround),
                "sla_breach_risk": int(sla_breach_risk),
            }
        ]
    )

    result = predict_bug(model, bug_input)
    probability = result["high_priority_probability"]

    if result["prediction"] == 1:
        st.error(f"Prediction: {result['label']}")
    else:
        st.success(f"Prediction: {result['label']}")

    st.metric("High Priority Probability", f"{probability:.1%}")
    st.write("### Input Summary")
    st.dataframe(bug_input, use_container_width=True)

st.write("---")
st.write("### How this demo works")
st.write(
    "The app uses a logistic regression model trained on a synthetic bug dataset. "
    "It scores a bug based on impact, environment, payment sensitivity, crash behavior, workaround availability, and customer importance."
)
