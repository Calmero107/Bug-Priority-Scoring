from pathlib import Path

import pandas as pd
import streamlit as st

from src.bug_priority.data_generation import GenerationConfig, save_dataset
from src.bug_priority.model import load_model, predict_bug, rank_bugs, save_model, train_model

DATA_PATH = Path("data") / "bugs.csv"
MODEL_PATH = Path("models") / "bug_priority_pipeline.joblib"
REQUIRED_BATCH_COLUMNS = [
    "environment",
    "module",
    "customer_tier",
    "affected_users_count",
    "frequency_last_24h",
    "days_since_release",
    "is_crash",
    "is_payment_related",
    "has_workaround",
    "sla_breach_risk",
]

st.set_page_config(page_title="Bug Priority Scoring", page_icon="🐞", layout="wide")


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

page = st.sidebar.radio(
    "Choose demo page",
    ["Single Bug Demo", "Batch Ranking Demo"],
)

if page == "Single Bug Demo":
    st.title("🐞 Bug Priority Scoring Demo")
    st.caption("Predict whether a single bug should be treated as high priority.")

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

else:
    st.title("📊 Batch Ranking Demo")
    st.caption("Upload multiple bugs, score them all, and rank them from highest to lowest priority.")

    st.write("### Expected CSV columns")
    st.code(", ".join(REQUIRED_BATCH_COLUMNS), language="text")

    sample_df = pd.read_csv(DATA_PATH).drop(columns=["high_priority"]).head(15)
    sample_csv = sample_df.to_csv(index=False)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="Download sample CSV",
            data=sample_csv,
            file_name="bug_batch_sample.csv",
            mime="text/csv",
        )
    with col2:
        st.caption("Tip: include an optional `bug_id` column to make the ranked output easier to read.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        missing_columns = [col for col in REQUIRED_BATCH_COLUMNS if col not in batch_df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            ranked_df = rank_bugs(model, batch_df)
            ranked_df["high_priority_probability"] = ranked_df["high_priority_probability"].round(4)

            st.write("### Ranked Results")
            st.dataframe(ranked_df, use_container_width=True)

            high_count = int((ranked_df["predicted_priority"] == "High Priority").sum())
            st.metric("Predicted High Priority Bugs", high_count)

            output_csv = ranked_df.to_csv(index=False)
            st.download_button(
                label="Download ranked results",
                data=output_csv,
                file_name="bug_batch_ranked.csv",
                mime="text/csv",
            )
    else:
        st.write("### Preview sample batch input")
        st.dataframe(sample_df, use_container_width=True)
