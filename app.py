from pathlib import Path

import pandas as pd
import streamlit as st

from src.bug_priority.data_generation import GenerationConfig, save_dataset
from src.bug_priority.log_parser import add_pattern_counts, parse_log, parse_logs_to_dataframe
from src.bug_priority.model import load_model, predict_single_log, rank_logs, save_model, train_model

DATA_PATH = Path("data") / "logs.csv"
MODEL_PATH = Path("models") / "log_priority_pipeline.joblib"
SAMPLE_BATCH_PATH = Path("data") / "demo_log_batch.txt"

st.set_page_config(page_title="Log Priority Scoring", page_icon="📜", layout="wide")


@st.cache_resource
def get_model():
    if not DATA_PATH.exists():
        save_dataset(DATA_PATH, GenerationConfig(rows=1800))
    if not MODEL_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        result = train_model(df)
        save_model(result["pipeline"], MODEL_PATH)
    return load_model(MODEL_PATH)


model = get_model()
page = st.sidebar.radio("Choose demo page", ["Single Log Demo", "Batch Log Ranking Demo"])

if page == "Single Log Demo":
    st.title("📜 Single Log Priority Demo")
    st.caption("Paste one real-world-like log line, extract signals with regex/rules, then score it with ML.")

    default_log = (
        "2026-04-15T10:21:11Z ERROR payment-service Charge failed order_id=92813 "
        "user=1842 error=DB_TIMEOUT env=prod enterprise customer affected repeated spike after deploy urgent sev1"
    )
    log_input = st.text_area("Log line", value=default_log, height=180)

    if st.button("Evaluate"):
        parsed = parse_log(log_input)
        parsed_df = add_pattern_counts(pd.DataFrame([parsed]))
        result = predict_single_log(model, parsed_df)

        if result["prediction"] == 1:
            st.error(f"Prediction: {result['label']}")
        else:
            st.success(f"Prediction: {result['label']}")

        st.metric("High Priority Probability", f"{result['high_priority_probability']:.1%}")
        st.write("### Extracted Features")
        feature_view = pd.DataFrame([result["row"]]).T.reset_index()
        feature_view.columns = ["feature", "value"]
        st.dataframe(feature_view, use_container_width=True)

else:
    st.title("📊 Batch Log Ranking Demo")
    st.caption("Paste many raw log lines, score them, and rank them from highest to lowest priority.")

    if SAMPLE_BATCH_PATH.exists():
        sample_text = SAMPLE_BATCH_PATH.read_text()
    else:
        sample_text = ""

    st.download_button(
        label="Download sample log batch",
        data=sample_text,
        file_name="demo_log_batch.txt",
        mime="text/plain",
    )

    log_batch = st.text_area("Paste multiple log lines (one per line)", value=sample_text, height=320)

    if st.button("Rank Logs"):
        log_lines = [line for line in log_batch.splitlines() if line.strip()]
        if not log_lines:
            st.warning("Please paste at least one log line.")
        else:
            parsed_df = parse_logs_to_dataframe(log_lines)
            parsed_df = add_pattern_counts(parsed_df)
            ranked_df = rank_logs(model, parsed_df)
            ranked_df["high_priority_probability"] = ranked_df["high_priority_probability"].round(4)

            display_columns = [
                "raw_log",
                "predicted_priority",
                "high_priority_probability",
                "environment",
                "level",
                "module",
                "service_name",
                "error_code",
                "repeat_count_same_pattern",
            ]
            st.write("### Ranked Results")
            st.dataframe(ranked_df[display_columns], use_container_width=True)

            st.download_button(
                label="Download ranked CSV",
                data=ranked_df.to_csv(index=False),
                file_name="ranked_logs.csv",
                mime="text/csv",
            )

st.write("---")
st.write("### How this demo works")
st.write(
    "This app first parses raw logs using regex and keyword rules to extract structured signals such as environment, level, service, "
    "error type, payment/auth impact, timeout/crash patterns, and repetition count. Those extracted features are then scored by a logistic regression model."
)
