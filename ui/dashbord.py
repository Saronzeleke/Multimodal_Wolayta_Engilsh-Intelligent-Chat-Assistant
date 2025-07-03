import os
import ui.dashbord as st
import pandas as pd
import altair as alt

# === Config ===
BLEU_LOG_PATH = "app/QA/bleu_evaluation_log.csv"
FEEDBACK_LOG_PATH = "app/QA/feedback_logs.csv"
MODEL_COMPARISON_CSV = "app/QA/reports/model_comparison_report.csv"

st.set_page_config(page_title="Multilingual Assistant Monitoring Dashboard", layout="wide")
st.title("📊 Multilingual Assistant Monitoring Dashboard")

# === BLEU Score Trend ===
st.header("📈 BLEU Score Over Time")
if os.path.exists(BLEU_LOG_PATH):
    bleu_df = pd.read_csv(BLEU_LOG_PATH)
    bleu_df['timestamp'] = pd.to_datetime(bleu_df['timestamp'])
    bleu_df = bleu_df.sort_values("timestamp")

    bleu_chart = alt.Chart(bleu_df).mark_line(point=True).encode(
        x="timestamp:T",
        y=alt.Y("average_bleu:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["timestamp:T", "num_samples", "average_bleu"]
    ).properties(width=700, height=300)
    st.altair_chart(bleu_chart, use_container_width=True)
    st.write("Most recent BLEU score:", bleu_df.tail(1))
else:
    st.warning("No BLEU evaluation log found.")

# Feedback Summary
st.header("🗣️ User Feedback Summary")
if os.path.exists(FEEDBACK_LOG_PATH):
    feedback_df = pd.read_csv(FEEDBACK_LOG_PATH, names=["timestamp", "question", "answer", "feedback", "comment"])
    feedback_counts = feedback_df["feedback"].value_counts().reset_index()
    feedback_counts.columns = ["Feedback", "Count"]

    feedback_pie = alt.Chart(feedback_counts).mark_arc().encode(
        theta="Count:Q",
        color="Feedback:N",
        tooltip=["Feedback:N", "Count:Q"]
    ).properties(width=350, height=300)

    st.altair_chart(feedback_pie, use_container_width=False)
    st.dataframe(feedback_df.tail(5), use_container_width=True)
else:
    st.warning("No feedback log found.")

# === Model Comparison (Optional) ===
st.header("🤖 Model Comparison Metrics")
if os.path.exists(MODEL_COMPARISON_CSV):
    comparison_df = pd.read_csv(MODEL_COMPARISON_CSV)
    comparison_melted = comparison_df.melt(id_vars="model_name", var_name="Metric", value_name="Score")

    model_bar = alt.Chart(comparison_melted).mark_bar().encode(
        x=alt.X("model_name:N", title="Model"),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color="Metric:N",
        column="Metric:N",
        tooltip=["model_name:N", "Metric:N", "Score:Q"]
    ).properties(height=300).configure_axis(labelAngle=0)

    st.altair_chart(model_bar, use_container_width=True)
    st.dataframe(comparison_df)
else:
    st.info("Model comparison report not found. Run `model_comparison.py` to generate it.")

# === Download Section ===
st.sidebar.header("⬇️ Download Logs")
if os.path.exists(BLEU_LOG_PATH):
    st.sidebar.download_button("Download BLEU Log", data=bleu_df.to_csv(index=False), file_name="bleu_log.csv")
if os.path.exists(FEEDBACK_LOG_PATH):
    st.sidebar.download_button("Download Feedback Log", data=feedback_df.to_csv(index=False), file_name="feedback_log.csv")
if os.path.exists(MODEL_COMPARISON_CSV):
    st.sidebar.download_button("Download Comparison Report", data=comparison_df.to_csv(index=False), file_name="model_comparison.csv")
