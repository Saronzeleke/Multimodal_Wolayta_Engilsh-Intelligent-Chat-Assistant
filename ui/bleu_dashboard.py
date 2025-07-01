# === Final Step: Optional Streamlit BLEU Score Visualizer ===
# File: ui/bleu_dashboard.py

import streamlit as st
import pandas as pd
import altair as alt
import os

BLEU_LOG_PATH = "app/QA/bleu_evaluation_log.csv"

st.title("📊 BLEU Score Evaluation Dashboard")

if os.path.exists(BLEU_LOG_PATH):
    df = pd.read_csv(BLEU_LOG_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")

    st.subheader("📈 BLEU Score Over Time")
    chart = alt.Chart(df).mark_line(point=True).encode(
        x="timestamp:T",
        y=alt.Y("average_bleu:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["timestamp:T", "num_samples", "average_bleu"]
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("📋 Latest Evaluation")
    st.write(df.tail(1))

    st.download_button("Download BLEU Log CSV", df.to_csv(index=False), "bleu_evaluation_log.csv")
else:
    st.warning("No BLEU evaluation log found. Run `eval_bleu.py` first.")
