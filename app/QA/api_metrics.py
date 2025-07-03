# === FILE: app/QA/api_metrics.py ===

import os
import csv
from fastapi import APIRouter, HTTPException
from datetime import datetime
import pandas as pd
from eval_bleu import evaluate_bleu
from pydantic import BaseModel
from typing import List

BLEU_LOG_PATH = "app/QA/bleu_evaluation_log.csv"

router = APIRouter()


@router.get("/metrics/bleu")
def get_latest_bleu():
    if not os.path.exists(BLEU_LOG_PATH):
        raise HTTPException(status_code=404, detail="BLEU log not found.")
    df = pd.read_csv(BLEU_LOG_PATH)
    if df.empty:
        raise HTTPException(status_code=400, detail="BLEU log is empty.")
    latest = df.sort_values("timestamp").iloc[-1]
    return latest.to_dict()


@router.get("/metrics/history")
def get_bleu_history():
    if not os.path.exists(BLEU_LOG_PATH):
        raise HTTPException(status_code=404, detail="BLEU log not found.")
    df = pd.read_csv(BLEU_LOG_PATH)
    return df.to_dict(orient="records")


@router.post("/evaluate-now")
def trigger_bleu_evaluation():
    score = evaluate_bleu()
    if score is None:
        raise HTTPException(status_code=500, detail="Evaluation failed or no data.")
    return {"status": "success", "average_bleu": score}


# === FILE: app/main.py ===

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.QA.api_metrics import router as metrics_router
from app.QA.qa import router as qa_router

app = FastAPI(title="Multimodal Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router, prefix="/api", tags=["QA"])
app.include_router(metrics_router, prefix="/api", tags=["Metrics"])

@app.get("/")
def root():
    return {"message": "✅ API is live"}


# === FILE: ui/bleu_dashboard_live.py ===

import streamlit as st
import pandas as pd
import altair as alt
import requests

API_BASE = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="BLEU Dashboard", layout="wide")
st.title("📊 Live BLEU Score Evaluation Dashboard")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔁 Evaluate Now"):
        try:
            res = requests.post(f"{API_BASE}/evaluate-now")
            st.success(f"Re-evaluated! Avg BLEU: {res.json()['average_bleu']:.4f}")
        except:
            st.error("❌ Failed to trigger evaluation")

    st.markdown("---")
    st.subheader("📋 Latest BLEU Result")
    try:
        latest = requests.get(f"{API_BASE}/metrics/bleu").json()
        st.json(latest)
    except:
        st.warning("⚠️ Could not load latest BLEU score")

with col2:
    st.subheader("📈 BLEU Score Over Time")
    try:
        history = requests.get(f"{API_BASE}/metrics/history").json()
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        chart = alt.Chart(df).mark_line(point=True).encode(
            x="timestamp:T",
            y=alt.Y("average_bleu:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["timestamp:T", "num_samples", "average_bleu"]
        ).properties(width=800, height=400)
        st.altair_chart(chart, use_container_width=True)
    except:
        st.warning("⚠️ Could not load BLEU history")
