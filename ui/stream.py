import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Wolaytta Multimodal Assistant", layout="wide")
st.title("🌍 Wolaytta ↔ English Multimodal Intelligent Assistant")

# === Sidebar Navigation ===
menu = st.sidebar.radio("Select Module", ["QA (Question Answering)", "Translation", "Summarization", "Sentiment Analysis", "Voice Translation", "Model Comparison Dashboard", "BLEU Dashboard"])

API_URLS = {
    "qa": "http://127.0.0.1:8000/api/qa",
    "translate": "http://127.0.0.1:8000/api/translate",
    "summarize": "http://127.0.0.1:8000/api/summarize",
    "sentiment": "http://127.0.0.1:8000/api/sentiment",
    "voice_pipeline": "http://127.0.0.1:8000/api/voice",
}

# === 1. QA ===
if menu == "QA (Question Answering)":
    st.subheader("💬 Ask Any Question")
    question = st.text_area("Enter your question")
    lang = st.selectbox("Select language", ["en", "wo"])

    if st.button("Get Answer"):
        with st.spinner("Answering..."):
            res = requests.post(API_URLS["qa"], json={"question": question, "lang": lang})
            if res.ok:
                st.success(res.json()["answer"])
            else:
                st.error("❌ Failed to get answer")

# === 2. Translation ===
elif menu == "Translation":
    st.subheader("🌐 Translate between English and Wolaytta")
    text = st.text_area("Enter text to translate")
    source_lang = st.selectbox("Source Language", ["en", "wo"])
    target_lang = st.selectbox("Target Language", ["wo", "en"])

    if st.button("Translate"):
        res = requests.post(API_URLS["translate"], json={"text": text, "source_lang": source_lang, "target_lang": target_lang})
        if res.ok:
            st.success(res.json()["translated_text"])
        else:
            st.error("Translation failed")

# === 3. Summarization ===
elif menu == "Summarization":
    st.subheader("📝 Summarize Wolaytta/English Text")
    text = st.text_area("Enter text to summarize")
    if st.button("Summarize"):
        res = requests.post(API_URLS["summarize"], json={"text": text})
        if res.ok:
            st.success(res.json()["summary"])
        else:
            st.error("Summarization failed")

# === 4. Sentiment ===
elif menu == "Sentiment Analysis":
    st.subheader("🧠 Analyze Sentiment")
    sentence = st.text_input("Enter a sentence")
    if st.button("Analyze"):
        res = requests.post(API_URLS["sentiment"], json={"sentence": sentence})
        if res.ok:
            st.success(f"Sentiment: {res.json()['sentiment']}")
        else:
            st.error("Sentiment analysis failed")

# === 5. Voice Translation ===
elif menu == "Voice Translation":
    st.subheader("🎙️ Upload Audio for Voice Translation")
    uploaded = st.file_uploader("Upload audio file (WAV/MP3)", type=["wav", "mp3"])
    if uploaded and st.button("Translate Audio"):
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded.read())
        res = requests.post(API_URLS["voice_pipeline"], files={"file": open("temp_audio.wav", "rb")})
        if res.ok:
            st.success("✅ Voice pipeline executed.")
        else:
            st.error("Voice translation failed")

# === 6. BLEU Dashboard ===
elif menu == "BLEU Dashboard":
    st.subheader("📊 BLEU Score Evaluation Dashboard")
    path = "app/QA/bleu_evaluation_log.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.line_chart(df.set_index("timestamp")["average_bleu"])
        st.dataframe(df.tail(5))
    else:
        st.warning("No BLEU evaluations found")

# === 7. Model Dashboard ===
elif menu == "Model Comparison Dashboard":
    st.subheader("📈 Compare Models with BLEU, ROUGE, METEOR, ChrF")
    report_path = "app/QA/reports/model_comparison_report.csv"
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        st.dataframe(df)
        st.bar_chart(df.set_index("model_name"))
    else:
        st.warning("No model comparison report found. Run model_comparison.py.")
