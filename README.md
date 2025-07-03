# Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant
# 🌍 Multimodal Wolaytta ↔ English Intelligent Assistant

A multilingual AI assistant that supports:
- 🔄 Bidirectional translation between Wolaytta and English
- ❓ Question Answering using Retrieval-Augmented Generation (RAG)
- 📝 Summarization and sentiment analysis
- 🎤 Voice input/output
- 📊 BLEU evaluation dashboard & feedback loop

## 🚀 Features
- Translation powered by fine-tuned `Sakuzas/t5-wolaytta-english`
- RAG pipeline over `kuye.pdf` using Sentence Transformers + FAISS
- Streamlit dashboard to monitor BLEU over time
- Real-time user feedback loop

## 🗂️ Project Structure

app/
├── QA/ → RAG + BLEU evaluation
├── Sentiment/ → LSTM-based sentiment classifier
├── Summarize/ → T5 summarizer fine-tuning + inference
├── translation/ → Fine-tuned T5 pipeline + API
├── routes/ → FastAPI endpoints
ui/ → Streamlit dashboard
data/ → PDF, CSVs, embeddings, logs
model_outputs/ → Evaluation results
main.py → FastAPI entrypoint

## 🔧 How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
Streamlit BLEU dashboard:
streamlit run ui/dashboard.py
🧪 Evaluation
BLEU, ROUGE, METEOR, ChrF metrics

Feedback logging

Model comparison in model_comparison.py

🧠 Models Used
Translation: Sakuzas/t5-wolaytta-english

RAG: mistralai/mixtral-8x7b

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2




