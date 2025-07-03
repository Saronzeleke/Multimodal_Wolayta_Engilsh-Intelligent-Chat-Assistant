import torch
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import fitz
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware
import logging
import pickle
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.translation.translation import translate_text 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
PDF_PATH = r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\kuye.pdf"
EMBEDDING_CACHE_PATH = r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\embeddings.pkl"
FAISS_INDEX_PATH = r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\faiss_index.idx"
QA_HISTORY_PATH = r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\qa_history.pkl"
QA_LOG_PATH = r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\qa_logs.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QA-RAG")
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text
def chunk_text(text, chunk_size=500):
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks
logger.info("📥 Reading PDF and preparing chunks...")
raw_text = extract_text_from_pdf(PDF_PATH)
docs = chunk_text(raw_text)
logger.info(f"📚 Loaded and chunked {len(docs)} segments from PDF")
logger.info("🔍 Generating embeddings and preparing FAISS index...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

if os.path.exists(EMBEDDING_CACHE_PATH) and os.path.exists(FAISS_INDEX_PATH):
    with open(EMBEDDING_CACHE_PATH, "rb") as file:
        embeddings = pickle.load(file)
    index = faiss.read_index(FAISS_INDEX_PATH)
    logger.info("✅ Loaded cached embeddings and FAISS index.")
else:
    embeddings = embedder.encode(docs, convert_to_tensor=True).cpu().numpy()
    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, "wb") as file:
        pickle.dump(embeddings, file)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info(f"✅ FAISS index built and cached with {len(docs)} chunks")
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
if os.path.exists(QA_HISTORY_PATH):
    with open(QA_HISTORY_PATH, "rb") as file:
        qa_history = pickle.load(file)
else:
    qa_history = []

def retrieve_chunks(question, k=TOP_K):
    query_embed = embedder.encode(question, convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_embed.reshape(1, -1), k)
    return [docs[i] for i in I[0] if i < len(docs)]

def generate_answer(question, lang="en"):
    if not question.strip():
        return "Please enter a valid question."

    if lang != "en":
        question_en = translate_text(question, source_lang=lang, target_lang="en")
    else:
        question_en = question

    chunks = retrieve_chunks(question_en)
    if not chunks:
        return "Sorry, I couldn't find relevant context to answer this question."

    context = "\n\n".join(chunks)
    prompt = f"""
    You are a multilingual assistant for Wolaytta and English users.
    Use the following CONTEXT to answer the QUESTION below. Answer carefully and concisely.don't hallucinate.

    CONTEXT:
    {context}

    QUESTION:
    {question_en}

    ANSWER:
    """
    try:
        response = client.chat.completions.create(
            model="baidu/ernie-4.5-300b-a47b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        answer_en = response.choices[0].message.content.strip()
        answer = translate_text(answer_en, source_lang="en", target_lang=lang) if lang != "en" else answer_en

        os.makedirs(os.path.dirname(QA_LOG_PATH), exist_ok=True)
        with open(QA_LOG_PATH, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([question, answer])

        qa_history.append({"question": question, "answer": answer, "lang": lang})
        with open(QA_HISTORY_PATH, "wb") as f:
            pickle.dump(qa_history, f)

        return answer
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        return "❌ Failed to generate answer from OpenRouter."

class QuestionRequest(BaseModel):
    question: str
    lang: str = "en"

router = APIRouter()

@router.post("/qa")
async def qa_endpoint(payload: QuestionRequest):
    try:
        answer = generate_answer(payload.question, lang=payload.lang)
        return {"question": payload.question, "answer": answer, "lang": payload.lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

app = FastAPI(title="QA RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api", tags=["QA"])

@app.get("/")
def root():
    return {"message": "✅ QA RAG API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qa:app", host="127.0.0.1", port=8000, reload=True)
