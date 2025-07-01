import os
import csv
from datetime import datetime
from typing import Optional

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

LOG_FILE_PATH = os.getenv("TRANSLATION_LOG_PATH", "data/translation_logs.csv")
MODEL_NAME = "Sakuzas/t5-wolaytta-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
translator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "unknown"

def translate_text(text: str, source_lang: Optional[str] = None, target_lang: Optional[str] = None) -> str:
    """
    Detects language if not provided, then translates.
    Supports auto language detection.
    Logs metadata for every translation.
    """
    if not text.strip():
        return ""
    if source_lang is None or source_lang == "auto":
        source_lang = detect_language(text)
    if target_lang is None:
        target_lang = "wolaytta" if source_lang == "en" else "en"

    # Prepare input text for your fine-tuned model
    # Assume model expects format like: "translate {source_lang} to {target_lang}: {text}"
    input_text = f"translate {source_lang} to {target_lang}: {text}"

    try:
        result = translator(input_text, max_length=512, num_beams=5, early_stopping=True)
        translated_text = result[0]['generated_text']

        log_translation(text, translated_text, source_lang, target_lang, success=True)

        return translated_text
    except Exception as e:
        log_translation(text, str(e), source_lang, target_lang, success=False)
        return f"[Translation failed: {str(e)}]"

def log_translation(
    source_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str,
    success: bool
):
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    header = ["timestamp", "source_text", "translated_text", "source_lang", "target_lang", "success"]
    file_exists = os.path.exists(LOG_FILE_PATH)

    with open(LOG_FILE_PATH, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([datetime.utcnow().isoformat(), source_text, translated_text, source_lang, target_lang, success])
