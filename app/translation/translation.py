from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging

MODEL_NAME = "Sakuzas/t5-wolaytta-english"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    raise RuntimeError(f"Failed to load translation pipeline: {e}")

def translate(text: str, source_lang: str = "en", target_lang: str = "woly") -> str:
    """
    Translate between English and Wolaytta using the fine-tuned T5 model.
    Args:
        text (str): Input sentence to translate.
        source_lang (str): 'en' for English, 'woly' for Wolaytta.
        target_lang (str): 'woly' for Wolaytta, 'en' for English.
    Returns:
        str: Translated sentence.
    """
    if not text.strip():
        return ""

    if source_lang == "woly" and target_lang == "en":
        prompt = f"translate Wolaytta to English: {text}"
    elif source_lang == "en" and target_lang == "woly":
        prompt = f"translate English to Wolaytta: {text}"
    else:
        raise ValueError(f"Unsupported translation direction: {source_lang} → {target_lang}")

    try:
        result = pipe(prompt, max_length=256, num_beams=4, early_stopping=True)
        return result[0]["generated_text"]
    except Exception as e:
        logging.error(f"[Translation error]: {e}")
        return "[Translation failed]"
