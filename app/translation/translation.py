from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import Optional

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-am"

try:
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load translation model: {e}")

def translate_text(text: str) -> Optional[str]:
    if not text.strip():
        return ""
    try:
        batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            generated_ids = model.generate(**batch, max_length=512, num_beams=4, early_stopping=True)
        translated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return translated
    except Exception as e:
        return f"[Translation error: {e}]"
