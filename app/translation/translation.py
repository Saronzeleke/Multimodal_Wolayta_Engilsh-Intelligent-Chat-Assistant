from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import Optional

def get_model_name(source_lang, target_lang):
    if source_lang == "en" and target_lang == "am":
        return "Helsinki-NLP/opus-mt-en-am"
    elif source_lang == "am" and target_lang == "en":
        return "Helsinki-NLP/opus-mt-am-en"
    # You can extend this logic for Wolaytta or return None
    return None

def translate(text: str, source_lang="en", target_lang="am") -> Optional[str]:
    if not text.strip():
        return ""

    model_name = get_model_name(source_lang, target_lang)
    if model_name is None:
        return f"[No translation model available for {source_lang} → {target_lang}]"

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()

        batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            generated_ids = model.generate(**batch, max_length=512, num_beams=4, early_stopping=True)
        translated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return translated
    except Exception as e:
        return f"[Translation error: {e}]"
