from transformers import MarianMTModel, MarianTokenizer
import torch

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-am"
tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME)
model.eval()

def translate_text(text: str) -> str:
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        generated_ids = model.generate(**batch)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
