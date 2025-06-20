from transformers import MarianMTModel, MarianTokenizer

# Use Amharic-English as a proxy for Wolaytta for now
MODEL_NAME = "Helsinki-NLP/opus-mt-en-am"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def translate(text: str, source_lang="en", target_lang="am") -> str:
    if not text:
        return ""

    # Tokenize input text
    tokens = tokenizer(text, return_tensors="pt", padding=True)

    # Generate translation
    translated_tokens = model.generate(**tokens)

    # Decode to human-readable string
    result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return result
