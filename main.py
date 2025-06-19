from app.translation import translate

# Try translating from English → Amharic as a placeholder
source_text = "How are you today?"
translated = translate(source_text, source_lang="en", target_lang="am")

print(f"Original: {source_text}")
print(f"Translated: {translated}")
