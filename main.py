from app.translation.translation import translate
from app.voice.pipline import voice_translate_pipeline
from app.voice.mic_record import record_audio
source_text = "How are you today?"
translated = translate(source_text, source_lang="en", target_lang="am")

print(f"Original: {source_text}")
print(f"Translated: {translated}")

if __name__ == "__main__":
    import os
    input_audio = "assets/sample_wolaytta.wav"
    if not os.path.exists(input_audio):
        print(f"[ERROR] File not found: {input_audio}")
    else:
        voice_translate_pipeline(input_audio)
        from app.voice.pipeline import voice_translate_pipeline


if __name__ == "__main__":
    import os

    # Step 1: Record audio via mic
    record_audio("assets/recorded.wav", duration=5)

    # Step 2: Run pipeline on recorded audio
    input_audio = "assets/recorded.wav"
    if not os.path.exists(input_audio):
        print(f"[ERROR] File not found: {input_audio}")
    else:
        voice_translate_pipeline(input_audio)
