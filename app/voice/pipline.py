from app.translation.translation import translate_text
from app.voice.voice_input import transcribe_audio
from app.voice.voice_output import speak_text

def voice_translate_pipeline(audio_path: str) -> None:
    print("[PIPELINE] Step 1: Transcribing input audio...")
    input_text = transcribe_audio(audio_path)
    print(f"[INPUT TEXT] {input_text}")

    if not input_text or input_text.startswith("["):
        print(f"[PIPELINE] Aborting due to ASR error.")
        return

    print("[PIPELINE] Step 2: Translating text...")
    translated_text = translate_text(input_text)
    print(f"[TRANSLATED TEXT] {translated_text}")

    print("[PIPELINE] Step 3: Generating speech from translated text...")
    speak_text(translated_text)
