from app.translation.translation import translate_text
from app.voice.voice_input import transcribe_audio
from app.voice.voice_output import speak_text


def voice_translate_pipeline(audio_path: str) -> None:
    print("[PIPELINE] Running voice → text → translation → speech")
    input_text = transcribe_audio(audio_path)
    if not input_text or input_text.startswith("["):
        print(f"[PIPELINE] Error or no speech detected: {input_text}")
        return
    translated_text = translate_text(input_text)
    speak_text(translated_text)
