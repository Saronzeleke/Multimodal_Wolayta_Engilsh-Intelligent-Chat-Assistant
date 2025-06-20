from gtts import gTTS
import os

TTS_LANGUAGE_CODE = "en"  # Adjust to 'en' for English

def speak_text(text: str, lang: str = TTS_LANGUAGE_CODE, out_path: str = "tts_output.mp3") -> None:
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    os.system(f"start {out_path}" if os.name == "nt" else f"xdg-open {out_path}")
