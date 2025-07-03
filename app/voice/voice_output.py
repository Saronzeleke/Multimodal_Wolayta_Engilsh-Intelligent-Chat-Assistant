from gtts import gTTS
import os

def speak_text(text: str, lang: str = "wol", out_path: str = "tts_output.mp3") -> None:
    if not text.strip():
        print("[TTS] Empty input. Skipping synthesis.")
        return
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(out_path)
        print(f"[TTS] Saved to {out_path}")
        os.system(f"start {out_path}" if os.name == "nt" else f"xdg-open {out_path}")
    except Exception as e:
        print(f"[TTS Error] {e}")
