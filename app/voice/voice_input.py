import speech_recognition as sr
import os
from typing import Optional

ASR_LANGUAGE_CODE = "en-US"  # For Wolaytta, try 'am-ET' or a custom code when model is trained


def transcribe_audio(audio_path: str, language: str = ASR_LANGUAGE_CODE) -> Optional[str]:
    if not os.path.exists(audio_path):
        return f"[File not found: {audio_path}]"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        result = recognizer.recognize_google(audio_data, language=language)
        return result
    except sr.UnknownValueError:
        return "[Unrecognized speech]"
    except sr.RequestError as e:
        return f"[ASR service error: {e}]"
    except Exception as e:
        return f"[ASR unknown error: {e}]"
