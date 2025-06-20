import speech_recognition as sr

ASR_LANGUAGE_CODE = "wal-ET"  # Adjust to 'am-ET' or 'wal' if needed

def transcribe_audio(audio_path: str, language: str = ASR_LANGUAGE_CODE) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return "[Unrecognized speech]"
    except sr.RequestError as e:
        return f"[Speech error: {e}]"
