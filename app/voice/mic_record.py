import pyaudio
import wave

def record_audio(output_file: str = "assets/recorded.wav", duration: int = 5, rate: int = 16000) -> None:
    chunk = 1024 
    format = pyaudio.paInt16  
    channels = 1  
    audio = pyaudio.PyAudio()

    print(f"[RECORDING] Recording {duration} seconds of audio...")
    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)

    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("[RECORDING] Done recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print(f"[RECORDING] Saved to {output_file}")
