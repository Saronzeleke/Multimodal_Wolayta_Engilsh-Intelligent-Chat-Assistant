
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.translation.translation import translate
from app.voice.pipline import voice_translate_pipeline
from app.voice.mic_record import record_audio
from app.QA.qa import router as qa
from app.translation.translation_api import router as translation_router
from app.routes.summarizer_api import router as summarizer_router
# You can import others like sentiment or voice here too
import os
source_text = "How are you today?"
translated = translate(source_text, source_lang="en", target_lang="am")

print(f"Original: {source_text}")
print(f"Translated: {translated}")
app = FastAPI(title="Multimodal Wolaytta ↔ English Intelligent Assistant")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(qa, prefix="/api/qa", tags=["Question Answering"])
app.include_router(translation_router, prefix="/api/translate", tags=["Translation"])
app.include_router(summarizer_router, prefix="/api/summarize", tags=["Summarization"])
# Future: app.include_router(voice_router, prefix="/api/voice", tags=["Voice Input/Output"])

@app.get("/")
def root():
    return {
        "message": "✅ Multimodal Wolaytta ↔ English Assistant is running",
        "available_endpoints": [
            "/api/qa",
            "/api/translate",
            "/api/summarize"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    record_audio("assets/recorded.wav", duration=5)
    input_audio = "assets/recorded.wav"
    if not os.path.exists(input_audio):
        print(f"[ERROR] File not found: {input_audio}")
    else:
        voice_translate_pipeline(input_audio)