from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from translation import translate_text

app = FastAPI(title="Wolaytta-English Translation API")

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto" 
    target_lang: str = None    

class TranslationResponse(BaseModel):
    translated_text: str

@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    translated = translate_text(request.text, request.source_lang, request.target_lang)
    return TranslationResponse(translated_text=translated)


@app.get("/")
async def root():
    return {"message": "Wolaytta-English Translation API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("translation_api:app", host="127.0.0.1", port=8001, reload=True)
