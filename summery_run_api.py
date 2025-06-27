from fastapi import FastAPI
from app.routes import symmerizer_api
import uvicorn

app = FastAPI(
    title="Wolaytta Chat Assistant API",
    description="Summarization module API for Wolaytta-English multilingual assistant",
    version="1.0.0"
)

app.include_router(symmerizer_api.router, prefix="/api", tags=["Summarization"])

@app.get("/")
def read_root():
    return {"message": "✅ Wolaytta Chat Assistant API is running."}

if __name__ == "__main__":
    uvicorn.run("run_api:app", host="127.0.0.1", port=8000, reload=True)
