import csv
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

FEEDBACK_PATH = "data/feedback_logs.csv"

router = APIRouter()

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    user_feedback: str  # "good", "bad", or textual correction
    lang: str

@router.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    try:
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        with open(FEEDBACK_PATH, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                payload.lang,
                payload.question,
                payload.answer,
                payload.user_feedback
            ])
        return {"message": "✅ Feedback submitted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to submit feedback: {str(e)}")
