import csv
import os
from datetime import datetime
from typing import Literal
import logging
FEEDBACK_LOG_PATH = "app/QA/feedback_logs.csv"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeedbackLogger")
os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
def log_feedback(
    question: str,
    answer: str,
    user_feedback: Literal["good", "bad", "neutral"],
    comment: str = "",
    user_id: str = "anonymous"
):
    """
    Log user feedback about a QA response.

    Args:
        question (str): The original user question.
        answer (str): The assistant's response.
        user_feedback (Literal): Feedback label - good, bad, or neutral.
        comment (str): Optional additional comment.
        user_id (str): ID or name of the user (optional).
    """
    timestamp = datetime.now().isoformat()

    # Check and write header if file doesn't exist
    file_exists = os.path.exists(FEEDBACK_LOG_PATH)
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "user_id", "question", "answer", "feedback", "comment"])
        writer.writerow([timestamp, user_id, question, answer, user_feedback, comment])
    
    logger.info(f"📝 Feedback logged: {user_feedback} - {comment}")
    return {"status": "success", "message": "Feedback logged."}


# === Optional: Standalone Test ===
if __name__ == "__main__":
    result = log_feedback(
        question="What is the origin of the Wolaytta language?",
        answer="The Wolaytta language originates from the Omotic language family in southern Ethiopia.",
        user_feedback="good",
        comment="Very helpful explanation.",
        user_id="test_user"
    )
    print(result)
