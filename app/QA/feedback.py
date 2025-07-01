import csv
import os
from datetime import datetime
from typing import Literal

FEEDBACK_LOG_PATH = "app/QA/feedback_logs.csv"

os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)


def log_feedback(question: str, answer: str, user_feedback: Literal["good", "bad", "neutral"], comment: str = ""):
    """Log user feedback on generated answers."""
    timestamp = datetime.now().isoformat()
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, question, answer, user_feedback, comment])
    return {"status": "success", "message": "Feedback logged."}


if __name__ == "__main__":
    # Example usage
    result = log_feedback(
        question="What is the history of the Wolaytta Kingdom?",
        answer="The Wolaytta Kingdom was a powerful polity in southern Ethiopia...",
        user_feedback="good",
        comment="Very informative."
    )
    print(result)
