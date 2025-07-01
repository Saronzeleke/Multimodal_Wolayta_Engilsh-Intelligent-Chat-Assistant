import csv
import os
import statistics
import logging
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# === CONFIG ===
FEEDBACK_LOG_PATH = "app/QA/feedback_logs.csv"
EVAL_LOG_PATH = "app/QA/bleu_evaluation_log.csv"

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BLEU-Evaluator")

def read_feedback_data(log_path):
    """Reads 'good' feedback entries from log."""
    data = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("feedback", "").lower() == "good":
                    reference = row.get("question", "").strip()
                    candidate = row.get("answer", "").strip()
                    if reference and candidate:
                        data.append((reference, candidate))
    except FileNotFoundError:
        logger.warning(f"⚠️ Feedback log not found at {log_path}")
    return data

def compute_bleu_scores(pairs):
    """Compute BLEU scores for a list of (reference, candidate) pairs."""
    scores = []
    smoothie = SmoothingFunction().method4
    for reference, candidate in pairs:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        scores.append(score)
    return scores

def evaluate_bleu():
    """Main evaluation function."""
    feedback_pairs = read_feedback_data(FEEDBACK_LOG_PATH)
    if not feedback_pairs:
        logger.warning("No valid 'good' feedback found for BLEU evaluation.")
        return None

    scores = compute_bleu_scores(feedback_pairs)
    avg_bleu = statistics.mean(scores)
    timestamp = datetime.utcnow().isoformat()

    os.makedirs(os.path.dirname(EVAL_LOG_PATH), exist_ok=True)
    with open(EVAL_LOG_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "num_samples", "average_bleu"])
        writer.writerow([timestamp, len(scores), avg_bleu])

    logger.info(f"✅ BLEU evaluation complete — {len(scores)} samples, Avg BLEU: {avg_bleu:.4f}")
    return avg_bleu

if __name__ == "__main__":
    evaluate_bleu()
