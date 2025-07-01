import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import statistics
import logging
from datetime import datetime

# === CONFIG ===
FEEDBACK_LOG_PATH = "app/QA/feedback_logs.csv"
EVAL_LOG_PATH = "app/QA/bleu_evaluation_log.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BLEU-Evaluator")

def read_feedback_data(log_path):
    data = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["feedback"].lower() == "good":  # Only evaluate if feedback is positive
                reference = row["question"].strip()
                candidate = row["answer"].strip()
                if reference and candidate:
                    data.append((reference, candidate))
    return data

def compute_bleu_scores(pairs):
    scores = []
    smoothie = SmoothingFunction().method4
    for reference, candidate in pairs:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        scores.append(score)
    return scores

def evaluate_bleu():
    feedback_pairs = read_feedback_data(FEEDBACK_LOG_PATH)
    if not feedback_pairs:
        logger.warning("No valid feedback data found for BLEU evaluation.")
        return None

    scores = compute_bleu_scores(feedback_pairs)
    average_score = statistics.mean(scores)

    # Log the result
    timestamp = datetime.utcnow().isoformat()
    os.makedirs("app/QA", exist_ok=True)
    with open(EVAL_LOG_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "num_samples", "average_bleu"])
        writer.writerow([timestamp, len(scores), average_score])

    logger.info(f"✅ BLEU Evaluation Complete: {len(scores)} samples, Avg BLEU: {average_score:.4f}")
    return average_score

if __name__ == "__main__":
    evaluate_bleu()
