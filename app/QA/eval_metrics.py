import csv
import os
import logging
from typing import List, Tuple, Dict, Optional

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import CHRF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvalMetrics")

# Constants for file paths
FEEDBACK_LOG_PATH = "app/QA/feedback_logs.csv"
EVAL_REPORTS_DIR = "app/QA/reports/"

# Helper: Read feedback data (question as reference, answer as candidate)
def read_feedback_data(log_path: str = FEEDBACK_LOG_PATH) -> List[Tuple[str, str]]:
    """
    Reads feedback CSV and returns list of (reference, candidate) pairs
    Only positive feedback ("good") is considered.
    """
    data = []
    if not os.path.exists(log_path):
        logger.warning(f"Feedback log not found at {log_path}")
        return data

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("feedback", "").lower() == "good":
                ref = row.get("question", "").strip()
                cand = row.get("answer", "").strip()
                if ref and cand:
                    data.append((ref, cand))
    logger.info(f"Loaded {len(data)} positive feedback samples")
    return data

# BLEU computation (sentence-level average)
def compute_bleu(pairs: List[Tuple[str, str]]) -> float:
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, cand in pairs:
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        scores.append(score)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"BLEU avg score: {avg_score:.4f}")
    return avg_score

# ROUGE computation (using rouge_scorer package)
def compute_rouge(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for ref, cand in pairs:
        scores = scorer.score(ref, cand)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    logger.info(f"ROUGE-1 avg: {avg_rouge1:.4f}, ROUGE-2 avg: {avg_rouge2:.4f}, ROUGE-L avg: {avg_rougeL:.4f}")
    return {"rouge1": avg_rouge1, "rouge2": avg_rouge2, "rougeL": avg_rougeL}

# METEOR computation (sentence-level average)
def compute_meteor(pairs: List[Tuple[str, str]]) -> float:
    scores = []
    for ref, cand in pairs:
        score = meteor_score([ref], cand)
        scores.append(score)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"METEOR avg score: {avg_score:.4f}")
    return avg_score

# ChrF computation (using sacrebleu)
def compute_chrf(pairs: List[Tuple[str, str]]) -> float:
    chrf_scorer = CHRF()
    scores = []
    for ref, cand in pairs:
        score = chrf_scorer.sentence_score(cand, [ref]).score
        scores.append(score)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"ChrF avg score: {avg_score:.4f}")
    return avg_score

# Main evaluation function to return all metrics
def evaluate_all_metrics(feedback_path: Optional[str] = None) -> Dict[str, float]:
    path = feedback_path if feedback_path else FEEDBACK_LOG_PATH
    data = read_feedback_data(path)
    if not data:
        logger.warning("No data to evaluate.")
        return {}

    results = {}
    results['BLEU'] = compute_bleu(data)
    rouge_results = compute_rouge(data)
    results.update(rouge_results)
    results['METEOR'] = compute_meteor(data)
    results['ChrF'] = compute_chrf(data)

    return results

if __name__ == "__main__":
    scores = evaluate_all_metrics()
    print("Evaluation results:", scores)
