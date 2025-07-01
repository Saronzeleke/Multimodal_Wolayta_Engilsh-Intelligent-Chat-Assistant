from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method4)

def evaluate_bleu(feedback_path: str):
    df = pd.read_csv(feedback_path, header=None,
                     names=["timestamp", "lang", "question", "model_answer", "user_feedback"])
    
    scored = df[df["user_feedback"].str.contains(" ")]

    bleu_scores = [
        compute_bleu(row["user_feedback"], row["model_answer"])
        for _, row in scored.iterrows()
    ]

    return {
        "average_bleu": round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0.0,
        "count": len(bleu_scores)
    }

if __name__ == "__main__":
    stats = evaluate_bleu("data/feedback_logs.csv")
    print("🔍 BLEU Evaluation Results:", stats)
