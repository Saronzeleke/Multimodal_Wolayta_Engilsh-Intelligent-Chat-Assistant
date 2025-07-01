# File: app/QA/evaluate_bleu.py

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datetime import datetime


def compute_bleu(reference, hypothesis):
    """Compute BLEU score for a single sentence pair."""
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)


def evaluate_bleu(reference_csv_path, generated_csv_path, save_results=True):
    ref_df = pd.read_csv(reference_csv_path)
    gen_df = pd.read_csv(generated_csv_path)

    assert len(ref_df) == len(gen_df), "Mismatch in number of reference and generated samples"

    scores = []
    for ref, gen in zip(ref_df['answer'], gen_df['answer']):
        score = compute_bleu(ref, gen)
        scores.append(score)

    average_bleu = sum(scores) / len(scores)
    print(f"\n✅ Average BLEU Score: {average_bleu:.4f}")

    if save_results:
        output_path = "app/QA/bleu_eval_log.csv"
        eval_df = pd.DataFrame({
            "question": ref_df['question'],
            "reference_answer": ref_df['answer'],
            "generated_answer": gen_df['answer'],
            "bleu_score": scores,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        eval_df.to_csv(output_path, index=False)
        print(f"💾 Saved detailed BLEU results to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Path to reference CSV file")
    parser.add_argument("--gen", required=True, help="Path to generated CSV file")
    args = parser.parse_args()

    evaluate_bleu(args.ref, args.gen)
