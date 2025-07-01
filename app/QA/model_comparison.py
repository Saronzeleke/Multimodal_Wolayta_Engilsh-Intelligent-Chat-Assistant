import os
import csv
import json
import logging
import matplotlib.pyplot as plt
from typing import List, Dict
from eval_metrics import evaluate_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelComparison")
MODEL_OUTPUTS_DIR = "model_outputs/"  
REPORTS_DIR = "app/QA/reports/"

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_model_output(filepath: str) -> List[Dict[str, str]]:
    """
    Load model output from CSV or JSON.
    Expect list of dicts with keys: question, answer
    """
    data = []
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    elif ext == ".json":
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        logger.error(f"Unsupported file format: {ext}")
    return data

def save_report(report_data: Dict[str, Dict], filename: str):
    """Save evaluation report as JSON and CSV summary"""
    json_path = os.path.join(REPORTS_DIR, filename + ".json")
    csv_path = os.path.join(REPORTS_DIR, filename + ".csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", *list(next(iter(report_data.values())).keys())])
        for model_name, metrics in report_data.items():
            writer.writerow([model_name] + [metrics[m] for m in metrics])

    logger.info(f"Saved evaluation report to {json_path} and {csv_path}")

def plot_comparison(report_data: Dict[str, Dict[str, float]], filename: str):
    """
    Generate bar plots comparing metric scores across models.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(report_data.keys())
    metrics = list(next(iter(report_data.values())).keys())
    n_models = len(models)
    n_metrics = len(metrics)
    data_matrix = []
    for metric in metrics:
        data_matrix.append([report_data[model][metric] for model in models])

    data_matrix = np.array(data_matrix)

    x = np.arange(n_metrics)
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        ax.bar(x + i*width, data_matrix[:, i], width, label=model)

    ax.set_xticks(x + width*(n_models-1)/2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Scores")
    ax.set_title("Model Comparison by Metric")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, filename + ".png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved comparison plot to {plot_path}")

def compare_models(model_outputs: Dict[str, str]):
    """
    Run evaluation on multiple model outputs.

    Args:
        model_outputs: Dict of model_name -> file_path containing generated answers
    Returns:
        Dict of model_name -> metric scores dict
    """
    report = {}
    for model_name, filepath in model_outputs.items():
        logger.info(f"Evaluating model: {model_name}")
        # For simplicity, here assume CSV file with columns question, answer, feedback=good
        # In real usage, you might merge feedback or use a test set
        # Let's just evaluate on all rows (ignoring feedback)
        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref = row.get("question", "").strip()
                cand = row.get("answer", "").strip()
                if ref and cand:
                    pairs.append((ref, cand))
        if not pairs:
            logger.warning(f"No valid data in model output file: {filepath}")
            continue

        # Compute metrics using eval_metrics.py logic
        from eval_metrics import compute_bleu, compute_rouge, compute_meteor, compute_chrf

        model_metrics = {}
        model_metrics['BLEU'] = compute_bleu(pairs)
        rouge_scores = compute_rouge(pairs)
        model_metrics.update(rouge_scores)
        model_metrics['METEOR'] = compute_meteor(pairs)
        model_metrics['ChrF'] = compute_chrf(pairs)

        report[model_name] = model_metrics

    # Save report and plots
    save_report(report, "model_comparison_report")
    plot_comparison(report, "model_comparison_plot")

    return report


if __name__ == "__main__":
    # Example usage
    model_files = {
        "model_v1": "model_outputs/model_v1_output.csv",
        "model_v2": "model_outputs/model_v2_output.csv",
    }

    comparison_report = compare_models(model_files)
    print("Model comparison report:", comparison_report)
