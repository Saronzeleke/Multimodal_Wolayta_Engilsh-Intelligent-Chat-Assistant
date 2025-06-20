import pandas as pd
from sklearn.metrics import classification_report

CSV_PATH = "data/test_predictions.csv"

def evaluate():
    df = pd.read_csv(CSV_PATH)

    if "true_label" not in df or "predicted_label" not in df:
        raise ValueError("CSV must have both 'true_label' and 'predicted_label' columns")

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    print("📊 Evaluation Report:")
    print(classification_report(y_true, y_pred, labels=["negative", "neutral", "positive"]))

if __name__ == "__main__":
    evaluate()
