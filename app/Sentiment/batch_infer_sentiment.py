import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/sentiment_model"
LABELS = ["negative", "neutral", "positive"]
INPUT_CSV = "data/test_sentences.csv"  
OUTPUT_CSV = "data/test_predictions.csv"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    labels = [LABELS[p] for p in preds]
    confidences = probs.max(dim=1).values.cpu().numpy()
    return labels, confidences

def main():
    df = pd.read_csv(INPUT_CSV)
    texts = df["text"].dropna().tolist()

    all_labels = []
    all_confidences = []

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        labels, confs = predict(batch_texts)
        all_labels.extend(labels)
        all_confidences.extend(confs)

    df["predicted_label"] = all_labels
    df["confidence"] = all_confidences
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Batch prediction completed. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
