import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
MODEL_DIR = "models/sentiment_model"
LABELS = ["negative", "neutral", "positive"]

# Load tokenizer and model
print("📦 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        label = LABELS[label_id]
        confidence = probs[0][label_id].item()

    return label, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python infer_sentiment.py \"<your sentence>\"")
        sys.exit(1)

    input_text = sys.argv[1]
    sentiment, conf = predict_sentiment(input_text)
    print(f"🗣️ Input: {input_text}")
    print(f"🔍 Predicted Sentiment: {sentiment} ({conf:.2%} confidence)")