import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import torch
import os
MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_PATH = "data/wolayta.csv"
MODEL_DIR = "models/sentiment_model"
NUM_LABELS = 3
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

# Load and preprocess data
def load_dataset_from_csv(path):
    df = pd.read_csv(path)
    df = df[df['label'].isin(LABEL_MAP.keys())].copy()
    df['label_id'] = df['label'].map(LABEL_MAP)
    return Dataset.from_pandas(df[['text', 'label_id']].dropna())

# ------------------------------
# -- Step 2: Tokenize Dataset --
# ------------------------------
def tokenize_dataset(dataset, tokenizer):
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label_id", "label")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized.train_test_split(test_size=0.2, seed=42)

# ----------------------------
# -- Step 3: Train the Model --
# ----------------------------
def train_sentiment_model():
    print("📥 Loading dataset...")
    dataset = load_dataset_from_csv(DATA_PATH)

    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("🧼 Tokenizing dataset...")
    encoded = tokenize_dataset(dataset, tokenizer)

    print("🧠 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    print("🛠️ Setting up Trainer...")
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    print("🚀 Training model...")
    trainer.train()

    print("💾 Saving model...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✅ Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Using CPU. (Training will be slower)")
    train_sentiment_model()
