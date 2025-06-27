import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

# Paths
DATA_PATH = "data/wolayta_summarization.csv"
MODEL_NAME = "t5-small"
SAVE_DIR = "models/summarization_model"

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "summary"])

# Add prefix as T5 expects
df["input"] = "summarize: " + df["text"]
df["target"] = df["summary"]

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["input", "target"]])
dataset = dataset.train_test_split(test_size=0.1)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Tokenization
def preprocess(example):
    model_inputs = tokenizer(example["input"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=20
)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Fine-tuned summarization model saved to {SAVE_DIR}")
