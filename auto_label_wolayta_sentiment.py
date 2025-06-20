import pandas as pd
from transformers import pipeline
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_CSV = "data/wolayta.csv"  
OUTPUT_CSV = "data/wolayta_sentiment_labeled.csv"

HF_TO_CUSTOM_LABEL = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL":"neutral"
}

def map_label(hf_label):
    return HF_TO_CUSTOM_LABEL.get(hf_label.upper(), "neutral")

def main():
    logger.info(f"Loading dataset from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if not {"Wolaytta", "English"}.issubset(df.columns):
        raise ValueError("Input CSV must have 'Wolaytta' and 'English' columns")

    logger.info("Loading Hugging Face English sentiment pipeline")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


    batch_size = 64
    labels = []

    logger.info(f"Labeling {len(df)} sentences in batches of {batch_size}")

    for start_idx in range(0, len(df), batch_size):
        batch_texts = df["English"].iloc[start_idx:start_idx + batch_size].tolist()
        results = sentiment_analyzer(batch_texts)

        batch_labels = [map_label(res["label"]) for res in results]
        labels.extend(batch_labels)

        logger.info(f"Labeled batch {start_idx} to {start_idx + len(batch_texts) - 1}")
        
    df["label"] = labels

    # Keep  text and label for training
    labeled_df = df[["Wolayta", "English", "label"]].rename(columns={"Wolayta": "text"})
    labeled_df=labeled_df.dropna(subset=["text","label"])
    logger.info(f"Saving labeled dataset to {OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    labeled_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Labeling completed successfully.")
if __name__ == "__main__":
    main()