import os
import logging
import pandas as pd
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_CSV =r"C:\Users\admin\Desktop\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\wolayta.csv"
OUTPUT_CSV =r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\wolayta_sentiment_labeled.csv"

def main():
    logger.info(f"Loading dataset from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if not {"Wolaytta", "English"}.issubset(df.columns):
        raise ValueError("CSV must contain 'Wolaytta' and 'English' columns")

    logger.info("Initializing 3-class sentiment pipeline")
    sentiment = pipeline(
        "text-classification",
        model="j-hartmann/sentiment-roberta-large-english-3-classes",
        return_all_scores=True
    )

    labels = []
    batch_size = 64
    for start in range(0, len(df), batch_size):
        batch = df["English"].iloc[start:start + batch_size].tolist()
        results = sentiment(batch)
        for res in results:
            label = max(res, key=lambda x: x["score"])["label"].lower()
            labels.append(label)

        logger.info(f"Labeled samples {start} to {start + len(batch) - 1}")

    df["label"] = labels
    labeled = df[["Wolaytta", "English", "label"]].rename(columns={"Wolaytta": "text"})
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    labeled.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"✅ Sentiment-labeled data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
