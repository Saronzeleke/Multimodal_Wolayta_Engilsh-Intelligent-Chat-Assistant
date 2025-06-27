

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

PRED_CSV = r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\test_predictions.csv"
df = pd.read_csv(PRED_CSV)

# Normalize labels and remove punctuation
for col in ["label", "predicted_label"]:
    df[col] = (
        df[col]
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
    )

print("📊 Classification Report")
print(classification_report(df["label"], df["predicted_label"], labels=["negative", "neutral", "positive"]))

cm = confusion_matrix(df["label"], df["predicted_label"], labels=["negative", "neutral", "positive"])
disp = ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Sentiment Confusion Matrix")
plt.show()
plt.close()

if "language" in df.columns:
    print("\n🌍 Per-Language Accuracy")
    for lang in df["language"].unique():
        sub_df = df[df["language"] == lang]
        acc = (sub_df["label"] == sub_df["predicted_label"]).mean()
        print(f"  {lang}: {acc:.2%}")

print("\n❌ Sample Wrong Predictions:")
wrong = df[df["label"] != df["predicted_label"]].head(5)
for _, row in wrong.iterrows():
    print(f"- Sentence: {row.get('sentence', '<N/A>')}")
    print(f"  True: {row['label']}, Predicted: {row['predicted_label']}, Confidence: {row.get('confidence', 0):.2%}")
