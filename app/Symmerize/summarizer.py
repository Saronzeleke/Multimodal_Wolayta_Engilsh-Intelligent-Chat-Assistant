import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap
import pandas as pd

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG = "en_XX"
TGT_LANG = "en_XX"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.src_lang = SRC_LANG 
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
def summarize(text, src_lang=SRC_LANG, tgt_lang=TGT_LANG):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(
        **inputs,
        decoder_start_token_id=tokenizer.lang_code_to_id[tgt_lang],
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        num_beams=4,
        min_length=30,
        max_length=120,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
def read_text_chunks(file_path, chunk_size=450):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks
if __name__ == "__main__":
    input_path = r"C:\Users\admin\Multimodal_Wolayta_Engilsh-Intelligent-Chat-Assistant\data\wolayta.csv"
    chunks = read_text_chunks(input_path)
    print("📚 Total Chunks:", len(chunks))
    print("🔍 Summarizing...\n")

    for i, chunk in enumerate(chunks, 1):
        summary = summarize(chunk)
        print(f"🧩 Chunk {i} Summary:")
        print(textwrap.fill(summary, width=80))
        print("="*80)

    all_summaries = [
        {"chunk": i+1, "text": chunks[i], "summary": summarize(chunks[i])}
        for i in range(len(chunks))
    ]
    pd.DataFrame(all_summaries).to_csv("data/wolayta_summaries.csv", index=False)
    print("✅ Saved summarized chunks to CSV.")
