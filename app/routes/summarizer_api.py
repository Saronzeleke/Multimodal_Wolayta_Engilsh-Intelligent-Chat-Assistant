from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
router = APIRouter()
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG = "en_XX"
TGT_LANG = "en_XX"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.src_lang = SRC_LANG
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load summarization model: {e}")

# Request schema
class SummarizationRequest(BaseModel):
    text: str
    src_lang: str = SRC_LANG
    tgt_lang: str = TGT_LANG

# POST endpoint
@router.post("/summarize/")
async def summarize_text(payload: SummarizationRequest):
    try:
        tokenizer.src_lang = payload.src_lang
        inputs = tokenizer(
            payload.text, return_tensors="pt", max_length=512, truncation=True
        ).to(device)

        summary_ids = model.generate(
            **inputs,
            decoder_start_token_id=tokenizer.lang_code_to_id[payload.tgt_lang],
            forced_bos_token_id=tokenizer.lang_code_to_id[payload.tgt_lang],
            num_beams=4,
            min_length=30,
            max_length=120,
            length_penalty=2.0,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
