import re
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
EMAIL_REGEX = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_REGEX = re.compile(r"\+?\d[\d\-\s\(\)]{5,}\d")    

ENTITY_THRESHOLDS = {
    "NAME_STUDENT": 0.30,
    "EMAIL": 0.60,
    "PHONE_NUM": 0.60,
    "URL_PERSONAL": 0.70,
    "USERNAME": 0.50,
    "ID_NUM": 0.60,
    "STREET_ADDRESS": 0.40,
    "DATE": 0.60,
}

def validate_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return 7 <= len(digits) <= 15

_ner = None

def get_ner_pipeline():
    global _ner
    if _ner is None:
        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_DIR = BASE_DIR / "notebooks" / "ner_model_2026-01-12"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_DIR, 
            local_files_only=True, 
            torch_dtype=torch.float32
        )

        model.eval()

        _ner = pipeline(
            "token-classification",
            model=str(MODEL_DIR),
            tokenizer=str(MODEL_DIR),
            aggregation_strategy="simple",
            device=-1  # CPU    
        )
    return _ner



def postprocess_entities(text: str, entities: list[dict]) -> list[dict]:
    final = []

    for e in entities:
        label = e.get("entity_group")
        score = float(e.get("score", 0.0))
        word = e.get("word", "")

        if score < ENTITY_THRESHOLDS.get(label, 0.5):
            continue

        normalized = word.replace(" ", "")

        if label == "EMAIL" and not EMAIL_REGEX.match(normalized):
            continue

        if label == "PHONE_NUM" and not validate_phone(word):
            continue

        final.append({
            "type": label,
            "value": word,
            "normalized": normalized,
            "score": round(score, 2),
            "start": int(e.get("start", -1)),
            "end": int(e.get("end", -1)),
        })

    return final

def detect_pii(text: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        ner = get_ner_pipeline()
        raw_entities = ner(text)
        return postprocess_entities(text, raw_entities)

    except Exception as e:
        print("NER failed, using regex fallback:", e)

    # Regex fallback
    entities = []

    # in case ner won't work, the regex fallback will at least catch emails and phone numbers   
    for m in EMAIL_REGEX.finditer(text):
        entities.append({
            "type": "EMAIL",
            "value": m.group(),
            "start": m.start(),
            "end": m.end(),
        })

    for m in PHONE_REGEX.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 7 <= len(digits) <= 15:
            entities.append({
                "type": "PHONE_NUM",
                "value": m.group(),
                "start": m.start(),
                "end": m.end(),
            })

    return entities

