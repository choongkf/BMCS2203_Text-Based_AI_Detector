from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from pathlib import Path

app = FastAPI(title="AI Text Detector API")

# Allow frontend to call this API during development
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # consider restricting in production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

model_dir = Path(__file__).parent / "model"
tokenizer = RobertaTokenizer.from_pretrained(str(model_dir))

# Auto device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained(str(model_dir))
model.to(device)
model.eval()

class PredictRequest(BaseModel):
	text: str

MIN_TOKENS = 20            # tokens including special tokens
CONF_THRESHOLD = 0.15       # confidence gap threshold
AI_THRESHOLD = 0.55         # minimum prob to call AI

def resolve_label_indices():
    id2 = getattr(model.config, "id2label", None)
    if isinstance(id2, dict) and len(id2) == 2:
        norm = {int(k): str(v).lower() for k, v in id2.items()}
        ai_idx = next((k for k, v in norm.items() if 'ai' in v), 1)
        human_idx = next((k for k, v in norm.items() if 'human' in v), 0)
        return int(ai_idx), int(human_idx), id2
    # fallback
    return 1, 0, None

AI_INDEX, HUMAN_INDEX, RAW_ID2 = resolve_label_indices()

@app.post("/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    token_count = int(enc["input_ids"].shape[1])
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs_tensor = torch.softmax(logits, dim=-1)[0]
        probs = probs_tensor.detach().cpu().tolist()

    prob_ai = float(probs[AI_INDEX])
    prob_human = float(probs[HUMAN_INDEX])
    confidence = abs(prob_ai - prob_human)

    # Decide label using confidence + custom AI threshold
    if confidence < CONF_THRESHOLD:
        label = "Uncertain"
    else:
        if prob_ai >= AI_THRESHOLD and prob_ai > prob_human:
            label = "AI"
        elif prob_human > prob_ai:
            label = "Human"
        else:
            label = "Uncertain"

    warning = None
    if token_count < MIN_TOKENS:
        warning = f"Input is short ({token_count} tokens); prediction may be unreliable."

    return {
        "prob_ai": prob_ai,
        "prob_human": prob_human,
        "label": label,
        "confidence": confidence,
        "token_count": token_count,
        "warning": warning,
        "ai_index": AI_INDEX,
        "human_index": HUMAN_INDEX
    }

@app.get("/health")
def health():
	return {"status": "ok"}

@app.get("/debug/indices")
def debug_indices():
    return {
        "ai_index": AI_INDEX,
        "human_index": HUMAN_INDEX,
        "raw_id2label": RAW_ID2,
        "thresholds": {
            "confidence_gap": CONF_THRESHOLD,
            "ai_threshold": AI_THRESHOLD,
            "min_tokens": MIN_TOKENS
        }
    }

if __name__ == "__main__":
	import uvicorn
	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

