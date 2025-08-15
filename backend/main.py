from fastapi import FastAPI
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
model = RobertaForSequenceClassification.from_pretrained(str(model_dir))
model.eval()

class PredictRequest(BaseModel):
	text: str

@app.post("/predict")
def predict(req: PredictRequest):
	inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
	with torch.no_grad():
		outputs = model(**inputs)
		probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
	# Assuming label 1 = AI, 0 = Human following training label encoding
	return {
		"prob_ai": probs[1],
		"prob_human": probs[0],
		"label": "AI" if probs[1] >= probs[0] else "Human"
	}

@app.get("/health")
def health():
	return {"status": "ok"}

