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

@app.post("/predict")
def predict(req: PredictRequest):
	text = (req.text or "").strip()
	if not text:
		raise HTTPException(status_code=400, detail="Text must not be empty.")

	inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
	# move to device
	inputs = {k: v.to(device) for k, v in inputs.items()}
	with torch.no_grad():
		outputs = model(**inputs)
		probs_tensor = torch.softmax(outputs.logits, dim=-1)[0]
		probs = probs_tensor.detach().cpu().tolist()
	# Assuming label 1 = AI, 0 = Human following training label encoding
	return {
		"prob_ai": probs[1],
		"prob_human": probs[0],
		"label": "AI" if probs[1] >= probs[0] else "Human"
	}

@app.get("/health")
def health():
	return {"status": "ok"}

if __name__ == "__main__":
	import uvicorn
	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

