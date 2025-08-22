## AI Text Detector (Human vs AI Generated Text)

Simple prototype that fine-tunes a RoBERTa model to classify text as Human or AI generated and serves predictions via a FastAPI backend with a lightweight HTML/JS frontend.

---
### 1. Project Structure
```
AI_text_detector_prototype/
├─ backend/
│  ├─ train_model.py        # Fine-tune / resume training script
│  ├─ main.py               # FastAPI app ( /health , /predict )
│  ├─ requirements.txt      # Python dependencies
│  └─ model/                # Saved model + checkpoints (ignored in git except .gitkeep)
├─ frontend/
│  ├─ index.html            # UI
│  ├─ script.js             # Calls backend /predict
│  └─ style.css             # Styling
├─ Training_Essay_Data.csv  # Dataset (ignored in git)
├─ venv/                    # Virtual environment (ignored)
└─ .gitignore
```

---
### 2. Requirements
* Python 3.11+ (3.12 tested)
* (Optional) GPU with CUDA for faster training
* Internet access initially to download the base `roberta-base` model (once cached/saved you can work offline)

---
### 3. Create & Activate Virtual Environment (Windows PowerShell)
From the project root:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```
If you later want to deactivate:
```powershell
deactivate
```

---
### 4. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r backend/requirements.txt
```

If you see model download warnings, allow the first run to complete so the files are cached.

---
### 5. Dataset Format
Expected CSV columns:
* `text` – the essay / passage content
* `generated` – label indicating source (e.g. `Human` / `AI` or similar)

The script automatically label-encodes `generated` (Human=0, AI=1 order depends on LabelEncoder). Ensure no empty rows.

Place the file as: `Training_Essay_Data.csv` in the project root (already done).

---
### 6. Training / Fine-Tuning
Run from project root or `backend/` after activating the venv:
```powershell
cd backend
python train_model.py
```
What it does:
1. Loads CSV (`../Training_Essay_Data.csv`).
2. Splits into train/validation (80/20).
3. Tokenizes with `roberta-base`.
4. Fine-tunes for 2 epochs (adjust in script).
5. Saves model + tokenizer to `backend/model/`.
6. Creates periodic checkpoints (`checkpoint-500`, `checkpoint-1000`, etc.).

#### Resume Training
You can stop training any time (Ctrl+C) and later run:
```powershell
python train_model.py
```
The script auto-detects the latest checkpoint and resumes.

#### Start Fresh
Delete checkpoint folders:
```powershell
Remove-Item -Recurse -Force .\model\checkpoint-*
```

Adjust hyperparameters in `train_model.py` (batch size, epochs, max_length).

---
### 7. Running the Backend API
After training (model files exist in `backend/model/`):
```powershell
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Endpoints:
* `GET /health` → `{ "status": "ok" }`
* `POST /predict` → Body: `{ "text": "Some text..." }`
  * Response example:
    ```json
    {
      "prob_ai": 0.72,
      "prob_human": 0.28,
      "label": "AI"
    }
    ```

Manual curl test:
```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text":"This is a test."}'
```

Docs (auto) at: http://127.0.0.1:8000/docs

---
### 8. Frontend
Files are in `frontend/`. To serve locally (so fetch calls work cleanly):
```powershell
cd frontend
python -m http.server 5500
```
Open: http://127.0.0.1:5500/index.html

Workflow:
1. Ensure backend is running (`uvicorn` at 127.0.0.1:8000).
2. Open frontend page.
3. Paste text → Detect → See probabilities and predicted label.

If you open `index.html` directly with `file://`, some browsers may block requests—use the simple server above.

---
### 9. Adjusting Prediction Logic
Current assumption: index 0 = Human, index 1 = AI after label encoding. If you find it reversed, swap `probs[0]` and `probs[1]` in `backend/main.py`.

---
### 10. Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| `HTTP 422` on /predict | Missing `text` field | Send `{ "text": "..." }` JSON |
| `400 Text must not be empty` | Empty/whitespace input | Provide actual text |
| CORS error in browser | Backend not running or different host | Use exact `http://127.0.0.1:8000` and keep CORS middleware |
| Model load error | Model directory empty | Re-run training to populate `backend/model/` |

---
### 11. Deployment Notes (Basic)
For a simple deployment later:
* Package backend with `uvicorn` behind a production server (e.g. `gunicorn -k uvicorn.workers.UvicornWorker`).
* Serve frontend as static files (S3, Netlify, simple nginx) pointing to your backend domain.
* Restrict CORS origins in `main.py`.
* Consider exporting a lighter model (DistilRoBERTa) if latency is a concern.

---
### 12. Future Improvements
* Add evaluation metrics (accuracy, F1) after each epoch.
* Persist training/eval metrics to a CSV or TensorBoard.
* Confidence calibration (e.g., temperature scaling).
* Add simple authentication or rate limiting to the API.
* Provide a download/share of JSON result.

---
### 13. Quick Start Recap
```powershell
# 1. Create env
python -m venv venv
.\venv\Scripts\Activate

# 2. Install deps
pip install -r backend/requirements.txt

# 3. Train (creates backend/model)
cd backend
python train_model.py

# 4. Run API
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 5. In another terminal serve frontend
cd ..\frontend
python -m http.server 5500

# 6. Open browser
http://127.0.0.1:5500/index.html
```

---
### 14. License / Use
Educational prototype. Review dataset licensing and Hugging Face model license before distribution.

---
### 15. Contact / Notes
Adjust epochs, batch size, and max_length in `train_model.py` for performance/accuracy trade-offs.

Happy detecting!
