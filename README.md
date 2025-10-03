#  MLOps Internship Project – Wine Quality Prediction API

## 📌 Overview
This project demonstrates an **end-to-end MLOps pipeline**:
- Data ingestion, validation, and feature engineering with **DVC**
- Model training and experiment tracking with **scikit-learn + MLflow**
- Serving the trained model as a **FastAPI** service
- Containerization with **Docker**
- Deployment on **Render (Free Plan)**

Dataset: **Red Wine Quality** dataset.  
Goal: Predict wine quality (integer score) from 11 chemical features.

---

## ⚙️ Workflow

```
Data → DVC (ingest/validate/features) → Train (scikit-learn + MLflow) 
   → API (FastAPI) → Docker → Deployment (Render)
```

**Pipeline stages:**
1. **Ingest**: load raw CSV data  
2. **Validate**: schema validation with Pandera  
3. **Features**: feature engineering, split into train/test  
4. **Train**: RandomForest model, metrics logged with MLflow  
5. **Serve**: FastAPI app with `/health`, `/docs`, `/predict`  
6. **Deploy**: Hosted live on Render  

---

##  Project Structure

```
├── data/                # raw & processed data
├── models/              # trained artifacts (model.pkl, transformer.joblib)
├── src/
│   ├── api/             # FastAPI app (main.py)
│   ├── data/            # ingestion & validation scripts
│   ├── features/        # feature engineering
│   └── models/          # training & inference utilities
├── dvc.yaml             # pipeline definition
├── requirements.txt     # dependencies
└── Dockerfile           # container definition
```

---

## Deployment (Render)

**Live URL:** [https://mlops-internship.onrender.com](https://mlops-internship.onrender.com)

### Endpoints
- `GET /health` → Health check
- `GET /docs` → Swagger UI (interactive docs)
- `POST /predict` → Predict wine quality

---

## Example Usage

### Request
```bash
curl -X POST https://mlops-internship.onrender.com/predict   -H "Content-Type: application/json"   -d '{"features": [[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]]}'
```

### Response
```json
{
  "predictions": [6],
  "probabilities": [[...]]
}
```

---

## 🔧 Local Development

### 1. Clone repo
```bash
git clone https://github.com/<your-username>/mlops-internship.git
cd mlops-internship
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run API locally
```bash
uvicorn src.api.main:app --reload
```
Open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 Monitoring
- **Logs**: View via Render Dashboard
- **Planned**: Weekly **data drift detection** using Evidently AI to compare new vs training data distributions.

---

## 📈 Results
- Trained on red wine dataset (`winequality-red.csv`)
- Baseline model: RandomForest
