# MLOps Internship Project (Weeks 8–12)

A practical, portfolio-ready MLOps project you can finish during Weeks 8–12. It follows a clear pipeline:
**Data → Validation → Features → Train → Track → Register → Serve → Containerize → Deploy → Monitor → Document**.

---

## 📅 Weekly Plan (Summary)

### Week 8 – Foundations & Data
- Repo + structure + CI (GitHub Actions)
- `requirements.txt`
- Data ingestion script (`src/data/make_dataset.py`)
- Pandera schema validation (`src/data/validate.py`)
- DVC for data versioning (`dvc init`, `dvc add`)

### Week 9 – Features & Training
- EDA (in `notebooks/`)
- Feature pipeline (`src/features/build_features.py`)
- Training (`src/models/train.py`) with MLflow logging
- DVC pipeline (`dvc.yaml`) and `dvc repro`

### Week 10 – CI/CD + Model Registry
- MLflow Model Registry (local or remote)
- GitHub Action for continuous training on pushes

### Week 11 – Serving & Container
- FastAPI app (`src/api/main.py`) with `/health` and `/predict`
- Dockerfile + local container testing

### Week 12 – Cloud & Monitoring & Docs
- Deploy container (Heroku / Cloud Run / App Runner)
- Structured logging
- Draft data drift plan with Evidently
- Final README + architecture diagram links


> **Note:** A local sample of the UCI Red Wine Quality dataset is included at `data/raw/winequality-red.csv` (semicolon-separated).
> To fetch the full original dataset from UCI later, rerun:
>
> ```bash
> python src/data/make_dataset.py --url https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv --sep ";" --save_path data/raw/winequality-red.csv
> ```


## 🧭 Quickstart

```bash
# 0) Create & activate venv (optional)
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
# venv\Scripts\activate

# 1) Install deps
pip install -r requirements.txt

# 2) Initialize DVC + Git (run inside repo root)
git init
dvc init

# 3) Ingest data (default: wine-quality red)
python src/data/make_dataset.py --url https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv --sep ";"

# 4) Validate data schema
python src/data/validate.py --input data/raw/data.csv --sep ";"

# 5) Build features
python src/features/build_features.py --input data/raw/data.csv --output data/processed/train.csv --sep ";"

# 6) Train & log with MLflow
python src/models/train.py --train data/processed/train.csv --target quality

# 7) Run API locally (loads latest MLflow model by path)
uvicorn src.api.main:app --reload
```

> Replace dataset URL with your internship dataset later. Keep the same commands—only change paths/columns in `params.yaml` or script flags.

---

## 📦 Repo Layout

```
.
├─ .github/workflows/
│  ├─ ci.yml
│  └─ cd-train.yml
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ src/
│  ├─ data/
│  │  ├─ make_dataset.py
│  │  └─ validate.py
│  ├─ features/
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ train.py
│  │  └─ infer_utils.py
│  └─ api/
│     └─ main.py
├─ dvc.yaml
├─ params.yaml
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## 🧪 CI
- `ci.yml` runs linting/quick import tests.
- `cd-train.yml` shows how to run DVC/MLflow on push (tweak secrets as needed).

---

## 🗃 DVC
Track large files, datasets and pipeline stages. Example:
```bash
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc data/.gitignore
git commit -m "Track raw data with DVC"
```

---

## 🧰 MLflow
- Local tracking default: `mlruns/` folder.
- To run UI: `mlflow ui --backend-store-uri mlruns`

---

## ☁️ Deployment
- Fill `infra/` with provider-specific scripts (Heroku/Cloud Run). See README sections inside.

---

## 🧩 Next Steps
- Swap dataset to your internship dataset.
- Update `params.yaml` with correct target/feature columns.
- Add unit tests and expand CI.
- Add data drift monitoring plan (Evidently) in README.
