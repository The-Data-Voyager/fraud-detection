# Real-Time Fraud Detection — Run Guide

End-to-end fraud detection on the IEEE-CIS dataset: load → engineer features →
train XGBoost → SHAP explanations → stream scoring → database → live dashboard.

## Setup
```bash
pip install -r requirements.txt
```
Data lives in `DATA/ieee-fraud-detection/` (the 5 competition CSVs).

## Pipeline (run from the project root)

| Phase | Command | What it does |
|------|---------|--------------|
| 1 Load | `python src/data_load.py` | Merge txn+identity, downcast, cache Parquet to `DATA/cache/` |
| 2 EDA | `python src/eda.py` | Class imbalance, missingness, leakage audit → `reports/phase2_eda.md` |
| 3 Features | `python src/features.py` | Fit `FeatureBuilder`, save `models/feature_builder.pkl` |
| 4 Train | `python src/train.py` | Time-ordered holdout XGBoost → `models/xgb_model.json` + metrics |
| 5 Explain | `python src/explain.py` | SHAP reason-sentence smoke test |
| 7 Stream | `python src/run_stream_sim.py --reset --limit 1000` | Score a live stream into the DB |
| 9 Dashboard | `streamlit run dashboard/app.py` | Live Streamlit + Plotly dashboard |

## Current results (time-respecting validation)
- Fraud rate: **3.50%** (~28:1) · 590,540 train rows · 182 days, time-sorted
- **PR-AUC 0.566 · ROC-AUC 0.911 · Precision@top-1% ≈ 90%**

## Demo (Phase 11)
1. `python src/run_stream_sim.py --reset --limit 2000 --delay 0.1`  (in one terminal)
2. `streamlit run dashboard/app.py`  (in another) — watch metrics, stream, and
   high-risk alerts update live, each flagged transaction showing a risk score
   (0–100) and a natural-language reason.

## Storage / infra backends
- **Database:** defaults to local SQLite (`DATA/fraud.db`). For PostgreSQL, set
  `FRAUD_DB_URL=postgresql+psycopg2://user:pass@host:5432/fraud` — no code change.
- **Streaming:** the simulator needs no broker. For real Kafka, set
  `KAFKA_BOOTSTRAP=localhost:9092`, run `src/stream_producer.py` and
  `src/stream_consumer.py` — identical scoring/DB code.

## Phase 8 — PySpark (distributed feature engineering)
```bash
python src/pyspark_features.py
```
Same entity aggregations as `features.py`, expressed as Spark Window ops that
scale past one machine. Needs Java 8+ and `pyspark==3.5.3` (Spark 4 needs Java 17).

## Phase 10 — GenAI reason layer (free Groq LLM)
1. Free key: https://console.groq.com/keys
2. `set GROQ_API_KEY=...`  (PowerShell: `$env:GROQ_API_KEY="..."`)
3. `python src/run_stream_sim.py --reset --limit 1000 --genai`

Flagged transactions get their reason rephrased by Llama (via Groq) into a
crisp investigator sentence. Without a key it silently uses the SHAP template
reason — the pipeline never depends on the LLM. Model via `FRAUD_LLM_MODEL`
(default `llama-3.1-8b-instant`).
