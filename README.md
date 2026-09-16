# AI-Powered Real-Time Fraud Detection & Investigation System

A working prototype of a Big Data + AI fraud-detection platform: it trains on
historical transactions, simulates a live stream, assigns each transaction a
**Fraud Risk Score (0–100)**, flags high-risk ones, and **explains why** — all
surfaced on a live dashboard.

Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
(590,540 training transactions, 3.5% fraud).

## Architecture

```
 CSV (train/test)                     ┌── PySpark (distributed feature agg, Phase 8)
        │                             │
        ▼                             ▼
  data_load.py ──► features.py ──► train.py (XGBoost) ──► models/
  (merge+downcast) (FeatureBuilder)  (time-CV, PR-AUC)      │
        │                                                   ▼
        │                                            explain.py (SHAP)
        ▼                                                   │
  stream_producer ──► [Kafka / in-proc queue] ──► scorer.py ──► db.py (Postgres/SQLite)
  (test rows, DT order)                          (score+explain)      │
                                                       │              ▼
                                          genai_explain.py      dashboard/app.py
                                          (Groq LLM, Phase 10)  (Streamlit + Plotly)
```

## Results (time-respecting validation — not a random split)

| Metric | Value |
|---|---|
| PR-AUC (average precision) | **0.566** |
| ROC-AUC | **0.911** |
| Precision @ top-1% riskiest | **~90%** |
| Fraud base rate | 3.50% (~28:1) |

Validation uses the latest ~20% of the 182-day span as holdout, mirroring how
the live stream arrives (strictly later in time).

## Tech stack
Python · pandas · **XGBoost** · **SHAP** · **PySpark** · SQLAlchemy
(**PostgreSQL**/SQLite) · **Kafka** (with in-process simulator) · **Streamlit + Plotly** ·
**Groq** (free LLM) for natural-language explanations.

## Quick start
```bash
pip install -r requirements.txt
python src/data_load.py       # Phase 1: load & cache
python src/eda.py             # Phase 2: EDA + leakage audit
python src/train.py           # Phases 3-4: features + XGBoost
python src/run_stream_sim.py --reset --limit 2000 --delay 0.1   # Phase 7: live scoring
streamlit run dashboard/app.py                                  # Phase 9: dashboard
```
Full guide, backends (Postgres/Kafka/Groq), and PySpark: see [HOWTO.md](HOWTO.md).

## Demo walkthrough (Phase 11)
1. **Historical training** — show `train.py` metrics: 90% precision on the riskiest 1%.
2. **Go live** — run the stream sim in one terminal; watch it score transactions.
3. **A fraud pops** — a HIGH-RISK alert appears with **Risk 100/100** and a reason
   like *"Reason: Transaction amount and behavioural pattern significantly differ
   from normal transaction patterns."*
4. **Dashboard** — total processed, fraud %, live stream, risk distribution, trend,
   and the high-risk alert panel, all refreshing live.
