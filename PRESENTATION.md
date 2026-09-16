# Project Walkthrough — Real-Time Fraud Detection
*Study guide for presentation. Read top to bottom; it's the whole story.*

---

## 1. The one-line pitch
> "I built a system that learns what fraud looks like from 590,000 past
> transactions, then watches a live stream of new transactions, scores each
> one 0–100 for fraud risk, flags the dangerous ones, and explains *why* — all
> shown on a live dashboard."

It combines **Big Data (Spark) + Machine Learning (XGBoost) + Real-time
streaming (Kafka) + Explainable AI (SHAP) + a dashboard (Streamlit)**.

---

## 2. The problem
Banks process millions of card transactions. A tiny fraction (~3.5%) are
fraud. You can't review them by hand, and you can't just block big amounts
(fraud is often *small* to avoid attention). We need a system that:
- learns fraud patterns automatically,
- scores transactions **in real time** as they arrive,
- explains its decision (a human investigator must know *why*),
- and shows it all on a dashboard.

**Dataset:** IEEE-CIS Fraud Detection (real anonymized card transactions).
590,540 training rows, 434 columns, 3.5% fraud.

---

## 3. The big picture (the flow)
```
Historical CSVs → Load & clean → Feature engineering → Train XGBoost model
                                                              │
                                                       SHAP explainer
                                                              │
Live stream (Kafka/simulator) → score each txn → save to database → Dashboard
```
Two halves: **(A) offline training** on history, **(B) online scoring** of the
live stream. The trained model is the bridge between them.

---

## 4. Walk through each phase (what + why)

**Phase 1 — Load the data** (`src/data_load.py`)
Merge the transaction file with the identity file (device/browser info) on
`TransactionID`. The CSV is 683 MB, so I **downcast** number types (float64→float32
etc.) to cut memory in half, and cache it as **Parquet** so it loads in seconds
next time instead of minutes.
*Talking point: "I optimized memory and caching because the raw data is huge."*

**Phase 2 — Explore & audit** (`src/eda.py` → `reports/phase2_eda.md`)
Measured the key facts that shape everything after:
- Fraud rate **3.5%** → the data is **imbalanced ~28:1**.
- Data is **sorted by time** over **182 days**.
- Ran a **leakage audit** — checked no column secretly gives away the answer
  (a classic mistake). It was clean.
*Talking point: "I grounded my design in real numbers before writing features."*

**Phase 3 — Feature engineering** (`src/features.py`)
Raw columns aren't enough. I engineered **41 new features**, the important ones:
- **Time features** — hour of day, day of week (fraud clusters at odd hours).
- **Entity key `card1_addr1`** — a card+location fingerprint to track behavior.
- **Behavioral (causal) features** — for each card: transaction *velocity*
  (how many recent txns), *time since last* txn, and *amount z-score* (is this
  amount unusual for this card?). "Causal" = each transaction only sees its own
  **past**, never the future — so there's no cheating.
- **Frequency encoding** — rare cards/emails/devices are riskier.
- **Missingness flags** — *missing* device info is itself a fraud signal.
*Talking point: "Behavioral features are what let it catch fraud that a single
transaction wouldn't reveal."*

**Phase 4 — Train the model** (`src/train.py`)
Model: **XGBoost** (gradient-boosted trees — strong on tabular data).
Two critical choices:
- **Time-respecting split** — I train on the earlier days and validate on the
  *latest* days, because in real life you predict the future from the past.
  A random split would leak future info and inflate the score dishonestly.
- **Handle imbalance** — `scale_pos_weight ≈ 28` tells the model fraud is rare
  but important, and I tuned for **PR-AUC** (the right metric for rare events),
  not plain accuracy (99% "accuracy" is trivial when fraud is 1%).

**Phase 5 — Explainability** (`src/explain.py`)
**SHAP** opens the black box: for any transaction it says which features pushed
the score up. I convert the top factors into a plain-English sentence:
*"Reason: transaction counting pattern and internal risk signal significantly
differ from normal patterns."*
*Talking point: "Explainability is required — an investigator won't act on a
number alone."*

**Phase 6 — Database** (`src/db.py`)
Every scored transaction is stored (id, amount, risk score, status, reason).
Uses **SQLAlchemy** — points at **PostgreSQL** in production, or a local
**SQLite** file for the demo, by changing one setting. Both the stream (writer)
and dashboard (reader) use this.

**Phase 7 — Real-time streaming** (`src/stream_producer.py`, `stream_consumer.py`,
`run_stream_sim.py`)
- **Producer** reads transactions in time order and emits them one-by-one to
  simulate a live feed (real version publishes to **Kafka**).
- **Consumer** reads each one, runs the model + SHAP, writes the result to the DB.
- To score correctly in real time, it keeps a small **history buffer per card**
  so the behavioral features (velocity etc.) are computed the same way as in
  training.
*Talking point: "The same feature code runs in training and in the live stream —
that consistency is what makes the predictions valid."*

**Phase 8 — PySpark** (`src/pyspark_features.py`)
The same behavioral aggregations, rewritten in **Spark** so they scale to
datasets too big for one machine. It confirmed the signal: fraud sits on cards
with **higher velocity (avg 519 vs 387 prior transactions)**.
*Talking point: "Spark is the Big Data piece — it does the heavy feature work
distributed across a cluster."*

**Phase 9 — Dashboard** (`dashboard/app.py`)
**Streamlit + Plotly**, auto-refreshing every 2 seconds. Shows: total processed,
frauds detected, fraud %, live transaction table, risk-score distribution, a
trend chart, and a **high-risk alert panel** with the score and reason.

**Phase 10 — GenAI (optional)** (`src/genai_explain.py`)
Wraps the SHAP reason through a **free LLM (Groq / Llama)** to phrase it more
naturally. Falls back to the template if no key — so it never breaks the demo.

---

## 5. The results (know these numbers cold)
Measured on the honest time-based validation set:
| Metric | Value | What it means |
|---|---|---|
| **Precision @ top-1%** | **~90%** | Of the 1% riskiest transactions, 90% are real fraud |
| **PR-AUC** | **0.566** | Strong for a 3.5%-fraud problem (random = 0.035) |
| **ROC-AUC** | **0.911** | The model ranks fraud above non-fraud 91% of the time |

**The headline line to say:** *"If the bank reviewed just the top 1% my model
flags, 9 out of 10 would be genuine fraud."*

---

## 6. Why some alerts show 100/100
Risk score = model probability × 100. Clear-fraud transactions get probabilities
like 0.999 → rounds to 100. Because I weighted the model for imbalance, it's very
confident on obvious fraud. Out of 2000 transactions only ~5 hit HIGH RISK —
**95% score LOW.** So 100/100 means "extremely confident," not "everything is fraud."

---

## 7. Live demo script (what to click tomorrow)
1. Show the **training metrics** — "90% precision on the riskiest 1%."
2. **Terminal 1:** `python src/run_stream_sim.py --reset --limit 5000 --delay 0.3`
3. **Terminal 2:** `streamlit run dashboard/app.py`
4. Point at the numbers climbing, the live table scrolling, the risk histogram.
5. Scroll to a **HIGH-RISK alert** — read out the score and the reason sentence.
6. Say: "This matches the goal — score, status, and a human explanation, live."

---

## 8. Likely questions (defense prep)
- **"Why XGBoost not deep learning?"** Tabular data — gradient-boosted trees beat
  neural nets here, train fast, and are easier to explain.
- **"How do you handle 3.5% imbalance?"** `scale_pos_weight` + PR-AUC metric,
  not accuracy.
- **"Isn't your 91% AUC just memorizing?"** No — time-based validation on unseen
  *future* days, so it's an honest estimate.
- **"What's 'leakage'?"** A feature that accidentally contains the answer. I
  audited for it and dropped absolute-time columns that wouldn't generalize.
- **"Is this real-time?"** Yes — the consumer scores each transaction as it
  arrives and keeps per-card history so behavioral features stay correct.
- **"Where's the Big Data part?"** Spark does distributed feature engineering;
  Kafka carries the live stream; PostgreSQL stores results.
- **"Could it run in production?"** Yes — swap SQLite→PostgreSQL and the
  simulator→real Kafka by changing config; the model/scoring code is unchanged.

---

## 9. One-sentence summary to close with
> "It's an end-to-end fraud platform: Spark and Kafka move and process the data,
> XGBoost scores it, SHAP explains it, and Streamlit visualizes it — turning raw
> transactions into an actionable, explained fraud alert in real time."
