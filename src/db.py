"""
Phase 6 - Database layer for scored transactions.

Single SQLAlchemy engine driven by the FRAUD_DB_URL env var:
  * PostgreSQL (plan target):  postgresql+psycopg2://user:pass@localhost:5432/fraud
  * SQLite (zero-setup default): sqlite:///<project>/DATA/fraud.db

Both the Kafka consumer (writer) and the Streamlit dashboard (reader) use
this module, so the storage backend is a one-line config change.

Schema (one table is enough for the demo; split later if needed):
  scored_transactions
    transaction_id   BIGINT PK
    transaction_dt   BIGINT      -- seconds from dataset reference
    amount           DOUBLE
    fraud_prob       DOUBLE      -- model probability 0..1
    risk_score       INT         -- 0..100
    status           TEXT        -- HIGH RISK / MEDIUM / LOW
    reason           TEXT        -- SHAP-derived natural-language explanation
    scored_at        TIMESTAMP   -- when the consumer processed it
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    BigInteger, Column, DateTime, Float, Integer, String, Text,
    create_engine, func, select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE = f"sqlite:///{(PROJECT_ROOT / 'DATA' / 'fraud.db').as_posix()}"
DB_URL = os.environ.get("FRAUD_DB_URL", DEFAULT_SQLITE)

Base = declarative_base()


class ScoredTransaction(Base):
    __tablename__ = "scored_transactions"
    transaction_id = Column(BigInteger, primary_key=True, autoincrement=False)
    transaction_dt = Column(BigInteger)
    amount = Column(Float)
    fraud_prob = Column(Float)
    risk_score = Column(Integer)
    status = Column(String(16))
    reason = Column(Text)
    scored_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


_engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=_engine, future=True)

# SQLite: enable WAL so the dashboard can read while the stream writes.
if DB_URL.startswith("sqlite"):
    from sqlalchemy import event

    @event.listens_for(_engine, "connect")
    def _set_sqlite_wal(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()


def init_db(reset: bool = False) -> None:
    if reset:
        Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)


def insert_scored(rec: dict) -> None:
    """Upsert-ish: ignore if the transaction_id already exists."""
    with SessionLocal() as s:
        exists = s.get(ScoredTransaction, rec["transaction_id"])
        if exists is None:
            s.add(ScoredTransaction(**rec))
            s.commit()


def summary_stats() -> dict:
    with SessionLocal() as s:
        total = s.scalar(select(func.count()).select_from(ScoredTransaction)) or 0
        frauds = s.scalar(
            select(func.count()).select_from(ScoredTransaction)
            .where(ScoredTransaction.status == "HIGH RISK")
        ) or 0
        avg_score = s.scalar(select(func.avg(ScoredTransaction.risk_score))) or 0
    return {
        "total": int(total),
        "high_risk": int(frauds),
        "high_risk_pct": round(100 * frauds / total, 2) if total else 0.0,
        "avg_risk_score": round(float(avg_score), 1),
    }


def recent(limit: int = 50):
    """Return recent scored rows as list[dict], newest first."""
    with SessionLocal() as s:
        rows = s.execute(
            select(ScoredTransaction)
            .order_by(ScoredTransaction.scored_at.desc())
            .limit(limit)
        ).scalars().all()
    return [
        {
            "transaction_id": r.transaction_id,
            "amount": r.amount,
            "fraud_prob": r.fraud_prob,
            "risk_score": r.risk_score,
            "status": r.status,
            "reason": r.reason,
            "scored_at": r.scored_at,
        }
        for r in rows
    ]


if __name__ == "__main__":
    init_db()
    print(f"DB ready at: {DB_URL}")
    print("stats:", summary_stats())
