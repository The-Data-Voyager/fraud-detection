"""
Phase 7 - Broker-free streaming simulator.

Ties producer -> scorer -> database in one process so the full real-time
pipeline runs with no Kafka/Postgres install. Producer thread reads
transactions in TransactionDT order and pushes to a queue; consumer thread
scores each (model + SHAP) and writes to the DB. Swap to real Kafka later by
running stream_producer.py + stream_consumer.py with KAFKA_BOOTSTRAP set --
the scoring and DB code are identical.

Usage:
  python run_stream_sim.py --source test --limit 1000 --delay 0.05 --reset
"""
from __future__ import annotations

import argparse
import queue
import threading
import time

import db
from scorer import StreamScorer
from stream_producer import iter_transactions

_SENTINEL = object()


def producer_thread(q: "queue.Queue", source: str, limit: int | None, delay: float) -> None:
    for rec in iter_transactions(source=source, limit=limit, delay=delay):
        q.put(rec)
    q.put(_SENTINEL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=0.03, help="sec between arrivals")
    ap.add_argument("--reset", action="store_true", help="wipe DB table first")
    ap.add_argument("--genai", action="store_true", help="polish reasons via Claude (needs API creds)")
    args = ap.parse_args()

    db.init_db(reset=args.reset)
    scorer = StreamScorer(with_explainer=True, genai=args.genai)
    if args.genai and scorer.genai is not None and not scorer.genai.available:
        print("(--genai requested but no Anthropic credentials found; using template reasons)")
    q: "queue.Queue" = queue.Queue(maxsize=1000)

    t = threading.Thread(
        target=producer_thread, args=(q, args.source, args.limit, args.delay), daemon=True
    )
    t.start()

    print(f"Streaming {args.limit} '{args.source}' transactions "
          f"(delay={args.delay}s)  ->  DB {db.DB_URL}\n")
    n = 0
    alerts = 0
    t0 = time.time()
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        scored = scorer.score(item)
        db.insert_scored(scored)
        n += 1
        if scored["status"] == "HIGH RISK":
            alerts += 1
            print(f"  [ALERT #{alerts}] txn {scored['transaction_id']}  "
                  f"Rs {scored['amount']:.0f}  risk={scored['risk_score']}/100")
            print(f"            {scored['reason']}")
        if n % 200 == 0:
            print(f"  ... {n} processed ({n/(time.time()-t0):.0f}/s)")

    print(f"\nDone. Processed {n} transactions, {alerts} HIGH-RISK alerts.")
    print("Summary:", db.summary_stats())
    print("\nStart the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
