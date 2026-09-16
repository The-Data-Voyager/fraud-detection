"""
Phase 9 - Streamlit + Plotly dashboard.

Reads scored transactions from the database (db.py) and shows the views the
project brief asks for: total processed, frauds detected, fraud %, live
stream table, risk-score distribution, high-risk alerts, trends, and the
per-transaction explanation.

Run:
  streamlit run dashboard/app.py

Point at the same DB the stream writes to via FRAUD_DB_URL (defaults to the
local SQLite file, so no config needed for the demo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import db  # noqa: E402

st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")

STATUS_COLORS = {"HIGH RISK": "#e63946", "MEDIUM": "#f4a261", "LOW": "#2a9d8f"}


def load_frame(limit: int = 500) -> pd.DataFrame:
    rows = db.recent(limit=limit)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["scored_at"] = pd.to_datetime(df["scored_at"])
    return df


@st.fragment(run_every=2)
def live_view() -> None:
    stats = db.summary_stats()
    df = load_frame(500)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions processed", f"{stats['total']:,}")
    c2.metric("High-risk detected", f"{stats['high_risk']:,}")
    c3.metric("Fraud rate", f"{stats['high_risk_pct']}%")
    c4.metric("Avg risk score", f"{stats['avg_risk_score']}")

    if df.empty:
        st.info("No transactions scored yet. Run:  python src/run_stream_sim.py --reset")
        return

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Live transaction stream")
        show = df[["transaction_id", "amount", "risk_score", "status", "reason", "scored_at"]].head(25)
        st.dataframe(
            show.style.apply(
                lambda r: [f"background-color:{STATUS_COLORS.get(r['status'], '')}22"] * len(r),
                axis=1,
            ),
            use_container_width=True, height=380,
        )

    with right:
        st.subheader("Risk score distribution")
        fig = px.histogram(df, x="risk_score", nbins=20, color="status",
                           color_discrete_map=STATUS_COLORS)
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk score over recent arrivals (trend)")
    trend = df.sort_values("scored_at")
    fig2 = px.scatter(trend, x="scored_at", y="risk_score", color="status",
                      color_discrete_map=STATUS_COLORS, hover_data=["transaction_id", "amount"])
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🚨 High-risk alerts")
    high = df[df["status"] == "HIGH RISK"].head(10)
    if high.empty:
        st.write("No high-risk transactions in the recent window.")
    else:
        for _, r in high.iterrows():
            with st.container(border=True):
                a, b = st.columns([1, 4])
                a.metric("Risk", f"{r['risk_score']}/100")
                b.markdown(
                    f"**Transaction #{r['transaction_id']}** — amount **{r['amount']:.2f}**  \n"
                    f"Status: **{r['status']}**  \n"
                    f"Reason: {r['reason']}"
                )


st.title("💳 AI-Powered Real-Time Fraud Detection")
st.caption("IEEE-CIS · XGBoost + SHAP · streaming scores written to the database, refreshed live")
live_view()
