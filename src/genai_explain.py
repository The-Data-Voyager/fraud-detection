"""
Phase 10 (optional) - GenAI natural-language reason layer (Groq / free LLM).

Takes the SHAP-derived base reason + the top contributing features for a
flagged transaction and asks a free-tier LLM (Groq, running Llama) to phrase
it as a crisp, human sentence for the demo/dashboard. Falls back to the
deterministic template reason when no API key is configured or the call
fails, so the pipeline still runs fully offline.

Setup (free):
  1. Get a free key at https://console.groq.com/keys
  2. set GROQ_API_KEY=...      (PowerShell: $env:GROQ_API_KEY="...")

Enable per run: StreamScorer(..., genai=True) or run_stream_sim.py --genai.
Model via env FRAUD_LLM_MODEL (default llama-3.1-8b-instant - fast & free).
"""
from __future__ import annotations

import os

_MODEL = os.environ.get("FRAUD_LLM_MODEL", "llama-3.1-8b-instant")
_SYSTEM = (
    "You are a fraud-analysis assistant. Given a base explanation and the top "
    "contributing risk factors for a flagged payment, write ONE clear sentence "
    "(max 30 words) a bank investigator could read, starting with 'Reason:'. "
    "Do not invent facts beyond the factors given. No preamble, no lists."
)


class GenAIExplainer:
    def __init__(self) -> None:
        self._client = None
        self._available = False
        if not os.environ.get("GROQ_API_KEY"):
            return
        try:
            from groq import Groq

            self._client = Groq()  # reads GROQ_API_KEY
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def enhance(self, base_reason: str, top_features: list[tuple[str, float]],
                amount: float, risk_score: int) -> str:
        """Return an LLM-polished reason, or base_reason on any failure."""
        if not self._available:
            return base_reason
        feats = ", ".join(f for f, _ in top_features) or "mixed weak signals"
        user = (
            f"Base explanation: {base_reason}\n"
            f"Top risk factors: {feats}\n"
            f"Amount: {amount:.2f}   Risk score: {risk_score}/100\n"
            "Write the one-sentence reason."
        )
        try:
            resp = self._client.chat.completions.create(
                model=_MODEL,
                max_tokens=120,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or base_reason
        except Exception:
            return base_reason


if __name__ == "__main__":
    g = GenAIExplainer()
    print("Groq credentials available:", g.available)
    demo = g.enhance(
        "Transaction flagged because its transaction counting pattern and internal "
        "risk signal significantly differ from normal transaction patterns.",
        [("C1", 2.5), ("V258", 1.3), ("ent_amt_z", 0.9)],
        amount=85000, risk_score=94,
    )
    print("Reason:", demo)
