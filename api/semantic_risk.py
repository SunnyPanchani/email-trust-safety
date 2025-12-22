# api/semantic_risk.py
import os
import logging
from typing import Optional

logger = logging.getLogger("semantic_risk")

# Optional: Groq integration for LLM explanations.
# Groq client usage example is hypothetical; replace with actual client usage per your SDK.
# We'll implement a minimal adapter: if GROQ_API_KEY exists, call the Groq API (user responsible for SDK).
# Otherwise, fallback to lightweight keyword heuristics.

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# Simple keyword-based fallback classifier for 'semantic risk' (0..1)
PROMO_KEYWORDS = {
    "urgent", "final notice", "congratulations", "claim", "exclusive", "winner", "selected",
    "verify", "verify now", "click", "bank details", "free", "limited time", "hurry", "reward"
}

def heuristic_semantic_score(text: str):
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    matches = 0
    for kw in PROMO_KEYWORDS:
        if kw in t:
            matches += 1
    # scale: more matches => more risk
    score = min(1.0, matches * 0.15)
    return float(score)

def llm_semantic_explain(text: str) -> dict:
    """
    If GROQ_API_KEY present, call LLM to get semantic phishing/spam score & explanation.
    Must be implemented against your Groq SDK.
    """
    if not GROQ_API_KEY:
        return {"score": heuristic_semantic_score(text), "explanation": None, "source": "heuristic"}
    try:
        # PSEUDO-CODE: replace with real Groq call
        # from groq import GroqClient
        # client = GroqClient(api_key=GROQ_API_KEY)
        # prompt = f"Classify this email body for phishing/spam intent. Return JSON: {{'score':0-1,'explanation':str}} \n\nEmail:\n{text}"
        # resp = client.run(prompt)
        # Parse resp to get score/explanation
        # Here we simulate:
        simulated = {"score": heuristic_semantic_score(text) * 1.1}
        return {"score": float(min(1.0, simulated["score"])), "explanation": "LLM explanation not enabled in fallback snippet", "source": "groq"}
    except Exception as e:
        logger.exception("LLM call failed, falling back to heuristic")
        return {"score": heuristic_semantic_score(text), "explanation": None, "source": "heuristic-fallback"}
