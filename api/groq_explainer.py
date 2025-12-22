import os
import logging
from typing import Dict, Any
logger = logging.getLogger("api.groq_explainer")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

class GroqExplainer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = None
        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized")
            except Exception as e:
                logger.exception("Groq init failed")
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def explain(self, context: Dict[str, Any]) -> str:
        """
        Build a prompt from context and call Groq LLM to generate a concise explanation.
        We must NOT send PII — predictor.prepare_explainer_context redacts emails.
        """
        if not self.client:
            return None
        prompt = (
            "You are a security analyst. Given the email subject, snippet and metadata, "
            "write a short 2-4 bullet explanation why this email looks suspicious or safe.\n\n"
            f"Subject: {context.get('subject')}\n\n"
            f"Body snippet: {context.get('body_snippet')}\n\n"
            f"Signals: {context.get('metadata')}\n\n"
            f"Additional reasons: {context.get('reasons')}\n\n"
            "Answer in plain English with 3 bullets, avoid revealing any PII."
        )
        try:
            resp = self.client.chat.completions.create(
                model="llama3-8b-8192",  # or smaller model available
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = resp.choices[0].message.content
            return text
        except Exception as e:
            logger.exception("Groq explain failed")
            return None
