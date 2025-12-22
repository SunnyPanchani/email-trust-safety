import re
def redact_pii(text: str) -> str:
    if not text:
        return text
    # redact emails and long tokens
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b[0-9]{6,}\b", "[REDACTED_NUMBER]", text)
    return text
