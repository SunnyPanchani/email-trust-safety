# api/url_threat.py
import re
import math
import ipaddress
from urllib.parse import urlparse, unquote
import tldextract
import numpy as np

URL_REGEX = re.compile(
    r'((?:http|https)://[^\s<>"]+|www\.[^\s<>"]+)',
    flags=re.IGNORECASE
)

SUSPICIOUS_TLDS = {
    "zip","review","top","tk","pw","cn","gq","ru","click","xyz","info","biz","ru","men","work"
}

EXEC_EXT = {".exe", ".scr", ".zip", ".bat", ".js", ".vbs", ".msi", ".cmd"}


def find_urls(text: str):
    """Return list of normalized URLs found in text."""
    if not text:
        return []
    matches = URL_REGEX.findall(text)
    cleaned = []
    for u in matches:
        u = u.strip().rstrip(").,;\"'")
        if u.startswith("www."):
            u = "http://" + u
        cleaned.append(u)
    return cleaned


def _has_ip(hostname: str):
    try:
        # strip port
        host = hostname.split(":")[0]
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def _entropy(s: str):
    if not s:
        return 0.0
    p, lns = {}, len(s)
    for ch in s:
        p[ch] = p.get(ch, 0) + 1
    probs = [v/lns for v in p.values()]
    return -sum(pv * math.log2(pv) for pv in probs)


def analyze_url(url: str):
    """
    Returns dict with features and a url risk score (0-1).
    Heuristics-based (fast, no external calls).
    """
    try:
        parsed = urlparse(unquote(url))
    except Exception:
        return {"url": url, "error": "parse_failed", "score": 0.0, "features": {}}

    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    netloc = parsed.netloc or host

    ext = ""
    for e in EXEC_EXT:
        if path.lower().endswith(e):
            ext = e
            break

    te = tldextract.extract(host)
    subdomain = te.subdomain or ""
    domain = te.domain or ""
    suffix = te.suffix or ""

    # heuristics
    has_ip = _has_ip(host)
    multi_tld = "." in domain and domain.count('.') >= 1 and suffix and "." in suffix and len(suffix.split('.')) >= 2
    suspicious_tld = suffix.lower().split('.')[-1] in SUSPICIOUS_TLDS if suffix else False
    long_url = len(url) > 200
    path_entropy = _entropy(path)
    has_exec = bool(ext)
    suspicious_tokens = 0
    for tok in ["verify", "login", "confirm", "update", "account", "secure", "claim", "reset", "download"]:
        if tok in url.lower():
            suspicious_tokens += 1

    score = 0.0
    # add weights
    if has_ip:
        score += 0.35
    if multi_tld:
        score += 0.25
    if suspicious_tld:
        score += 0.15
    if long_url:
        score += 0.05
    if path_entropy > 3.5:
        score += 0.10
    if has_exec:
        score += 0.25
    score += min(0.05 * suspicious_tokens, 0.25)

    # clamp
    score = max(0.0, min(1.0, score))
    features = {
        "host": host,
        "domain": f"{domain}.{suffix}" if domain and suffix else domain or host,
        "subdomain": subdomain,
        "has_ip": bool(has_ip),
        "multi_tld": bool(multi_tld),
        "suspicious_tld": bool(suspicious_tld),
        "long_url": bool(long_url),
        "path_entropy": float(path_entropy),
        "has_exec": bool(has_exec),
        "suspicious_tokens": suspicious_tokens,
    }
    return {"url": url, "score": float(score), "features": features}


def url_list_score(text: str):
    """Analyze all URLs in text, return max score and list of dicts"""
    urls = find_urls(text)
    if not urls:
        return 0.0, []
    results = [analyze_url(u) for u in urls]
    max_score = max(r["score"] for r in results)
    return float(max_score), results
