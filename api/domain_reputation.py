# api/domain_reputation.py
import time
import ssl
import socket
from datetime import datetime
import whois
import dns.resolver
from pathlib import Path
import logging

logger = logging.getLogger("domain_reputation")

SAFE_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "microsoft.com", "amazon.com", "google.com"}

def whois_domain_age(domain: str):
    try:
        w = whois.whois(domain)
        # attempt many fields: creation_date may be list
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        if not cd:
            return None
        if isinstance(cd, str):
            cd = datetime.fromisoformat(cd)
        age_days = (datetime.utcnow() - cd).days
        return max(0, age_days)
    except Exception as e:
        logger.debug("whois fail: %s", e)
        return None

def dns_health_checks(domain: str):
    out = {"has_mx": False, "mx_count": 0, "has_a": False, "has_ns": False}
    try:
        mx = dns.resolver.resolve(domain, "MX", lifetime=5)
        out["has_mx"] = True
        out["mx_count"] = len(mx)
    except Exception:
        pass
    try:
        a = dns.resolver.resolve(domain, "A", lifetime=5)
        out["has_a"] = True
    except Exception:
        pass
    try:
        ns = dns.resolver.resolve(domain, "NS", lifetime=5)
        out["has_ns"] = True
    except Exception:
        pass
    return out

def ssl_certificate_age(hostname: str):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(5)
            s.connect((hostname, 443))
            cert = s.getpeercert()
            not_before = cert.get("notBefore")
            if not_before:
                # format example: 'Jun 25 12:00:00 2024 GMT'
                dt = datetime.strptime(not_before, "%b %d %H:%M:%S %Y %Z")
                age_days = (datetime.utcnow() - dt).days
                return max(0, age_days)
    except Exception:
        return None

def domain_reputation_score(domain: str):
    """
    Lightweight reputation scoring:
      - new domains => risk
      - no MX => risk
      - suspicious TLD => risk (checked earlier)
      - no A or NS => risk
    Returns dict {score, features}
    """
    from tldextract import extract
    te = extract(domain)
    base = f"{te.domain}.{te.suffix}" if te.domain and te.suffix else domain
    score = 0.0
    features = {}
    age = whois_domain_age(base)
    features["domain_age_days"] = age
    if age is None:
        score += 0.25  # unknown whois -> risk
    elif age < 90:
        score += 0.35
    elif age < 365:
        score += 0.15

    dns = dns_health_checks(base)
    features.update(dns)
    if not dns.get("has_mx", False):
        score += 0.2
    if not dns.get("has_a", False):
        score += 0.15
    if not dns.get("has_ns", False):
        score += 0.1

    # SSL check
    ssl_age = ssl_certificate_age(base)
    features["ssl_age_days"] = ssl_age
    if ssl_age is None:
        score += 0.05

    # normalized
    score = max(0.0, min(1.0, score))
    return {"domain": base, "score": float(score), "features": features}
