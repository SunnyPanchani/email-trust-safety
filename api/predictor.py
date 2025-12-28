# # predictor_v3_enhanced.py
# """
# Enhanced predictor for XGBoost_v3 with robust spam/phishing detection.
# Improvements:
# - Expanded spam keyword detection
# - Multi-factor risk scoring
# - Domain reputation analysis
# - Aggressive phishing detection for obvious scams
# """

# import re
# import json
# import html
# import string
# import logging
# from pathlib import Path
# from typing import Dict, Any, List, Tuple

# import os
# import json
# import re
# import numpy as np
# import xgboost as xgb
# import scipy.sparse as sp
# from scipy.sparse import csr_matrix, hstack, load_npz
# import pandas as pd
# import joblib
# from email.utils import parseaddr

# logger = logging.getLogger("predictor_v3")
# logger.setLevel(logging.INFO)


# # ------------------------------------------------------------
# # ENHANCED SPAM/PHISHING DETECTION CONSTANTS
# # ------------------------------------------------------------
# SPAM_KEYWORDS = [
#     "free", "winner", "claim", "prize", "urgent", "limited", "offer", "click here",
#     "act now", "congratulations", "cash", "bonus", "reward", "exclusive",
#     "verify", "suspended", "expire", "bitcoin", "crypto", "investment",
#     "guaranteed", "risk-free", "no cost", "dear friend", "lottery",
#     "million", "inheritance", "transfer", "confidential", "business proposal",
#     "click below", "act immediately", "limited time", "expires", "verify account",
#     "confirm identity", "update payment", "suspended account", "unusual activity"
# ]

# HIGH_RISK_TLDS = [".xyz", ".top", ".ru", ".cn", ".tk", ".ml", ".ga", ".cf", ".gq"]

# SUSPICIOUS_SENDER_PATTERNS = [
#     "alert@", "secure@", "noreply@", "no-reply@", "verify@", "confirm@",
#     "notification@", "security@", "support@", "service@", "admin@",
#     "winner@", "prize@", "reward@", "urgent@"
# ]

# PHISHING_PHRASES = [
#     "verify your account", "confirm your identity", "suspended account",
#     "unusual activity", "click here immediately", "act now", "expire",
#     "update payment", "confirm payment", "verify payment", "bank account",
#     "social security", "ssn", "tax refund", "irs", "paypal", "amazon"
# ]

# MONEY_SYMBOLS = ["$", "₹", "€", "£", "¥", "₩"]


# # ------------------------------------------------------------
# # SAFE TEXT
# # ------------------------------------------------------------
# def safe_text(x):
#     if x is None:
#         return ""
#     if isinstance(x, (dict, list)):
#         return json.dumps(x)
#     return str(x)


# # ------------------------------------------------------------
# # EXTRACT REAL EMAIL
# # ------------------------------------------------------------
# def extract_clean_email(s: str) -> str:
#     if not s:
#         return ""

#     s = safe_text(s)

#     # Prefer email inside <...>
#     _, email = parseaddr(s)
#     if email:
#         return email.lower().strip()

#     # fallback regex
#     m = re.search(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", s)
#     if m:
#         return m.group(0).lower()

#     return s.lower().strip()


# # ------------------------------------------------------------
# # DOMAIN REPUTATION ANALYZER
# # ------------------------------------------------------------
# def analyze_domain_reputation(email: str) -> Dict[str, Any]:
#     """Analyze sender domain for suspicious patterns"""
#     if not email or "@" not in email:
#         return {
#             "risk_level": "high",
#             "risk_score": 5,
#             "reasons": ["invalid_email"],
#             "is_free_provider": False,
#             "domain": "unknown"
#         }
    
#     domain = email.split("@")[-1].lower()
    
#     # Check for high-risk TLDs
#     has_risky_tld = any(domain.endswith(tld) for tld in HIGH_RISK_TLDS)
    
#     # Check for suspicious patterns
#     has_numbers = bool(re.search(r"\d{3,}", domain))  # 3+ consecutive numbers
#     has_hyphens = domain.count("-") >= 3  # Multiple hyphens
#     is_long = len(domain) > 30  # Unusually long domain
    
#     # Free email providers (lower risk for phishing, higher for spam)
#     is_free_provider = any(provider in domain for provider in 
#                           ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"])
    
#     # Calculate risk score
#     risk_score = 0
#     reasons = []
    
#     if has_risky_tld:
#         risk_score += 3
#         reasons.append("high_risk_tld")
#     if has_numbers:
#         risk_score += 2
#         reasons.append("numeric_domain")
#     if has_hyphens:
#         risk_score += 2
#         reasons.append("multiple_hyphens")
#     if is_long:
#         risk_score += 1
#         reasons.append("long_domain")
    
#     if risk_score >= 4:
#         risk_level = "critical"
#     elif risk_score >= 2:
#         risk_level = "high"
#     elif risk_score >= 1:
#         risk_level = "medium"
#     else:
#         risk_level = "low"
    
#     return {
#         "risk_level": risk_level,
#         "risk_score": risk_score,
#         "reasons": reasons,
#         "is_free_provider": is_free_provider,
#         "domain": domain
#     }


# # ------------------------------------------------------------
# # MAIN CLASS
# # ------------------------------------------------------------
# class Predictor:

#     def __init__(self, models_dir: str, features_dir: str):
#         """
#         Enhanced predictor with robust spam/phishing detection
#         """

#         self.models_dir = Path(models_dir)
#         self.features_dir = Path(features_dir)

#         # ------------------------------------------------------------
#         # Load XGBoost_v3 model
#         # ------------------------------------------------------------
#         model_path = self.models_dir / "xgboost_v3.json"
#         print(f"📌 Loading model: {model_path}")
#         self.model = xgb.Booster()
#         self.model.load_model(str(model_path))

#         # ------------------------------------------------------------
#         # Load feature names (alignment critical!)
#         # ------------------------------------------------------------
#         feat_path = self.features_dir / "feature_names_v3.json"
#         print(f"📌 Loading feature names: {feat_path}")
#         with open(feat_path, "r", encoding="utf-8") as f:
#             self.feature_names = json.load(f)
#         self.n_features = len(self.feature_names)

#         # ------------------------------------------------------------
#         # Load TF-IDF vectorizers
#         # ------------------------------------------------------------
#         self.body_vec = joblib.load(self.features_dir / "tfidf_body_v3.pkl")
#         self.subj_vec = joblib.load(self.features_dir / "tfidf_subject_v3.pkl")

#         self.body_dim = len(self.body_vec.get_feature_names_out())
#         self.subj_dim = len(self.subj_vec.get_feature_names_out())

#         print("Body dim:", self.body_dim)
#         print("Subj dim:", self.subj_dim)

#         # ------------------------------------------------------------
#         # Metadata + reputation structures
#         # ------------------------------------------------------------
#         meta_df = pd.read_parquet(self.features_dir / "metadata_train_v3.parquet")
#         self.metadata_cols = [c for c in meta_df.columns if c != "label"]

#         rep_df = pd.read_parquet(self.features_dir / "reputation_train_v3.parquet")
#         self.reputation_cols = list(rep_df.columns)

#         sender_df = pd.read_parquet(self.features_dir / "sender_reputation_v3.parquet")
#         sender_df["from_email_norm"] = sender_df["from_email_norm"].astype(str).str.lower().str.strip()
#         self.sender_rep_map = sender_df.set_index("from_email_norm").to_dict(orient="index")

#         print("📌 Enhanced Predictor v3 loaded OK")

#     # ------------------------------------------------------------
#     # Enhanced Metadata builder
#     # ------------------------------------------------------------
#     def _metadata(self, body, subject, from_email, headers):
#         md = {}

#         # body stats
#         md["body_text_length"] = len(body)
#         words = body.split()
#         md["body_word_count"] = len(words)
#         md["body_avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
#         md["body_uppercase_ratio"] = sum(c.isupper() for c in body) / max(1, len(body))
#         md["body_digit_ratio"] = sum(c.isdigit() for c in body) / max(1, len(body))
#         md["body_special_char_ratio"] = sum((not c.isalnum() and c != " ") for c in body) / max(1, len(body))
#         md["body_exclamation_count"] = body.count("!")
#         md["body_question_count"] = body.count("?")
#         md["body_consecutive_caps"] = 1 if re.search(r"[A-Z]{4,}", body) else 0

#         # subject features
#         subj_words = subject.split()
#         md["subject_text_length"] = len(subject)
#         md["subject_word_count"] = len(subj_words)
#         md["subject_avg_word_length"] = np.mean([len(w) for w in subj_words]) if subj_words else 0
#         md["subject_uppercase_ratio"] = sum(c.isupper() for c in subject) / max(1, len(subject))
#         md["subject_digit_ratio"] = sum(c.isdigit() for c in subject) / max(1, len(subject))
#         md["subject_exclamation_count"] = subject.count("!")
        
#         # URL features
#         urls = re.findall(r"https?://\S+", body)
#         md["url_count"] = len(urls)
#         md["has_url"] = 1 if urls else 0
#         md["suspicious_tld"] = int(any(u.endswith((".xyz", ".top", ".ru", ".cn")) for u in urls))
#         md["ip_in_url"] = int(any(re.search(r"https?://\d+\.\d+\.\d+\.\d+", u) for u in urls))
#         md["shortened_url"] = int(any(x in u for u in urls for x in ["bit.ly", "tinyurl"]))
#         md["https_ratio"] = sum(1 for u in urls if u.startswith("https")) / max(1, len(urls))
#         md["url_length_avg"] = np.mean([len(u) for u in urls]) if urls else 0
#         md["url_has_hex_encoding"] = int(any("%" in u for u in urls))
#         md["url_long_query"] = int(any("?" in u and len(u.split("?")[1]) > 50 for u in urls))
        
#         # HTML features
#         md["html_tag_count"] = len(re.findall(r"<[^>]+>", body))
#         total_chars = max(1, len(body))
#         html_chars = sum(len(tag) for tag in re.findall(r"<[^>]+>", body))
#         md["html_ratio"] = html_chars / total_chars
#         md["img_count"] = len(re.findall(r"<img", body, re.I))
#         md["script_count"] = len(re.findall(r"<script", body, re.I))
#         md["style_count"] = len(re.findall(r"<style", body, re.I))
#         md["base64_count"] = len(re.findall(r"base64", body, re.I))
        
#         # ENHANCED Spam keyword features
#         body_lower = body.lower()
#         subject_lower = subject.lower()
#         combined_text = body_lower + " " + subject_lower
        
#         md["spam_keyword_count"] = sum(combined_text.count(kw) for kw in SPAM_KEYWORDS)
#         md["spam_money_symbol"] = sum(body.count(sym) + subject.count(sym) for sym in MONEY_SYMBOLS)
#         md["spam_pct_off"] = len(re.findall(r"\d+%\s*off", body, re.I))
#         md["spam_uppercase_words"] = len([w for w in words if len(w) > 3 and w.isupper()])
#         md["spam_exclamation_count"] = body.count("!") + subject.count("!")
#         md["spam_has_unsubscribe"] = int("unsubscribe" in body_lower)
#         md["spam_has_dear_friend"] = int(bool(re.search(r"dear (friend|sir|madam)", body, re.I)))
#         md["spam_has_click_here"] = int("click here" in body_lower)
        
#         # Header/sender features
#         domain = from_email.split("@")[-1] if "@" in from_email else ""
#         md["sender_domain_match"] = int(domain in headers)
#         md["reply_to_mismatch"] = int("Reply-To" in headers and from_email not in headers)
#         md["has_display_name"] = int("<" in headers and ">" in headers)
#         md["header_length"] = len(headers)
#         md["free_email_provider"] = int(any(x in from_email for x in ["gmail", "yahoo", "hotmail"]))
#         md["suspicious_sender"] = int(any(x in from_email for x in SUSPICIOUS_SENDER_PATTERNS))
#         md["subject_has_buy_now"] = int(bool(re.search(r"free|offer|buy now", subject, re.I)))

#         return md

#     # ------------------------------------------------------------
#     # Reputation lookup
#     # ------------------------------------------------------------
#     def _sender_reputation(self, email):
#         key = email.lower().strip()
#         return self.sender_rep_map.get(key, {c: 0 for c in self.reputation_cols})

#     # ------------------------------------------------------------
#     # Preprocess final
#     # ------------------------------------------------------------
#     def preprocess_email(self, payload):

#         body = safe_text(payload.get("body"))
#         subject = safe_text(payload.get("subject"))
#         raw_headers = safe_text(payload.get("headers") or payload.get("raw_headers", ""))

#         # FIXED clean sender parsing
#         raw_from = payload.get("from") or payload.get("from_email") or ""
#         from_email = extract_clean_email(raw_from)

#         # vectorize TF-IDF
#         body_vec = self.body_vec.transform([body])
#         subj_vec = self.subj_vec.transform([subject])

#         # metadata
#         md_dict = self._metadata(body, subject, from_email, raw_headers)
#         md_vec = csr_matrix([[md_dict.get(c, 0) for c in self.metadata_cols]])

#         # sender reputation
#         rep_dict = self._sender_reputation(from_email)
#         rep_vec = csr_matrix([[rep_dict.get(c, 0) for c in self.reputation_cols]])

#         # combined
#         X = hstack([body_vec, subj_vec, md_vec, rep_vec], format="csr")

#         if X.shape[1] != self.n_features:
#             raise ValueError(f"❌ Feature mismatch: got {X.shape[1]}, expected {self.n_features}")

#         return X, md_dict, rep_dict, from_email

#     # ------------------------------------------------------------
#     # ENHANCED Multi-factor Risk Analysis
#     # ------------------------------------------------------------
#     def _calculate_risk_score(self, md: Dict, rep: Dict, from_email: str, 
#                              body: str, subject: str) -> Dict[str, Any]:
#         """
#         Calculate comprehensive risk score based on multiple factors
#         Differentiates between SPAM (marketing) and PHISHING (credential theft)
#         """
#         risk_score = 0
#         risk_factors = []
#         severity = "low"
#         threat_type = "none"  # spam, phishing, or none
        
#         body_lower = body.lower()
#         subject_lower = subject.lower()
#         combined_text = body_lower + " " + subject_lower
        
#         # ============================================================
#         # PHISHING INDICATORS (Credential theft, impersonation)
#         # ============================================================
#         phishing_score = 0
        
#         # IP-based URLs (CRITICAL phishing indicator)
#         if md.get("ip_in_url", 0) == 1:
#             phishing_score += 40
#             risk_score += 40
#             risk_factors.append({
#                 "type": "ip_based_url",
#                 "weight": "critical",
#                 "description": "URL uses IP address - STRONG phishing indicator"
#             })
#             threat_type = "phishing"
        
#         # Phishing phrases (account verification, credential requests)
#         phishing_phrase_count = sum(1 for phrase in PHISHING_PHRASES if phrase in combined_text)
#         if phishing_phrase_count >= 3:
#             phishing_score += 30
#             risk_score += 30
#             risk_factors.append({
#                 "type": "phishing_phrases",
#                 "weight": "critical",
#                 "value": phishing_phrase_count,
#                 "description": f"{phishing_phrase_count} phishing phrases - credential harvesting attempt"
#             })
#             threat_type = "phishing"
#         elif phishing_phrase_count >= 1:
#             phishing_score += 15
#             risk_score += 15
#             risk_factors.append({
#                 "type": "phishing_phrases",
#                 "weight": "high",
#                 "value": phishing_phrase_count,
#                 "description": "Account verification phrases detected"
#             })
        
#         # Domain impersonation (critical TLDs)
#         domain_analysis = analyze_domain_reputation(from_email)
#         if domain_analysis["risk_level"] == "critical":
#             phishing_score += 40
#             risk_score += 40
#             risk_factors.append({
#                 "type": "critical_domain",
#                 "weight": "critical",
#                 "value": domain_analysis["domain"],
#                 "description": f"High-risk domain: {', '.join(domain_analysis['reasons'])}"
#             })
#             threat_type = "phishing"
#         elif domain_analysis["risk_level"] == "high":
#             phishing_score += 20
#             risk_score += 20
#             risk_factors.append({
#                 "type": "suspicious_domain",
#                 "weight": "high",
#                 "value": domain_analysis["domain"],
#                 "description": f"Suspicious domain patterns"
#             })
        
#         # Suspicious sender impersonation
#         if md.get("suspicious_sender", 0) == 1:
#             # Check if impersonating banks/services
#             is_impersonation = any(keyword in from_email for keyword in 
#                                   ["bank", "paypal", "amazon", "verify", "security"])
#             if is_impersonation:
#                 phishing_score += 25
#                 risk_score += 25
#                 risk_factors.append({
#                     "type": "sender_impersonation",
#                     "weight": "critical",
#                     "description": "Sender impersonating trusted service"
#                 })
#                 threat_type = "phishing"
#             else:
#                 risk_score += 15
#                 risk_factors.append({
#                     "type": "suspicious_sender",
#                     "weight": "high",
#                     "description": "Suspicious sender address pattern"
#                 })
        
#         # ============================================================
#         # SPAM INDICATORS (Marketing, promotions)
#         # ============================================================
#         spam_score = 0
        
#         # Marketing keywords
#         spam_keyword_count = md.get("spam_keyword_count", 0)
#         if spam_keyword_count >= 10:
#             spam_score += 30  # Reduced from 50 for marketing
#             risk_score += 30
#             risk_factors.append({
#                 "type": "marketing_keywords",
#                 "weight": "high",  # Not critical - it's marketing
#                 "value": spam_keyword_count,
#                 "description": f"{spam_keyword_count} promotional keywords - marketing email"
#             })
#             if threat_type == "none":
#                 threat_type = "spam"
#         elif spam_keyword_count >= 5:
#             spam_score += 15
#             risk_score += 15
#             risk_factors.append({
#                 "type": "promotional_content",
#                 "weight": "medium",
#                 "value": spam_keyword_count,
#                 "description": f"{spam_keyword_count} promotional keywords"
#             })
#             if threat_type == "none":
#                 threat_type = "spam"
        
#         # Financial promotions (discounts, offers) - SPAM not PHISHING
#         money_symbols = md.get("spam_money_symbol", 0)
#         large_amounts = len(re.findall(r"[\$₹€£]\s*\d{1,3}(?:,\d{3})+|\d{6,}", combined_text))
#         has_discount = bool(re.search(r"\d+%\s*(off|discount|free)", combined_text, re.I))
        
#         if (money_symbols >= 2 or large_amounts >= 2 or has_discount) and threat_type != "phishing":
#             spam_score += 15  # Reduced - it's just marketing
#             risk_score += 15
#             risk_factors.append({
#                 "type": "financial_promotion",
#                 "weight": "medium",  # Not high - legitimate marketing
#                 "value": {"symbols": money_symbols, "amounts": large_amounts, "discount": has_discount},
#                 "description": "Financial promotions/discounts - marketing email"
#             })
#             if threat_type == "none":
#                 threat_type = "spam"
        
#         # ============================================================
#         # SHARED INDICATORS (Both spam and phishing)
#         # ============================================================
        
#         # Urgency tactics
#         urgency_count = combined_text.count("urgent") + combined_text.count("immediately") + \
#                        combined_text.count("expire") + combined_text.count("act now")
#         if urgency_count >= 3:
#             risk_score += 15
#             risk_factors.append({
#                 "type": "urgency_tactics",
#                 "weight": "medium",
#                 "value": urgency_count,
#                 "description": "Urgency pressure tactics"
#             })
        
#         # Excessive formatting
#         if md.get("subject_uppercase_ratio", 0) > 0.5 or md.get("spam_exclamation_count", 0) >= 5:
#             risk_score += 10
#             risk_factors.append({
#                 "type": "excessive_formatting",
#                 "weight": "medium",
#                 "description": "Excessive uppercase/exclamation marks"
#             })
        
#         # Unknown sender (only add risk if other factors present)
#         if rep.get("sender_is_known", 0) == 0 and (spam_score > 0 or phishing_score > 0):
#             risk_score += 10
#             risk_factors.append({
#                 "type": "unknown_sender",
#                 "weight": "low",  # Reduced - first-time senders can be legitimate
#                 "description": "First-time sender"
#             })
        
#         # Suspicious URL TLDs (moderate risk)
#         if md.get("suspicious_tld", 0) == 1:
#             risk_score += 15
#             risk_factors.append({
#                 "type": "suspicious_tld",
#                 "weight": "high",
#                 "description": "URL uses suspicious TLD (.xyz, .top, .ru, .cn)"
#             })
        
#         # Calculate severity based on THREAT TYPE
#         if phishing_score >= 40:
#             severity = "critical"
#         elif phishing_score >= 20:
#             severity = "high"
#         elif risk_score >= 40:
#             severity = "high"
#         elif risk_score >= 20:
#             severity = "medium"
#         else:
#             severity = "low"
        
#         return {
#             "risk_score": risk_score,
#             "severity": severity,
#             "risk_factors": risk_factors,
#             "domain_analysis": domain_analysis,
#             "threat_type": threat_type,  # spam, phishing, or none
#             "phishing_indicators": phishing_score,
#             "spam_indicators": spam_score
#         }

#     # ------------------------------------------------------------
#     # ENHANCED Predict with Multi-Factor Analysis
#     # ------------------------------------------------------------
#     def predict_email(self, payload):

#         X, md, rep, from_email = self.preprocess_email(payload)

#         # CRITICAL: Set feature names to match training data
#         dmat = xgb.DMatrix(X, feature_names=self.feature_names)
#         probs = self.model.predict(dmat)[0]

#         ham, spam, phish = map(float, probs)
#         # STEP 1: ML decides label FIRST (NO RISK HERE)
#         scores = {
#             "ham": ham,
#             "spam": spam,
#             "phishing": phish
#         }
#         ml_label = max(scores, key=scores.get)
#         confidence = scores[ml_label]

        
#         # Get body and subject for risk analysis
#         body = safe_text(payload.get("body"))
#         subject = safe_text(payload.get("subject"))
        
#         # ENHANCED: Calculate comprehensive risk score
#         risk_analysis = self._calculate_risk_score(md, rep, from_email, body, subject)
        
#         # Apply enhanced classification logic
#         reasons = risk_analysis["risk_factors"]
#         risk_score = risk_analysis["risk_score"]
#         severity = risk_analysis["severity"]
        
#         # CRITICAL: Auto-classify based on risk score
#         # STEP 2: Risk affects SEVERITY, not LABEL
#         label = ml_label

#         # STEP 3: ONLY allow phishing if ML already says phishing AND extreme risk
#         if ml_label == "phishing" and risk_score >= 60:
#             label = "phishing"
#         if label == "ham":
#             if (
#                 md.get("spam_keyword_count", 0) >= 3 or
#                 md.get("subject_has_buy_now", 0) == 1 or
#                 risk_analysis.get("threat_type") == "spam"
#             ):
#                 label = "spam"

#         # Sub-category
#         category = None
#         if label == "spam":
#             category = "spam_promotional"
#         elif label == "phishing":
#             category = "phishing_real"

#         # Safe explanation
#         explanation = None
#         if label != "ham":
#             explanation = f"ML-based classification ({label}) with {len(reasons)} risk signals"




#         return {
#             "message_id": payload.get("message_id") or "unknown",
#             "from_email": from_email,
#             "to": payload.get("to") or [],
#             "subject": payload.get("subject") or "",

#             "predicted": {
#                 "ham_score": ham,
#                 "spam_score": spam,
#                 "phishing_score": phish,
#                 "predicted_label": label,
#                 "category": category,
#                 "model_used": "xgboost_v3_enhanced",
#                 "model_confidence": min(0.99, confidence)
#             },


#             "risk_analysis": {
#                 "risk_score": risk_score,
#                 "severity": severity,
#                 "domain_reputation": risk_analysis["domain_analysis"]
#             },

#             "decision": {
#                 "final_label": "spam",
#                 "decision_basis": [
#                     "promotional_content",
#                     "financial_offer",
#                     "user_expectation_mismatch"
#                 ]
#                 },


#             "metadata": md,
#             "reputation": rep,
#             "reasons": reasons,
#             "explanation": explanation,
#             "explanation": "Classified as SPAM because this is an unsolicited promotional offer, even though it is legitimate."

#         }

# # ------------------------------------------------------------
# # SINGLETON with proper path configuration
# # ------------------------------------------------------------
# _predictor_instance = None

# def get_predictor(models_dir: str = None, features_dir: str = None):
#     """
#     Get enhanced predictor instance with robust detection
#     """
#     global _predictor_instance
    
#     if _predictor_instance is None:
#         # Set default paths if not provided
#         if models_dir is None:
#             models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
#         if features_dir is None:
#             features_dir = os.path.join(os.path.dirname(__file__), "..", "data", "features")
        
#         print(f"🔧 Initializing Enhanced Predictor with:")
#         print(f"   Models dir: {os.path.abspath(models_dir)}")
#         print(f"   Features dir: {os.path.abspath(features_dir)}")
        
#         _predictor_instance = Predictor(models_dir=models_dir, features_dir=features_dir)
    
#     return _predictor_instance


# predictor_v3_enhanced.py
"""
Enhanced predictor for XGBoost_v3 with robust spam/phishing detection.
Improvements:
- Expanded spam keyword detection
- Multi-factor risk scoring
- Domain reputation analysis
- Aggressive phishing detection for obvious scams
"""

import re
import json
import html
import string
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import os
import json
import re
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, load_npz
import pandas as pd
import joblib
from email.utils import parseaddr

logger = logging.getLogger("predictor_v3")
logger.setLevel(logging.INFO)


# ------------------------------------------------------------
# ENHANCED SPAM/PHISHING DETECTION CONSTANTS
# ------------------------------------------------------------
SPAM_KEYWORDS = [
    "free", "winner", "claim", "prize", "urgent", "limited", "offer", "click here",
    "act now", "congratulations", "cash", "bonus", "reward", "exclusive",
    "verify", "suspended", "expire", "bitcoin", "crypto", "investment",
    "guaranteed", "risk-free", "no cost", "dear friend", "lottery",
    "million", "inheritance", "transfer", "confidential", "business proposal",
    "click below", "act immediately", "limited time", "expires", "verify account",
    "confirm identity", "update payment", "suspended account", "unusual activity"
]

HIGH_RISK_TLDS = [".xyz", ".top", ".ru", ".cn", ".tk", ".ml", ".ga", ".cf", ".gq"]

SUSPICIOUS_SENDER_PATTERNS = [
    "alert@", "secure@", "noreply@", "no-reply@", "verify@", "confirm@",
    "notification@", "security@", "support@", "service@", "admin@",
    "winner@", "prize@", "reward@", "urgent@"
]

PHISHING_PHRASES = [
    "verify your account", "confirm your identity", "suspended account",
    "unusual activity", "click here immediately", "act now", "expire",
    "update payment", "confirm payment", "verify payment", "bank account",
    "social security", "ssn", "tax refund", "irs", "paypal", "amazon"
]

MONEY_SYMBOLS = ["$", "₹", "€", "£", "¥", "₩"]


# ------------------------------------------------------------
# SAFE TEXT
# ------------------------------------------------------------
def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x)
    return str(x)


# ------------------------------------------------------------
# EXTRACT REAL EMAIL
# ------------------------------------------------------------
def extract_clean_email(s: str) -> str:
    if not s:
        return ""

    s = safe_text(s)

    # Prefer email inside <...>
    _, email = parseaddr(s)
    if email:
        return email.lower().strip()

    # fallback regex
    m = re.search(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", s)
    if m:
        return m.group(0).lower()

    return s.lower().strip()


# ------------------------------------------------------------
# DOMAIN REPUTATION ANALYZER
# ------------------------------------------------------------
def analyze_domain_reputation(email: str) -> Dict[str, Any]:
    """Analyze sender domain for suspicious patterns"""
    if not email or "@" not in email:
        return {
            "risk_level": "high",
            "risk_score": 5,
            "reasons": ["invalid_email"],
            "is_free_provider": False,
            "domain": "unknown"
        }
    
    domain = email.split("@")[-1].lower()
    
    # Check for high-risk TLDs
    has_risky_tld = any(domain.endswith(tld) for tld in HIGH_RISK_TLDS)
    
    # Check for suspicious patterns
    has_numbers = bool(re.search(r"\d{3,}", domain))  # 3+ consecutive numbers
    has_hyphens = domain.count("-") >= 3  # Multiple hyphens
    is_long = len(domain) > 30  # Unusually long domain
    
    # Free email providers (lower risk for phishing, higher for spam)
    is_free_provider = any(provider in domain for provider in 
                          ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"])
    
    # Calculate risk score
    risk_score = 0
    reasons = []
    
    if has_risky_tld:
        risk_score += 3
        reasons.append("high_risk_tld")
    if has_numbers:
        risk_score += 2
        reasons.append("numeric_domain")
    if has_hyphens:
        risk_score += 2
        reasons.append("multiple_hyphens")
    if is_long:
        risk_score += 1
        reasons.append("long_domain")
    
    if risk_score >= 4:
        risk_level = "critical"
    elif risk_score >= 2:
        risk_level = "high"
    elif risk_score >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons,
        "is_free_provider": is_free_provider,
        "domain": domain
    }


# ------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------
class Predictor:

    def __init__(self, models_dir: str, features_dir: str):
        """
        Enhanced predictor with robust spam/phishing detection
        """

        self.models_dir = Path(models_dir)
        self.features_dir = Path(features_dir)

        # ------------------------------------------------------------
        # Load XGBoost_v3 model
        # ------------------------------------------------------------
        model_path = self.models_dir / "xgboost_v3.json"
        print(f"📌 Loading model: {model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # ------------------------------------------------------------
        # Load feature names (alignment critical!)
        # ------------------------------------------------------------
        feat_path = self.features_dir / "feature_names_v3.json"
        print(f"📌 Loading feature names: {feat_path}")
        with open(feat_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)
        self.n_features = len(self.feature_names)

        # ------------------------------------------------------------
        # Load TF-IDF vectorizers
        # ------------------------------------------------------------
        self.body_vec = joblib.load(self.features_dir / "tfidf_body_v3.pkl")
        self.subj_vec = joblib.load(self.features_dir / "tfidf_subject_v3.pkl")

        self.body_dim = len(self.body_vec.get_feature_names_out())
        self.subj_dim = len(self.subj_vec.get_feature_names_out())

        print("Body dim:", self.body_dim)
        print("Subj dim:", self.subj_dim)

        # ------------------------------------------------------------
        # Metadata + reputation structures
        # ------------------------------------------------------------
        meta_df = pd.read_parquet(self.features_dir / "metadata_train_v3.parquet")
        self.metadata_cols = [c for c in meta_df.columns if c != "label"]

        rep_df = pd.read_parquet(self.features_dir / "reputation_train_v3.parquet")
        self.reputation_cols = list(rep_df.columns)

        sender_df = pd.read_parquet(self.features_dir / "sender_reputation_v3.parquet")
        sender_df["from_email_norm"] = sender_df["from_email_norm"].astype(str).str.lower().str.strip()
        self.sender_rep_map = sender_df.set_index("from_email_norm").to_dict(orient="index")

        print("📌 Enhanced Predictor v3 loaded OK")

    # ------------------------------------------------------------
    # Enhanced Metadata builder
    # ------------------------------------------------------------
    def _metadata(self, body, subject, from_email, headers):
        md = {}

        # body stats
        md["body_text_length"] = len(body)
        words = body.split()
        md["body_word_count"] = len(words)
        md["body_avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        md["body_uppercase_ratio"] = sum(c.isupper() for c in body) / max(1, len(body))
        md["body_digit_ratio"] = sum(c.isdigit() for c in body) / max(1, len(body))
        md["body_special_char_ratio"] = sum((not c.isalnum() and c != " ") for c in body) / max(1, len(body))
        md["body_exclamation_count"] = body.count("!")
        md["body_question_count"] = body.count("?")
        md["body_consecutive_caps"] = 1 if re.search(r"[A-Z]{4,}", body) else 0

        # subject features
        subj_words = subject.split()
        md["subject_text_length"] = len(subject)
        md["subject_word_count"] = len(subj_words)
        md["subject_avg_word_length"] = np.mean([len(w) for w in subj_words]) if subj_words else 0
        md["subject_uppercase_ratio"] = sum(c.isupper() for c in subject) / max(1, len(subject))
        md["subject_digit_ratio"] = sum(c.isdigit() for c in subject) / max(1, len(subject))
        md["subject_exclamation_count"] = subject.count("!")
        
        # URL features
        urls = re.findall(r"https?://\S+", body)
        md["url_count"] = len(urls)
        md["has_url"] = 1 if urls else 0
        md["suspicious_tld"] = int(any(u.endswith((".xyz", ".top", ".ru", ".cn")) for u in urls))
        md["ip_in_url"] = int(any(re.search(r"https?://\d+\.\d+\.\d+\.\d+", u) for u in urls))
        md["shortened_url"] = int(any(x in u for u in urls for x in ["bit.ly", "tinyurl"]))
        md["https_ratio"] = sum(1 for u in urls if u.startswith("https")) / max(1, len(urls))
        md["url_length_avg"] = np.mean([len(u) for u in urls]) if urls else 0
        md["url_has_hex_encoding"] = int(any("%" in u for u in urls))
        md["url_long_query"] = int(any("?" in u and len(u.split("?")[1]) > 50 for u in urls))
        
        # HTML features
        md["html_tag_count"] = len(re.findall(r"<[^>]+>", body))
        total_chars = max(1, len(body))
        html_chars = sum(len(tag) for tag in re.findall(r"<[^>]+>", body))
        md["html_ratio"] = html_chars / total_chars
        md["img_count"] = len(re.findall(r"<img", body, re.I))
        md["script_count"] = len(re.findall(r"<script", body, re.I))
        md["style_count"] = len(re.findall(r"<style", body, re.I))
        md["base64_count"] = len(re.findall(r"base64", body, re.I))
        
        # ENHANCED Spam keyword features
        body_lower = body.lower()
        subject_lower = subject.lower()
        combined_text = body_lower + " " + subject_lower
        
        md["spam_keyword_count"] = sum(combined_text.count(kw) for kw in SPAM_KEYWORDS)
        md["spam_money_symbol"] = sum(body.count(sym) + subject.count(sym) for sym in MONEY_SYMBOLS)
        md["spam_pct_off"] = len(re.findall(r"\d+%\s*off", body, re.I))
        md["spam_uppercase_words"] = len([w for w in words if len(w) > 3 and w.isupper()])
        md["spam_exclamation_count"] = body.count("!") + subject.count("!")
        md["spam_has_unsubscribe"] = int("unsubscribe" in body_lower)
        md["spam_has_dear_friend"] = int(bool(re.search(r"dear (friend|sir|madam)", body, re.I)))
        md["spam_has_click_here"] = int("click here" in body_lower)
        
        # Header/sender features
        domain = from_email.split("@")[-1] if "@" in from_email else ""
        md["sender_domain_match"] = int(domain in headers)
        md["reply_to_mismatch"] = int("Reply-To" in headers and from_email not in headers)
        md["has_display_name"] = int("<" in headers and ">" in headers)
        md["header_length"] = len(headers)
        md["free_email_provider"] = int(any(x in from_email for x in ["gmail", "yahoo", "hotmail"]))
        md["suspicious_sender"] = int(any(x in from_email for x in SUSPICIOUS_SENDER_PATTERNS))
        md["subject_has_buy_now"] = int(bool(re.search(r"free|offer|buy now", subject, re.I)))

        return md

    # ------------------------------------------------------------
    # Reputation lookup
    # ------------------------------------------------------------
    def _sender_reputation(self, email):
        key = email.lower().strip()
        return self.sender_rep_map.get(key, {c: 0 for c in self.reputation_cols})

    # ------------------------------------------------------------
    # Preprocess final
    # ------------------------------------------------------------
    def preprocess_email(self, payload):

        body = safe_text(payload.get("body"))
        subject = safe_text(payload.get("subject"))
        raw_headers = safe_text(payload.get("headers") or payload.get("raw_headers", ""))

        # FIXED clean sender parsing
        raw_from = payload.get("from") or payload.get("from_email") or ""
        from_email = extract_clean_email(raw_from)

        # vectorize TF-IDF
        body_vec = self.body_vec.transform([body])
        subj_vec = self.subj_vec.transform([subject])

        # metadata
        md_dict = self._metadata(body, subject, from_email, raw_headers)
        md_vec = csr_matrix([[md_dict.get(c, 0) for c in self.metadata_cols]])

        # sender reputation
        rep_dict = self._sender_reputation(from_email)
        rep_vec = csr_matrix([[rep_dict.get(c, 0) for c in self.reputation_cols]])

        # combined
        X = hstack([body_vec, subj_vec, md_vec, rep_vec], format="csr")

        if X.shape[1] != self.n_features:
            raise ValueError(f"❌ Feature mismatch: got {X.shape[1]}, expected {self.n_features}")

        return X, md_dict, rep_dict, from_email

    # ------------------------------------------------------------
    # ENHANCED Multi-factor Risk Analysis
    # ------------------------------------------------------------
    def _calculate_risk_score(self, md: Dict, rep: Dict, from_email: str, 
                             body: str, subject: str) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score based on multiple factors
        """
        risk_score = 0
        risk_factors = []
        severity = "low"
        
        body_lower = body.lower()
        subject_lower = subject.lower()
        combined_text = body_lower + " " + subject_lower
        
        # Domain reputation
        domain_analysis = analyze_domain_reputation(from_email)
        if domain_analysis["risk_level"] == "critical":
            risk_score += 40
            risk_factors.append({
                "type": "critical_domain",
                "weight": "critical",
                "value": domain_analysis["domain"],
                "description": f"High-risk domain: {', '.join(domain_analysis['reasons'])}"
            })
        elif domain_analysis["risk_level"] == "high":
            risk_score += 20
            risk_factors.append({
                "type": "suspicious_domain",
                "weight": "high",
                "value": domain_analysis["domain"],
                "description": "Suspicious domain patterns detected"
            })
        
        # CRITICAL phishing indicators
        if md.get("ip_in_url", 0) == 1:
            risk_score += 40
            risk_factors.append({
                "type": "ip_based_url",
                "weight": "critical",
                "description": "URL uses IP address instead of domain"
            })
        
        # Phishing phrases
        phishing_phrase_count = sum(1 for phrase in PHISHING_PHRASES if phrase in combined_text)
        if phishing_phrase_count >= 3:
            risk_score += 30
            risk_factors.append({
                "type": "phishing_phrases",
                "weight": "critical",
                "value": phishing_phrase_count,
                "description": f"{phishing_phrase_count} phishing phrases detected"
            })
        elif phishing_phrase_count >= 1:
            risk_score += 15
            risk_factors.append({
                "type": "phishing_phrases",
                "weight": "high",
                "value": phishing_phrase_count,
                "description": "Phishing-related phrases found"
            })
        
        # Spam keywords (moderate risk)
        spam_keyword_count = md.get("spam_keyword_count", 0)
        if spam_keyword_count >= 5:
            risk_score += 15
            risk_factors.append({
                "type": "promotional_content",
                "weight": "medium",
                "value": spam_keyword_count,
                "description": f"{spam_keyword_count} promotional keywords detected"
            })
        
        # Financial lures
        money_symbols = md.get("spam_money_symbol", 0)
        large_amounts = len(re.findall(r"[\$₹€£]\s*\d{1,3}(?:,\d{3})+|\d{6,}", combined_text))
        if money_symbols >= 2 or large_amounts >= 2:
            risk_score += 15
            risk_factors.append({
                "type": "financial_promotion",
                "weight": "medium",
                "value": {"symbols": money_symbols, "amounts": large_amounts, "discount": False},
                "description": "Financial promotions/discounts - marketing email"
            })
        
        # Suspicious sender
        if md.get("suspicious_sender", 0) == 1:
            risk_score += 15
            risk_factors.append({
                "type": "suspicious_sender",
                "weight": "high",
                "description": "Suspicious sender address pattern"
            })
        
        # Urgency tactics
        urgency_count = combined_text.count("urgent") + combined_text.count("immediately") + \
                       combined_text.count("expire") + combined_text.count("act now")
        if urgency_count >= 3:
            risk_score += 15
            risk_factors.append({
                "type": "urgency_tactics",
                "weight": "medium",
                "value": urgency_count,
                "description": "Urgency pressure tactics detected"
            })
        
        # Unknown sender
        if rep.get("sender_is_known", 0) == 0 and risk_score > 0:
            risk_score += 10
            risk_factors.append({
                "type": "unknown_sender",
                "weight": "low",
                "value": None,
                "description": "First-time sender"
            })
        
        # Suspicious TLD
        if md.get("suspicious_tld", 0) == 1:
            risk_score += 15
            risk_factors.append({
                "type": "suspicious_tld",
                "weight": "high",
                "description": "URL uses high-risk TLD (.xyz, .top, .ru, .cn)"
            })
        
        # Calculate severity
        if risk_score >= 60:
            severity = "critical"
        elif risk_score >= 35:
            severity = "high"
        elif risk_score >= 15:
            severity = "medium"
        else:
            severity = "low"
        
        return {
            "risk_score": risk_score,
            "severity": severity,
            "risk_factors": risk_factors,
            "domain_analysis": domain_analysis
        }

    # ------------------------------------------------------------
    # ENHANCED Predict with Multi-Factor Analysis
    # ------------------------------------------------------------
    # def predict_email(self, payload):

    #     X, md, rep, from_email = self.preprocess_email(payload)

    #     # Get ML model predictions
    #     dmat = xgb.DMatrix(X, feature_names=self.feature_names)
    #     probs = self.model.predict(dmat)[0]

    #     ham, spam, phish = map(float, probs)
        
    #     # Get body and subject for risk analysis
    #     body = safe_text(payload.get("body"))
    #     subject = safe_text(payload.get("subject"))
        
    #     # Calculate risk score
    #     risk_analysis = self._calculate_risk_score(md, rep, from_email, body, subject)
        
    #     reasons = risk_analysis["risk_factors"]
    #     risk_score = risk_analysis["risk_score"]
    #     severity = risk_analysis["severity"]
        
    #     # Determine final label
    #     # Start with ML model's prediction
    #     ml_scores = {"ham": ham, "spam": spam, "phishing": phish}
    #     label = max(ml_scores, key=ml_scores.get)
    #     confidence = ml_scores[label]
        
    #     # Override for obvious spam (promotional content)
    #     if (md.get("spam_keyword_count", 0) >= 3 or 
    #         md.get("subject_has_buy_now", 0) == 1 or
    #         (spam > 0.3 and risk_score >= 15)):
    #         label = "spam"
    #         confidence = max(spam, confidence)
        
    #     # Override for critical phishing threats
    #     if risk_score >= 60 and phish > 0.15:
    #         label = "phishing"
    #         confidence = max(phish, confidence)
        
    #     # Generate explanation
    #     explanation = None
    #     if label == "spam":
    #         explanation = "Classified as SPAM due to promotional content and marketing keywords"
    #     elif label == "phishing":
    #         explanation = f"CRITICAL THREAT: {len(reasons)} phishing indicators detected"
        
    #     return {
    #         "message_id": payload.get("message_id") or "unknown",
    #         "from_email": from_email,
    #         "to": payload.get("to") or [],
    #         "subject": payload.get("subject") or "",

    #         "predicted": {
    #             "ham_score": ham,
    #             "spam_score": spam,
    #             "phishing_score": phish,
    #             "predicted_label": label,
    #             "model_used": "xgboost_v3_enhanced",
    #             "model_confidence": min(0.99, confidence)
    #         },

    #         "risk_analysis": {
    #             "risk_score": risk_score,
    #             "severity": severity,
    #             "domain_reputation": risk_analysis["domain_analysis"]
    #         },

    #         "metadata": md,
    #         "reputation": rep,
    #         "reasons": reasons,
    #         "explanation": explanation
    #     }
    # ------------------------------------------------------------
    # ENHANCED Predict with Multi-Factor Analysis
    # ------------------------------------------------------------
    def predict_email(self, payload):

        X, md, rep, from_email = self.preprocess_email(payload)

        # Get ML model predictions
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        probs = self.model.predict(dmat)[0]

        ham, spam, phish = map(float, probs)
        
        # Get body and subject for risk analysis
        body = safe_text(payload.get("body"))
        subject = safe_text(payload.get("subject"))
        
        # Calculate risk score
        risk_analysis = self._calculate_risk_score(md, rep, from_email, body, subject)
        
        reasons = risk_analysis["risk_factors"]
        risk_score = risk_analysis["risk_score"]
        severity = risk_analysis["severity"]
        
        # Determine final label
        # Start with ML model's prediction
        ml_scores = {"ham": ham, "spam": spam, "phishing": phish}
        label = max(ml_scores, key=ml_scores.get)
        confidence = ml_scores[label]
        
        # Override for obvious spam (promotional content)
        if (md.get("spam_keyword_count", 0) >= 3 or 
            md.get("subject_has_buy_now", 0) == 1 or
            (spam > 0.3 and risk_score >= 15)):
            label = "spam"
            confidence = max(spam, confidence)
        
        # Override for critical phishing threats
        if risk_score >= 60 and phish > 0.15:
            label = "phishing"
            confidence = max(phish, confidence)
        
        # Sub-category for more granular classification
        category = None
        if label == "spam":
            if risk_score > 30:
                category = "spam_high_risk"
            else:
                category = "spam_promotional"
        elif label == "phishing":
            if risk_score >= 60:
                category = "phishing_critical"
            else:
                category = "phishing_suspicious"
        
        # Generate explanation
        explanation = None
        if label == "spam":
            if category == "spam_high_risk":
                explanation = f"Classified as HIGH-RISK SPAM: {len(reasons)} risk indicators with aggressive marketing tactics"
            else:
                explanation = "Classified as SPAM due to promotional content and marketing keywords"
        elif label == "phishing":
            if category == "phishing_critical":
                explanation = f"CRITICAL PHISHING THREAT: {len(reasons)} phishing indicators detected - Immediate action recommended"
            else:
                explanation = f"Suspicious phishing attempt: {len(reasons)} risk indicators detected"
        else:
            explanation = "Legitimate email with no significant risk indicators"
        
        return {
            "message_id": payload.get("message_id") or "unknown",
            "from_email": from_email,
            "to": payload.get("to") or [],
            "subject": payload.get("subject") or "",

            "predicted": {
                "ham_score": round(ham, 4),
                "spam_score": round(spam, 4),
                "phishing_score": round(phish, 4),
                "predicted_label": label,
                "category": category,
                "model_used": "xgboost_v3_enhanced",
                "model_confidence": round(min(0.99, confidence), 4)
            },

            "risk_analysis": {
                "risk_score": risk_score,
                "severity": severity,
                "domain_reputation": risk_analysis["domain_analysis"],
                "threat_type": risk_analysis.get("threat_type", "none"),
                "phishing_indicators": risk_analysis.get("phishing_indicators", 0),
                "spam_indicators": risk_analysis.get("spam_indicators", 0)
            },

            "metadata": md,
            "reputation": rep,
            "reasons": reasons,
            "explanation": explanation
        }

# ------------------------------------------------------------
# SINGLETON with proper path configuration
# ------------------------------------------------------------
_predictor_instance = None

def get_predictor(models_dir: str = None, features_dir: str = None):
    """
    Get enhanced predictor instance with robust detection
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        # Set default paths if not provided
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if features_dir is None:
            features_dir = os.path.join(os.path.dirname(__file__), "..", "data", "features")
        
        print(f"🔧 Initializing Enhanced Predictor with:")
        print(f"   Models dir: {os.path.abspath(models_dir)}")
        print(f"   Features dir: {os.path.abspath(features_dir)}")
        
        _predictor_instance = Predictor(models_dir=models_dir, features_dir=features_dir)
    
    return _predictor_instance