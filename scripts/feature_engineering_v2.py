# """
# Enhanced Feature Engineering V2 - Phishing-Aware Features
# Adds domain-specific features for better phishing/spam detection
# Works with your clean dataset structure
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import logging
# from typing import Dict
# import re
# from collections import defaultdict
# import gc
# import json

# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import save_npz, csr_matrix
# import joblib

# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger(__name__)


# class EnhancedFeatureExtractor:
#     """Extract enhanced features with phishing-aware patterns"""
    
#     # Phishing keywords
#     URGENT_KEYWORDS = [
#         'urgent', 'immediate', 'action required', 'verify', 'confirm',
#         'suspended', 'locked', 'expires', 'expiring', 'limited time',
#         'act now', 'click here', 'click below', 'update now'
#     ]
    
#     PRIZE_KEYWORDS = [
#         'winner', 'won', 'prize', 'reward', 'cash', 'million', 'lottery',
#         'selected', 'chosen', 'congratulations', 'claim', 'free', 'bonus'
#     ]
    
#     THREAT_KEYWORDS = [
#         'suspended', 'closed', 'locked', 'terminated', 'cancelled',
#         'final notice', 'last chance', 'deadline', 'legal action'
#     ]
    
#     MONEY_KEYWORDS = [
#         '$', 'usd', 'dollars', 'payment', 'refund', 'transfer', 'bank',
#         'account', 'credit card', 'wire', 'paypal', 'bitcoin'
#     ]
    
#     SUSPICIOUS_TLDS = [
#         '.tk', '.ml', '.ga', '.cf', '.gq', '.zip', '.xyz', '.top',
#         '.work', '.click', '.link', '.ru', '.cn'
#     ]
    
#     def __init__(self, processed_data_dir: Path, features_dir: Path):
#         self.processed_dir = Path(processed_data_dir)
#         self.features_dir = Path(features_dir)
#         self.features_dir.mkdir(parents=True, exist_ok=True)
        
#         self.tfidf_body = None
#         self.tfidf_subject = None
#         self.feature_names = []
    
#     # =========================================================================
#     # PHISHING KEYWORD DETECTION
#     # =========================================================================
    
#     def extract_phishing_keywords(self, text: str) -> Dict:
#         """Extract phishing-specific keyword features"""
#         if pd.isna(text):
#             text = ""
        
#         text_lower = str(text).lower()
        
#         return {
#             'urgent_keyword_count': sum(1 for kw in self.URGENT_KEYWORDS if kw in text_lower),
#             'prize_keyword_count': sum(1 for kw in self.PRIZE_KEYWORDS if kw in text_lower),
#             'threat_keyword_count': sum(1 for kw in self.THREAT_KEYWORDS if kw in text_lower),
#             'money_keyword_count': sum(1 for kw in self.MONEY_KEYWORDS if kw in text_lower),
#         }
    
#     def extract_url_features(self, text: str) -> Dict:
#         """Enhanced URL analysis"""
#         if pd.isna(text):
#             return {
#                 'url_count': 0,
#                 'suspicious_tld': 0,
#                 'ip_in_url': 0,
#                 'url_length_avg': 0.0,
#                 'https_ratio': 0.0
#             }
        
#         text = str(text)
#         url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
#         urls = re.findall(url_pattern, text, re.IGNORECASE)
        
#         if not urls:
#             return {
#                 'url_count': 0,
#                 'suspicious_tld': 0,
#                 'ip_in_url': 0,
#                 'url_length_avg': 0.0,
#                 'https_ratio': 0.0
#             }
        
#         https_count = sum(1 for url in urls if url.startswith('https'))
#         has_suspicious_tld = any(any(tld in url.lower() for tld in self.SUSPICIOUS_TLDS) for url in urls)
        
#         ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
#         has_ip = any(re.search(ip_pattern, url) for url in urls)
        
#         avg_url_length = np.mean([len(url) for url in urls])
        
#         return {
#             'url_count': len(urls),
#             'suspicious_tld': int(has_suspicious_tld),
#             'ip_in_url': int(has_ip),
#             'url_length_avg': avg_url_length,
#             'https_ratio': https_count / len(urls)
#         }
    
#     def extract_text_statistics(self, text: str) -> Dict:
#         """Enhanced text statistics"""
#         if pd.isna(text):
#             return {
#                 'text_length': 0,
#                 'word_count': 0,
#                 'uppercase_ratio': 0,
#                 'digit_ratio': 0,
#                 'exclamation_count': 0,
#                 'consecutive_caps': 0
#             }
        
#         text = str(text)
#         text_len = len(text)
        
#         if text_len == 0:
#             return {
#                 'text_length': 0,
#                 'word_count': 0,
#                 'uppercase_ratio': 0,
#                 'digit_ratio': 0,
#                 'exclamation_count': 0,
#                 'consecutive_caps': 0
#             }
        
#         words = text.split()
#         uppercase_ratio = sum(1 for c in text if c.isupper()) / text_len
#         digit_ratio = sum(1 for c in text if c.isdigit()) / text_len
#         consecutive_caps = len(re.findall(r'[A-Z]{3,}', text))
        
#         return {
#             'text_length': text_len,
#             'word_count': len(words),
#             'uppercase_ratio': uppercase_ratio,
#             'digit_ratio': digit_ratio,
#             'exclamation_count': text.count('!'),
#             'consecutive_caps': consecutive_caps
#         }
    
#     def extract_sender_features(self, from_addr: str) -> Dict:
#         """Extract sender-based features"""
#         features = {
#             'free_email_provider': 0,
#             'suspicious_sender': 0
#         }
        
#         if pd.isna(from_addr):
#             return features
        
#         from_addr = str(from_addr).lower()
        
#         if '@' in from_addr:
#             sender_domain = from_addr.split('@')[-1]
#             free_providers = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol']
#             features['free_email_provider'] = int(any(prov in sender_domain for prov in free_providers))
        
#         suspicious_patterns = ['noreply', 'donotreply', 'no-reply', 'alert', 'verify']
#         features['suspicious_sender'] = int(any(pat in from_addr for pat in suspicious_patterns))
        
#         return features
    
#     # =========================================================================
#     # FEATURE EXTRACTION PIPELINE
#     # =========================================================================
    
#     def fit_tfidf(self, train_df: pd.DataFrame):
#         """Fit TF-IDF vectorizers"""
#         logger.info("\n📊 Fitting TF-IDF vectorizers...")
        
#         self.tfidf_body = TfidfVectorizer(
#             max_features=5000,
#             ngram_range=(1, 2),
#             min_df=5,
#             max_df=0.95,
#             stop_words='english'
#         )
        
#         bodies = train_df['body'].fillna('').astype(str)
#         self.tfidf_body.fit(bodies)
        
#         self.tfidf_subject = TfidfVectorizer(
#             max_features=1000,
#             ngram_range=(1, 2),
#             min_df=3,
#             max_df=0.95,
#             stop_words='english'
#         )
        
#         subjects = train_df['subject'].fillna('').astype(str)
#         self.tfidf_subject.fit(subjects)
        
#         joblib.dump(self.tfidf_body, self.features_dir / 'tfidf_body_v2.pkl')
#         joblib.dump(self.tfidf_subject, self.features_dir / 'tfidf_subject_v2.pkl')
        
#         # Build feature names
#         self.feature_names = []
#         self.feature_names.extend([f'body_tfidf_{term}' for term in self.tfidf_body.get_feature_names_out()])
#         self.feature_names.extend([f'subject_tfidf_{term}' for term in self.tfidf_subject.get_feature_names_out()])
        
#         logger.info(f"✅ TF-IDF fitted: {len(self.tfidf_body.vocabulary_)} body, {len(self.tfidf_subject.vocabulary_)} subject")
    
#     def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Extract all enhanced features"""
#         features_list = []
        
#         for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
#             feat = {}
            
#             # Body features
#             body_text = row.get('body', '')
#             body_stats = self.extract_text_statistics(body_text)
#             body_phishing = self.extract_phishing_keywords(body_text)
#             body_urls = self.extract_url_features(body_text)
            
#             for k, v in body_stats.items():
#                 feat[f'body_{k}'] = v
#             for k, v in body_phishing.items():
#                 feat[f'body_{k}'] = v
#             feat.update(body_urls)
            
#             # Subject features
#             subject_text = row.get('subject', '')
#             subject_stats = self.extract_text_statistics(subject_text)
#             subject_phishing = self.extract_phishing_keywords(subject_text)
            
#             for k, v in subject_stats.items():
#                 feat[f'subject_{k}'] = v
#             for k, v in subject_phishing.items():
#                 feat[f'subject_{k}'] = v
            
#             # Sender features
#             sender_feats = self.extract_sender_features(row.get('from_email', ''))
#             feat.update(sender_feats)
            
#             features_list.append(feat)
        
#         return pd.DataFrame(features_list)
    
#     def extract_features(self, split_name: str):
#         """Extract features for a split"""
#         logger.info(f"\n📝 Extracting features for {split_name}...")
        
#         input_file = self.processed_dir / f'{split_name}.parquet'
#         df = pd.read_parquet(input_file)
        
#         # TF-IDF
#         logger.info("   Extracting TF-IDF...")
#         bodies = df['body'].fillna('').astype(str)
#         subjects = df['subject'].fillna('').astype(str)
        
#         body_tfidf = self.tfidf_body.transform(bodies)
#         subject_tfidf = self.tfidf_subject.transform(subjects)
        
#         save_npz(self.features_dir / f'tfidf_body_{split_name}_v2.npz', body_tfidf)
#         save_npz(self.features_dir / f'tfidf_subject_{split_name}_v2.npz', subject_tfidf)
        
#         # Enhanced metadata
#         logger.info("   Extracting enhanced metadata...")
#         metadata_df = self.extract_all_features(df)
        
#         # Add feature names
#         if split_name == 'train':
#             self.feature_names.extend(list(metadata_df.columns))
#             with open(self.features_dir / 'feature_names_v2.json', 'w') as f:
#                 json.dump(self.feature_names, f, indent=2)
#             logger.info(f"✅ Feature names saved ({len(self.feature_names)} total)")
        
#         metadata_df.to_parquet(self.features_dir / f'metadata_{split_name}_v2.parquet', index=False)
        
#         logger.info(f"✅ Saved features: {metadata_df.shape}")
#         del metadata_df
#         gc.collect()
    
#     def build_sender_reputation(self, train_df: pd.DataFrame):
#         """Build sender reputation"""
#         logger.info("\n🔍 Building sender reputation...")
        
#         reputation = defaultdict(lambda: {
#             'email_count': 0,
#             'spam_count': 0,
#             'phishing_count': 0,
#             'spam_ratio': 0.0
#         })
        
#         for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Reputation"):
#             sender = row.get('from_email', '')
#             if pd.isna(sender) or sender == '':
#                 continue
            
#             label = row.get('label', 'ham')
#             reputation[sender]['email_count'] += 1
            
#             if label == 'spam':
#                 reputation[sender]['spam_count'] += 1
#             elif label == 'phishing':
#                 reputation[sender]['phishing_count'] += 1
        
#         for sender in reputation:
#             total = reputation[sender]['email_count']
#             bad = reputation[sender]['spam_count'] + reputation[sender]['phishing_count']
#             reputation[sender]['spam_ratio'] = bad / total if total > 0 else 0.0
        
#         reputation_df = pd.DataFrame.from_dict(reputation, orient='index')
#         reputation_df.to_parquet(self.features_dir / 'sender_reputation_v2.parquet')
        
#         logger.info(f"✅ Reputation for {len(reputation)} senders")
#         return dict(reputation)
    
#     def extract_reputation_features(self, split_name: str, reputation_db: Dict):
#         """Extract reputation features"""
#         logger.info(f"\n👤 Extracting reputation for {split_name}...")
        
#         df = pd.read_parquet(self.processed_dir / f'{split_name}.parquet')
        
#         features_list = []
#         for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reputation"):
#             sender = row.get('from_email', '')
            
#             if pd.isna(sender) or sender == '' or sender not in reputation_db:
#                 feat = {
#                     'sender_email_count': 0,
#                     'sender_spam_count': 0,
#                     'sender_spam_ratio': 0.5,
#                     'sender_is_known': 0
#                 }
#             else:
#                 rep = reputation_db[sender]
#                 feat = {
#                     'sender_email_count': rep['email_count'],
#                     'sender_spam_count': rep['spam_count'],
#                     'sender_spam_ratio': rep['spam_ratio'],
#                     'sender_is_known': 1
#                 }
            
#             features_list.append(feat)
        
#         reputation_df = pd.DataFrame(features_list)
#         reputation_df.to_parquet(self.features_dir / f'reputation_{split_name}_v2.parquet', index=False)
        
#         logger.info(f"✅ Saved: {reputation_df.shape}")
#         del reputation_df
#         gc.collect()
    
#     def run_feature_extraction(self):
#         """Run complete feature extraction"""
#         logger.info("\n" + "=" * 80)
#         logger.info("ENHANCED FEATURE EXTRACTION V2")
#         logger.info("=" * 80)
        
#         # Load training data
#         train_df = pd.read_parquet(self.processed_dir / 'train.parquet')
#         logger.info(f"Loaded {len(train_df)} training samples")
        
#         # Fit TF-IDF
#         self.fit_tfidf(train_df)
        
#         # Extract features
#         for split in ['train', 'val', 'test']:
#             self.extract_features(split)
        
#         # Build reputation
#         reputation_db = self.build_sender_reputation(train_df)
        
#         # Extract reputation features
#         for split in ['train', 'val', 'test']:
#             self.extract_reputation_features(split, reputation_db)
        
#         logger.info("\n" + "=" * 80)
#         logger.info("✅ FEATURE EXTRACTION COMPLETE!")
#         logger.info("=" * 80)
        
#         logger.info(f"\n✅ Total features: {len(self.feature_names)}")
#         logger.info("   - TF-IDF: 6000")
#         logger.info(f"   - Enhanced metadata: {len(self.feature_names) - 6000}")


# def main():
#     processed_dir = Path('data/processed')
#     features_dir = Path('data/features')
    
#     if not (processed_dir / 'train.parquet').exists():
#         logger.error("❌ train.parquet not found!")
#         return 1
    
#     extractor = EnhancedFeatureExtractor(
#         processed_data_dir=processed_dir,
#         features_dir=features_dir
#     )
    
#     extractor.run_feature_extraction()
    
#     logger.info("\n✅ Done! Next: Train balanced model in Colab")
#     return 0


# if __name__ == '__main__':
#     import sys
#     sys.exit(main())




#!/usr/bin/env python3
"""
Feature Engineering v3 (Industrial — Option A)
Generates rich metadata, URL/html/header features, sender reputation and TF-IDF artifacts.

Outputs placed in: data/features/
- metadata_train_v3.parquet, metadata_val_v3.parquet, metadata_test_v3.parquet
- reputation_train_v3.parquet, reputation_val_v3.parquet, reputation_test_v3.parquet
- sender_reputation_v3.parquet
- tfidf_body_train_v3.npz, tfidf_subject_train_v3.npz (and val/test)
- tfidf_body_v3.pkl, tfidf_subject_v3.pkl
- feature_names_v3.json
"""
import os
import re
import json
import gc
import html
import string
import logging
from pathlib import Path
from typing import List, Dict, Any
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed, dump

from tqdm import tqdm

# -------------------------
# Config / Paths
# -------------------------
ROOT = Path(".")
RAW_PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# TF-IDF sizes
BODY_MAX_FEATURES = 5000
SUBJ_MAX_FEATURES = 1000

# Parallelization
N_JOBS = max(1, cpu_count() - 1)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Dictionaries / Regex / Helpers
# -------------------------
URL_RE = re.compile(r"(https?://[^\s'\"<>)+,]*)", flags=re.IGNORECASE)
SHORT_URL_DOMAINS = set([
    "bit.ly","t.co","tinyurl.com","goo.gl","ow.ly","buff.ly","adf.ly","bitly.com","is.gd","rb.gy"
])
SUSPICIOUS_TLDS = {".xyz", ".top", ".click", ".work", ".online", ".club", ".country", ".site", ".win", ".loan"}
SPAM_KEYWORDS = [
    "offer","discount","sale","promo","deal","limited time","buy now","order now",
    "free trial","special offer","save","% off","coupon","unsubscribe","dear friend",
    "act now","final notice","congratulations","winner","click here","apply now","best price"
]
MONEY_SYMBOLS = ["$", "£", "€", "₹", "¥"]
HTML_TAG_RE = re.compile(r"<([a-zA-Z0-9]+)(\s|>)")
IMG_TAG_RE = re.compile(r"<img\s", re.IGNORECASE)
SCRIPT_TAG_RE = re.compile(r"<script\s", re.IGNORECASE)
STYLE_TAG_RE = re.compile(r"<style\s", re.IGNORECASE)
BASE64_RE = re.compile(r"data:[a-zA-Z0-9/+-\.]+;base64,")
IP_IN_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")

# -------------------------
# Text cleaning helpers
# -------------------------
def safe_text(x):
    if pd.isna(x):
        return ""
    if isinstance(x, bytes):
        try:
            x = x.decode("utf8", errors="ignore")
        except:
            x = str(x, errors="ignore")
    x = str(x)
    x = html.unescape(x)
    x = x.replace("\r", " ").replace("\n", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def polylength(xs: List[str]) -> float:
    if not xs:
        return 0.0
    return float(np.mean([len(x) for x in xs]))

# -------------------------
# URL / HTML / Header Feature Functions
# -------------------------
def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return URL_RE.findall(text)

def is_shortened(url: str) -> int:
    try:
        host = re.sub(r"https?://", "", url).split("/")[0].lower()
    except:
        return 0
    host = host.split(":")[0]
    return int(any(host.endswith(d) or host == d for d in SHORT_URL_DOMAINS))

def has_suspicious_tld(url: str) -> int:
    for tld in SUSPICIOUS_TLDS:
        if url.lower().endswith(tld) or ("/" in url and tld in url.lower()):
            return 1
    return 0

def url_contains_ip(url: str) -> int:
    return int(bool(IP_IN_URL_RE.search(url)))

def extract_html_features(text: str) -> Dict[str, Any]:
    # measures on HTML presence
    if not text:
        return {
            "html_tag_count":0, "html_ratio":0.0,
            "img_count":0, "script_count":0, "style_count":0,
            "base64_count":0
        }
    tags = HTML_TAG_RE.findall(text)
    html_tag_count = len(tags)
    img_count = len(IMG_TAG_RE.findall(text))
    script_count = len(SCRIPT_TAG_RE.findall(text))
    style_count = len(STYLE_TAG_RE.findall(text))
    base64_count = len(BASE64_RE.findall(text))
    # html ratio = fraction of characters that look like HTML tags (simple)
    html_like_chars = sum(1 for _ in re.finditer(r"<[^>]+>", text))
    html_ratio = html_like_chars / max(1, len(text))
    return {
        "html_tag_count": html_tag_count,
        "html_ratio": float(html_ratio),
        "img_count": img_count,
        "script_count": script_count,
        "style_count": style_count,
        "base64_count": base64_count
    }

def extract_header_features(raw_headers: str, from_email: str, subject: str) -> Dict[str, Any]:
    # raw_headers could be full header text or None
    if not raw_headers:
        # fallback using from_email/subject heuristics
        return {
            "sender_domain_match": 0,
            "reply_to_mismatch": 0,
            "has_display_name": int(bool(re.search(r'".+?"', from_email))),
            "header_length": 0
        }
    r = str(raw_headers)
    header_length = len(r)
    # reply-to mismatch detection
    reply_to_mismatch = 0
    m_reply = re.search(r"Reply-To:\s*(.*)", r, flags=re.IGNORECASE)
    if m_reply:
        reply = m_reply.group(1)
        reply_to_mismatch = int("@" in reply and reply.strip().lower() not in from_email.lower())
    # sender domain match (does From domain match Return-Path / Received first hop?)
    from_domain = from_email.split("@")[-1] if "@" in from_email else ""
    m_return = re.search(r"Return-Path:\s*<(.*?)>", r, flags=re.IGNORECASE)
    sender_domain_match = 0
    if m_return:
        return_path = m_return.group(1)
        sender_domain_match = int(from_domain and return_path.endswith("@" + from_domain))
    has_display_name = int(bool(re.search(r'^[^<]+<', from_email)))
    return {
        "sender_domain_match": int(sender_domain_match),
        "reply_to_mismatch": int(reply_to_mismatch),
        "has_display_name": int(has_display_name),
        "header_length": int(header_length)
    }

# -------------------------
# Spam / Marketing Features
# -------------------------
def spam_marketing_features(text: str) -> Dict[str, Any]:
    t = text.lower() if text else ""
    keyword_count = sum(t.count(w) for w in SPAM_KEYWORDS)
    money_symbol_count = int(any(sym in t for sym in MONEY_SYMBOLS))
    pct_off = int("% off" in t or "off%" in t)
    uppercase_words = len(re.findall(r"\b[A-Z]{2,}\b", text))
    exclamation_count = text.count("!")
    has_unsubscribe = int("unsubscribe" in t)
    has_dear_friend = int("dear friend" in t or "dear valued" in t)
    has_click_here = int("click here" in t)
    return {
        "spam_keyword_count": int(keyword_count),
        "spam_money_symbol": int(money_symbol_count),
        "spam_pct_off": int(pct_off),
        "spam_uppercase_words": int(uppercase_words),
        "spam_exclamation_count": int(exclamation_count),
        "spam_has_unsubscribe": int(has_unsubscribe),
        "spam_has_dear_friend": int(has_dear_friend),
        "spam_has_click_here": int(has_click_here)
    }

# -------------------------
# Core per-row feature extraction (fast, vectorizable)
# -------------------------
def extract_row_features(row: pd.Series) -> Dict[str, Any]:
    # required input columns: body, subject, from_email, raw_headers (optional)
    body = safe_text(row.get("body", ""))
    subject = safe_text(row.get("subject", ""))
    from_email = safe_text(row.get("from_email", "")).lower()
    raw_headers = safe_text(row.get("raw_headers", ""))

    # basic body/subject stats
    body_words = body.split()
    subject_words = subject.split()
    features = {
        "body_text_length": len(body),
        "body_word_count": len(body_words),
        "body_avg_word_length": float(np.mean([len(w) for w in body_words]) if body_words else 0.0),
        "body_uppercase_ratio": sum(1 for c in body if c.isupper())/max(1,len(body)),
        "body_digit_ratio": sum(1 for c in body if c.isdigit())/max(1,len(body)),
        "body_special_char_ratio": sum(1 for c in body if c in string.punctuation)/max(1,len(body)),
        "body_exclamation_count": body.count("!"),
        "body_question_count": body.count("?"),
        "body_consecutive_caps": int(bool(re.search(r"[A-Z]{3,}", body))),
        "subject_text_length": len(subject),
        "subject_word_count": len(subject_words),
        "subject_avg_word_length": float(np.mean([len(w) for w in subject_words]) if subject_words else 0.0),
        "subject_uppercase_ratio": sum(1 for c in subject if c.isupper())/max(1,len(subject)),
        "subject_digit_ratio": sum(1 for c in subject if c.isdigit())/max(1,len(subject)),
        "subject_exclamation_count": subject.count("!"),
    }

    # URLs
    urls = extract_urls(body + " " + subject)
    features["url_count"] = len(urls)
    features["has_url"] = int(len(urls) > 0)
    if urls:
        features["suspicious_tld"] = int(any(has_suspicious_tld(u) for u in urls))
        features["ip_in_url"] = int(any(url_contains_ip(u) for u in urls))
        features["shortened_url"] = int(any(is_shortened(u) for u in urls))
        features["https_ratio"] = float(sum(1 for u in urls if u.lower().startswith("https"))/len(urls))
        features["url_length_avg"] = float(np.mean([len(u) for u in urls]))
        # obfuscation: hex encoding, @ in path, long query strings
        features["url_has_hex_encoding"] = int(any(re.search(r"%[0-9A-Fa-f]{2}", u) for u in urls))
        features["url_long_query"] = int(any("?" in u and len(u.split("?",1)[1])>30 for u in urls))
    else:
        features.update({
            "suspicious_tld":0, "ip_in_url":0, "shortened_url":0,
            "https_ratio":0.0, "url_length_avg":0.0,
            "url_has_hex_encoding":0, "url_long_query":0
        })

    # HTML features
    html_feats = extract_html_features(body)
    features.update(html_feats)

    # spam marketing features
    spam_feats = spam_marketing_features(body + " " + subject)
    features.update(spam_feats)

    # header features
    header_feats = extract_header_features(raw_headers, from_email, subject)
    features.update(header_feats)

    # sender features
    free_domains = {"gmail.com","yahoo.com","hotmail.com","outlook.com","aol.com"}
    sender_domain = from_email.split("@")[-1] if "@" in from_email else ""
    features["free_email_provider"] = int(sender_domain in free_domains)
    features["suspicious_sender"] = int(any(token in from_email for token in ["noreply","secure","account","support","admin","info"]))
    features["has_display_name"] = int(bool(re.search(r'^[^<]+<', row.get("from", "") or from_email)))
    # simple URL-looking names in subject/from
    features["subject_has_buy_now"] = int("buy now" in subject.lower())

    return features

# -------------------------
# Batch extraction utilities
# -------------------------
def df_extract_features(df: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    # Use parallel map over rows
    rows = []
    if n_jobs == 1:
        for _, r in tqdm(df.iterrows(), total=len(df), desc="extract rows"):
            rows.append(extract_row_features(r))
    else:
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(extract_row_features)(row) for _, row in df.iterrows()
        )
        rows = res
    meta = pd.DataFrame(rows)
    return meta

# -------------------------
# Sender reputation
# -------------------------
def build_sender_reputation(train_df: pd.DataFrame) -> pd.DataFrame:
    # compute counts and ratios
    df = train_df.copy()
    df["from_email_norm"] = df["from_email"].astype(str).str.lower().fillna("")
    grp = df.groupby("from_email_norm")["label"].value_counts().unstack(fill_value=0)
    # keep counts and ratio spam
    cols = {}
    cols["sender_email_count"] = grp.sum(axis=1)
    if 0 in grp.columns:
        cols["sender_ham_count"] = grp[0]
    else:
        cols["sender_ham_count"] = 0
    if 1 in grp.columns:
        cols["sender_spam_count"] = grp[1]
    else:
        cols["sender_spam_count"] = 0
    # phishing label might be 2 or 'phishing', unify: we'll map labels earlier in pipeline to ints 0/1/2
    if 2 in grp.columns:
        cols["sender_phish_count"] = grp[2]
    else:
        cols["sender_phish_count"] = 0
    rep = pd.DataFrame(cols)
    rep["sender_spam_ratio"] = rep["sender_spam_count"] / rep["sender_email_count"].replace(0,1)
    rep["sender_is_known"] = (rep["sender_email_count"] > 1).astype(int)
    rep.index.name = "from_email_norm"
    rep.reset_index(inplace=True)
    return rep

# -------------------------
# TF-IDF utilities
# -------------------------
def fit_and_save_tfidf(train_df: pd.DataFrame):
    logging.info("Fitting TF-IDF for body and subject")
    body_vec = TfidfVectorizer(max_features=BODY_MAX_FEATURES, stop_words="english", ngram_range=(1,2))
    subj_vec = TfidfVectorizer(max_features=SUBJ_MAX_FEATURES, stop_words="english", ngram_range=(1,2))

    # fit on train
    body_vec.fit(train_df["body"].astype(str).values)
    subj_vec.fit(train_df["subject"].astype(str).values)

    # persist vectorizers
    dump(body_vec, FEATURE_DIR / "tfidf_body_v3.pkl")
    dump(subj_vec, FEATURE_DIR / "tfidf_subject_v3.pkl")

    def transform_and_save(vec, df, name):
        X = vec.transform(df.astype(str).values)
        save_npz(FEATURE_DIR / f"{name}_v3.npz", X)
        return X.shape

    # transform train/val/test
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logging.info(f"Transform TF-IDF {split}")
        if split == "train":
            X_shape = transform_and_save(body_vec, df["body"], f"tfidf_body_{split}")
            transform_and_save(subj_vec, df["subject"], f"tfidf_subject_{split}")
        else:
            transform_and_save(body_vec, df["body"], f"tfidf_body_{split}")
            transform_and_save(subj_vec, df["subject"], f"tfidf_subject_{split}")

    return body_vec, subj_vec

# -------------------------
# Feature name builder (exact ordering)
# -------------------------
def build_feature_names(body_vec, subj_vec, metadata_cols, reputation_cols) -> List[str]:
    # token names from vectorizers
    try:
        body_feats = [f"body_tfidf_{tok}" for tok in body_vec.get_feature_names_out()]
        subj_feats = [f"subject_tfidf_{tok}" for tok in subj_vec.get_feature_names_out()]
    except:
        body_feats = [f"body_tfidf_{i}" for i in range(BODY_MAX_FEATURES)]
        subj_feats = [f"subject_tfidf_{i}" for i in range(SUBJ_MAX_FEATURES)]
    names = body_feats + subj_feats + list(metadata_cols) + list(reputation_cols)
    # save
    with open(FEATURE_DIR / "feature_names_v3.json", "w", encoding="utf8") as f:
        json.dump(names, f, indent=2)
    return names

# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    logging.info("Starting Feature Engineering v3 (Industrial Option A)")

    # load processed parquet splits
    train_path = RAW_PROCESSED / "train.parquet"
    val_path = RAW_PROCESSED / "val.parquet"
    test_path = RAW_PROCESSED / "test.parquet"

    if not train_path.exists():
        logging.error("train.parquet not found at data/processed/. Run build dataset first.")
        raise SystemExit(1)

    train_df = pd.read_parquet(train_path).reset_index(drop=True)
    val_df = pd.read_parquet(val_path).reset_index(drop=True) if val_path.exists() else pd.DataFrame(columns=train_df.columns)
    test_df = pd.read_parquet(test_path).reset_index(drop=True) if test_path.exists() else pd.DataFrame(columns=train_df.columns)

    # Ensure label mapping to ints: {ham:0, spam:1, phishing:2}
    # Accept labels encoded as strings 'ham','spam','phishing' or 0/1/2
    def normalize_label_series(ser):
        if ser.dtype == int or np.issubdtype(ser.dtype, np.integer):
            return ser.astype(int)
        s = ser.astype(str).str.lower()
        mapping = {}
        mapping.update({v:0 for v in ["ham","0","ham\n"]})
        mapping.update({v:1 for v in ["spam","1"]})
        mapping.update({v:2 for v in ["phishing","phish","phisher","2"]})
        return s.map(lambda x: mapping.get(x.strip(), np.nan)).astype("Int64").astype(int)

    if "label" not in train_df.columns:
        logging.error("train.parquet must include label column")
        raise SystemExit(1)

    # apply normalization
    train_df["label"] = normalize_label_series(train_df["label"])
    if not val_df.empty:
        val_df["label"] = normalize_label_series(val_df["label"])
    if not test_df.empty:
        test_df["label"] = normalize_label_series(test_df["label"])

    # Extract metadata features
    logging.info("Extracting metadata features for train")
    meta_train = df_extract_features(train_df, n_jobs=N_JOBS)
    logging.info("Extracting metadata features for val")
    meta_val = df_extract_features(val_df, n_jobs=N_JOBS) if not val_df.empty else pd.DataFrame(columns=meta_train.columns)
    logging.info("Extracting metadata features for test")
    meta_test = df_extract_features(test_df, n_jobs=N_JOBS) if not test_df.empty else pd.DataFrame(columns=meta_train.columns)

    # Attach label into metadata (CRITICAL)
    meta_train["label"] = train_df["label"].values
    meta_val["label"] = val_df["label"].values if not val_df.empty else pd.Series(dtype="int")
    meta_test["label"] = test_df["label"].values if not test_df.empty else pd.Series(dtype="int")

    # Save metadata parquet
    meta_train.to_parquet(FEATURE_DIR / "metadata_train_v3.parquet", index=False)
    meta_val.to_parquet(FEATURE_DIR / "metadata_val_v3.parquet", index=False)
    meta_test.to_parquet(FEATURE_DIR / "metadata_test_v3.parquet", index=False)
    logging.info("Saved metadata parquet files")

    # Build sender reputation from train and join to produce reputation files
    logging.info("Building sender reputation from train")
    rep = build_sender_reputation(train_df)
    rep.to_parquet(FEATURE_DIR / "sender_reputation_v3.parquet", index=False)
    # join per-split
    def join_rep(df, meta):
        df2 = df.copy()
        df2["from_email_norm"] = df2["from_email"].astype(str).str.lower().fillna("")
        merged = df2.merge(rep, left_on="from_email_norm", right_on="from_email_norm", how="left")
        # missing -> zeros
        merged = merged.fillna(0)
        cols = ["sender_email_count","sender_ham_count","sender_spam_count","sender_phish_count","sender_spam_ratio","sender_is_known"]
        # ensure all columns exist
        for c in cols:
            if c not in merged.columns:
                merged[c] = 0
        return merged[cols]

    rep_train = join_rep(train_df, meta_train)
    rep_val = join_rep(val_df, meta_val) if not val_df.empty else pd.DataFrame(columns=rep_train.columns)
    rep_test = join_rep(test_df, meta_test) if not test_df.empty else pd.DataFrame(columns=rep_train.columns)

    rep_train.to_parquet(FEATURE_DIR / "reputation_train_v3.parquet", index=False)
    rep_val.to_parquet(FEATURE_DIR / "reputation_val_v3.parquet", index=False)
    rep_test.to_parquet(FEATURE_DIR / "reputation_test_v3.parquet", index=False)
    logging.info("Saved reputation parquet files")

    # Fit TF-IDF on train and transform all splits
    body_vec, subj_vec = fit_and_save_tfidf(train_df)

    # Build final feature names (ordering: body_tfidf tokens, subject_tfidf tokens, metadata cols, reputation cols)
    metadata_cols = list(meta_train.columns.drop("label"))
    reputation_cols = list(rep_train.columns)
    feature_names = build_feature_names(body_vec, subj_vec, metadata_cols, reputation_cols)
    logging.info(f"Built {len(feature_names)} feature names")

    logging.info("🔧 Feature Engineering v3 complete — artifacts saved in data/features/")
    # cleanup
    gc.collect()
