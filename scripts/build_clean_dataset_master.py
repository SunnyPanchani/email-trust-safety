

"""
Fast Dataset Builder with Parallel Processing
Processes 75K+ emails in minutes instead of hours
"""

import os
import re
import email
import html
import pandas as pd
from pathlib import Path
from email import policy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


import os
import re
import email
import html
import pandas as pd
from pathlib import Path
from email import policy
from concurrent.futures import ProcessPoolExecutor

# ============================
# CLEAN OLD PROCESSED DATA
# ============================
import shutil

PROCESSED_DIR = Path("data/processed")

print("🧹 Cleaning old processed data...")

shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("✔ Old processed data removed.\n")

RAW_DIR = Path("data/raw")


# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Number of parallel workers
NUM_WORKERS = mp.cpu_count()

print(f"🚀 Using {NUM_WORKERS} CPU cores for parallel processing")


# =======================================================
# Helper Functions
# =======================================================

def clean_text(txt):
    """Clean and normalize text"""
    if not txt:
        return ""
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def parse_email_file(file_path):
    """Parse a single email file"""
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        subject = msg.get("subject", "") or ""
        sender = msg.get("from", "") or ""
        
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body += part.get_content()
                    except:
                        pass
        else:
            try:
                body = msg.get_content()
            except:
                body = ""
        
        return {
            'from_email': sender,
            'subject': clean_text(subject),
            'body': clean_text(body)
        }
    except:
        return None


def process_email_batch(args):
    """Process a batch of email files (for parallel processing)"""
    files, label, source = args
    results = []
    
    for file_path in files:
        parsed = parse_email_file(file_path)
        if parsed and len(parsed['body']) > 20:
            parsed['label'] = label
            parsed['source'] = source
            results.append(parsed)
    
    return results


# =======================================================
# 1. Load SpamAssassin (Parallel)
# =======================================================

def load_spamassassin_fast():
    """Load SpamAssassin with parallel processing"""
    print("\n📥 Loading SpamAssassin...")
    
    root = RAW_DIR / "extracted_spamassassin"
    
    # Collect all files by category
    ham_files = []
    spam_files = []
    
    for subset in root.glob("*"):
        if not subset.is_dir():
            continue
        
        for folder in subset.glob("*"):
            if not folder.is_dir():
                continue
            
            folder_name = folder.name.lower()
            
            if "ham" in folder_name:
                ham_files.extend(list(folder.glob("*")))
            elif "spam" in folder_name:
                spam_files.extend(list(folder.glob("*")))
    
    # Remove non-files
    ham_files = [f for f in ham_files if f.is_file()]
    spam_files = [f for f in spam_files if f.is_file()]
    
    print(f"   Found {len(ham_files)} ham files, {len(spam_files)} spam files")
    
    # Split into batches for parallel processing
    batch_size = 500
    
    ham_batches = [ham_files[i:i+batch_size] for i in range(0, len(ham_files), batch_size)]
    spam_batches = [spam_files[i:i+batch_size] for i in range(0, len(spam_files), batch_size)]
    
    # Process ham in parallel
    print("   Processing ham emails...")
    ham_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_email_batch, (batch, 'ham', 'spamassassin')) 
                   for batch in ham_batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ham"):
            ham_results.extend(future.result())
    
    # Process spam in parallel
    print("   Processing spam emails...")
    spam_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_email_batch, (batch, 'spam', 'spamassassin')) 
                   for batch in spam_batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Spam"):
            spam_results.extend(future.result())
    
    print(f"✅ SpamAssassin: {len(ham_results)} ham, {len(spam_results)} spam")
    
    return pd.DataFrame(ham_results), pd.DataFrame(spam_results)


# =======================================================
# 2. Load TREC (Parallel)
# =======================================================

def load_trec_fast():
    """Load TREC 2007 with parallel processing"""
    print("\n📥 Loading TREC 2007...")
    
    trec_root = RAW_DIR / "trec" / "extracted" / "trec07p"
    labels_file = trec_root / "full" / "index"
    data_dir = trec_root / "data"
    
    if not labels_file.exists() or not data_dir.exists():
        print("⚠️  TREC files not found, skipping")
        return pd.DataFrame(), pd.DataFrame()
    
    # Read labels
    print("   Reading labels...")
    label_map = {}
    with open(labels_file, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                lbl, rel_path = parts
                filename = Path(rel_path).name
                label_map[filename] = "spam" if lbl.lower() == "spam" else "ham"
    
    print(f"   Found {len(label_map)} labels")
    
    # Separate ham and spam files
    ham_files = []
    spam_files = []
    
    for fname, label in label_map.items():
        email_path = data_dir / fname
        if email_path.exists():
            if label == "ham":
                ham_files.append(email_path)
            else:
                spam_files.append(email_path)
    
    print(f"   Files: {len(ham_files)} ham, {len(spam_files)} spam")
    
    # Split into batches
    batch_size = 500
    ham_batches = [ham_files[i:i+batch_size] for i in range(0, len(ham_files), batch_size)]
    spam_batches = [spam_files[i:i+batch_size] for i in range(0, len(spam_files), batch_size)]
    
    # Process ham
    print("   Processing ham emails...")
    ham_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_email_batch, (batch, 'ham', 'trec2007')) 
                   for batch in ham_batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ham"):
            ham_results.extend(future.result())
    
    # Process spam
    print("   Processing spam emails...")
    spam_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_email_batch, (batch, 'spam', 'trec2007')) 
                   for batch in spam_batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Spam"):
            spam_results.extend(future.result())
    
    print(f"✅ TREC: {len(ham_results)} ham, {len(spam_results)} spam")
    
    return pd.DataFrame(ham_results), pd.DataFrame(spam_results)


# =======================================================
# 3. Load CSVs (Fast)
# =======================================================

def load_nazario():
    """Load Nazario phishing CSV"""
    print("\n📥 Loading Nazario phishing...")
    
    nazario_csv = RAW_DIR / "Nazario" / "Nazario_5.csv"
    
    if not nazario_csv.exists():
        print("⚠️  Nazario CSV not found")
        return pd.DataFrame()
    
    df = pd.read_csv(nazario_csv)
    df = df.rename(columns={"sender": "from_email", "subject": "subject", "body": "body"})
    df["label"] = "phishing"
    df["source"] = "nazario"
    
    df = df[["from_email", "subject", "body", "label", "source"]].dropna()
    print(f"✅ Nazario: {len(df)} phishing emails")
    
    return df


def load_ceas():
    """Load CEAS CSV"""
    print("\n📥 Loading CEAS...")
    
    ceas_csv = RAW_DIR / "ceas" / "CEAS_08.csv"
    
    if not ceas_csv.exists():
        print("⚠️  CEAS CSV not found")
        return pd.DataFrame()
    
    df = pd.read_csv(ceas_csv)
    df = df.rename(columns={"sender": "from_email", "subject": "subject", "body": "body"})
    df["label"] = df["label"].apply(lambda x: "spam" if int(x) == 1 else "ham")
    df["source"] = "ceas08"
    
    df = df[["from_email", "subject", "body", "label", "source"]].dropna()
    print(f"✅ CEAS: {len(df)} emails")
    
    return df


# =======================================================
# Dedupe & Split
# =======================================================

def dedupe(df):
    """Remove duplicates and short emails"""
    print(f"   Before dedupe: {len(df)}")
    df = df.drop_duplicates(subset=["subject", "body"])
    df = df[df["body"].str.len() > 30].reset_index(drop=True)
    print(f"   After dedupe: {len(df)}")
    return df


def split_data(df):
    """Split into train/val/test (80/10/10)"""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    
    train = df[:int(0.8 * n)]
    val = df[int(0.8 * n):int(0.9 * n)]
    test = df[int(0.9 * n):]
    
    return train, val, test


# =======================================================
# MAIN
# =======================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 FAST DATASET BUILDER")
    print("=" * 60)
    
    # Load all datasets in parallel
    sa_ham, sa_spam = load_spamassassin_fast()
    trec_ham, trec_spam = load_trec_fast()
    nazario = load_nazario()
    ceas = load_ceas()
    
    # Combine and dedupe
    print("\n" + "=" * 60)
    print("🔗 Combining datasets...")
    
    print("\n📊 Combining ham...")
    ham = dedupe(pd.concat([sa_ham, trec_ham, ceas[ceas["label"] == "ham"]], ignore_index=True))
    
    print("\n📊 Combining spam...")
    spam = dedupe(pd.concat([sa_spam, trec_spam, ceas[ceas["label"] == "spam"]], ignore_index=True))
    
    print("\n📊 Combining phishing...")
    phishing = dedupe(nazario)
    
    # Save class-wise
    print("\n💾 Saving class-specific files...")
    ham.to_parquet(PROCESSED_DIR / "clean_ham.parquet", index=False)
    spam.to_parquet(PROCESSED_DIR / "clean_spam.parquet", index=False)
    phishing.to_parquet(PROCESSED_DIR / "clean_phishing.parquet", index=False)
    
    print(f"   ✅ clean_ham.parquet: {len(ham)} emails")
    print(f"   ✅ clean_spam.parquet: {len(spam)} emails")
    print(f"   ✅ clean_phishing.parquet: {len(phishing)} emails")
    
    # Combine all and split
    print("\n📊 Creating train/val/test splits...")
    df_all = pd.concat([ham, spam, phishing], ignore_index=True)
    
    print(f"\n   Total emails: {len(df_all)}")
    print(f"   Label distribution:")
    print(df_all['label'].value_counts())
    
    train, val, test = split_data(df_all)
    
    # Save splits
    print("\n💾 Saving splits...")
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    
    print(f"   ✅ train.parquet: {len(train)} emails")
    print(f"   ✅ val.parquet: {len(val)} emails")
    print(f"   ✅ test.parquet: {len(test)} emails")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ DATASET BUILD COMPLETE!")
    print("=" * 60)
    
    print("\n📊 Final Statistics:")
    print(f"   Total: {len(df_all):,} emails")
    print(f"   Ham: {len(ham):,} ({len(ham)/len(df_all)*100:.1f}%)")
    print(f"   Spam: {len(spam):,} ({len(spam)/len(df_all)*100:.1f}%)")
    print(f"   Phishing: {len(phishing):,} ({len(phishing)/len(df_all)*100:.1f}%)")
    
    print("\n📁 Output: data/processed/")
    print("   - clean_ham.parquet")
    print("   - clean_spam.parquet")
    print("   - clean_phishing.parquet")
    print("   - train.parquet")
    print("   - val.parquet")
    print("   - test.parquet")
    
    print("\n🎯 Next step:")
    print("   python scripts/feature_engineering_v2.py")


if __name__ == "__main__":
    main()