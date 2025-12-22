# scripts/train_models_v3.py
"""
Industrial training script (v3)
Trains XGBoost + CatBoost using v2 features.
Produces:
 - models/xgboost_v3.json
 - models/catboost_v3.cbm
 - models/model_metrics_v3.json
 - models/feature_importance_v3.json
"""
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import multiprocessing

# Imbalanced tools
from imblearn.over_sampling import SMOTE, RandomOverSampler

# XGBoost & CatBoost
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Paths
ROOT = Path(".")
FEATURE_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Files (these must exist)
TFIDF_BODY = FEATURE_DIR / "tfidf_body_train_v2.npz"
TFIDF_SUBJ = FEATURE_DIR / "tfidf_subject_train_v2.npz"
METADATA = FEATURE_DIR / "metadata_train_v2.parquet"
REPUTATION = FEATURE_DIR / "reputation_train_v2.parquet"
FEATURE_NAMES = FEATURE_DIR / "feature_names_v2.json"

# Training settings
RANDOM_STATE = 42
N_JOBS = max(1, multiprocessing.cpu_count() - 1)
print(f"Using {N_JOBS} CPU workers")

def load_features():
    print("🔁 Loading TF-IDF (body + subject) sparse matrices...")
    X_body = load_npz(TFIDF_BODY)
    X_subj = load_npz(TFIDF_SUBJ)
    print("  body shape:", X_body.shape, "subject shape:", X_subj.shape)

    print("🔁 Loading metadata and reputation tables...")
    meta = pd.read_parquet(METADATA).reset_index(drop=True)
    rep = pd.read_parquet(REPUTATION).reset_index(drop=True)

    # Align lengths check
    n_rows = X_body.shape[0]
    assert X_subj.shape[0] == n_rows, "Subject TF-IDF rows != body TF-IDF rows"
    assert len(meta) == n_rows and len(rep) == n_rows, "Metadata/Reputation rows do not match TF-IDF rows"

    # Convert metadata + rep to sparse matrix
    meta_df = pd.concat([meta, rep.drop(columns=[c for c in rep.columns if c in meta.columns], errors='ignore')], axis=1)
    meta_cols = meta_df.columns.tolist()
    meta_vals = csr_matrix(meta_df.fillna(0).values)

    print("  metadata shape:", meta_vals.shape, "cols:", len(meta_cols))

    # horizontally stack: [body_tfidf | subject_tfidf | metadata/reputation]
    X = hstack([X_body, X_subj, meta_vals], format='csr')
    print("✅ Combined feature matrix shape:", X.shape)

    # Load labels (label column should be in metadata or rep)
    if "label" in meta_df.columns:
        y_ser = meta_df["label"]
    else:
        # if label not present, try reading from processed dataset
        raise RuntimeError("Label column missing in metadata. Ensure metadata_train_v2.parquet contains 'label'.")

    return X, y_ser.values, meta_df, meta_cols, X_body.shape[1], X_subj.shape[1]

def align_feature_names(x_body_cols, x_subj_cols, meta_cols):
    # Load feature_names_v2.json to confirm order expected by predictor
    with open(FEATURE_NAMES, "r", encoding="utf8") as f:
        feature_names = json.load(f)

    # feature_names should equal concatenation:
    # [body_tfidf_features..., subject_tfidf_features..., metadata_feature_names...]
    # Here we don't have the TF-IDF token names (they're in tfidf_body_v2.pkl and tfidf_subject_v2.pkl),
    # but we only need to guarantee the final order matches feature_names_v2.json.
    return feature_names

def prepare_train_data(X, y):
    # Label encode y to 0..k-1
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_.tolist()
    print("Label classes:", class_names)
    return y_enc, le

def balance_data(X, y_enc):
    # We want to avoid destroying phishing signal. Strategy:
    # 1) Use RandomOverSampler to upsample phishing moderately to at least 5% of major class
    # 2) Use SMOTE to balance ham and spam classes
    unique, counts = np.unique(y_enc, return_counts=True)
    print("Class distribution before balancing:", dict(zip(unique, counts)))

    classes = dict(zip(unique, counts))
    # find labels for ham/spam/phishing by counts heuristic (smallest -> phishing)
    sorted_by_count = sorted(classes.items(), key=lambda x: x[1])
    minority_label = sorted_by_count[0][0]  # assume phishing
    major_label = sorted_by_count[-1][0]

    # Step A: Slightly oversample minority (phishing) with RandomOverSampler to reach 5% of major
    major_count = classes[major_label]
    target_phishing = max(classes[minority_label], int(0.05 * major_count))
    ros = RandomOverSampler(sampling_strategy={minority_label: target_phishing}, random_state=RANDOM_STATE)
    X_ros, y_ros = ros.fit_resample(X, y_enc)
    print("After RandomOverSampler:", {k: v for k, v in zip(*np.unique(y_ros, return_counts=True))})

    # Step B: SMOTE for ham & spam to reach balance between major classes (if ham/spam exist)
    # Build sampling strategy: make all classes equal to the largest (after ROS)
    unique2, counts2 = np.unique(y_ros, return_counts=True)
    max_count = max(counts2)
    sampling = {int(k): int(max_count) for k in unique2}
    sm = SMOTE(sampling_strategy=sampling, random_state=RANDOM_STATE, n_jobs=N_JOBS, k_neighbors=5)
    X_bal, y_bal = sm.fit_resample(X_ros, y_ros)
    print("After SMOTE balancing:", {k: v for k, v in zip(*np.unique(y_bal, return_counts=True))})

    return X_bal, y_bal

def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    print("\n🚀 Training XGBoost (v3)...")
    params = {
        "objective": "multi:softprob",
        "num_class": len(np.unique(y_train)),
        "eta": 0.15,
        "max_depth": 9,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "verbosity": 1,
        "nthread": N_JOBS
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    evals = [(dtrain, "train"), (dval, "val")]

    bst = xgb.train(params, dtrain, num_boost_round=400, early_stopping_rounds=30, evals=evals, verbose_eval=10)
    model_path = MODELS_DIR / "xgboost_v3.json"
    bst.save_model(str(model_path))
    print("✅ XGBoost saved to", model_path)
    return bst

def train_catboost(X_train, y_train, X_val, y_val):
    print("\n🚀 Training CatBoost (v3)...")
    # CatBoost expects dense for categorical handling; we pass sparse arrays via Pool (works)
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)

    cb = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.08,
        depth=8,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_STATE,
        early_stopping_rounds=50,
        task_type="CPU",
        thread_count=N_JOBS,
        verbose=100
    )
    cb.fit(train_pool, eval_set=val_pool)
    model_path = MODELS_DIR / "catboost_v3.cbm"
    cb.save_model(str(model_path))
    print("✅ CatBoost saved to", model_path)
    return cb

def evaluate_model(predict_fn, X, y_true, classes):
    y_pred = predict_fn(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    rep = classification_report(y_true, y_pred, target_names=classes, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "f1_weighted": f1, "report": rep, "confusion_matrix": cm.tolist()}

def main():
    start = time.time()
    X, y_raw, meta_df, meta_cols, n_body, n_subj = load_features()
    y_enc, le = prepare_train_data(X, y_raw)
    classes = le.classes_.tolist()

    # feature names alignment (we do a safe default list matching sizes)
    # Build placeholder feature names: body_tfidf_0..N, subject_tfidf_0..M, then meta_cols
    feature_names = []
    for i in range(n_body):
        feature_names.append(f"body_tfidf_{i}")
    for i in range(n_subj):
        feature_names.append(f"subject_tfidf_{i}")
    for c in meta_cols:
        feature_names.append(c)

    # Split to train/val (we already have overall train, but allow local split to create early-val)
    X_train_all, X_hold, y_train_all, y_hold = train_test_split(X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc)
    # use hold for validation
    X_train_bal, y_train_bal = balance_data(X_train_all, y_train_all)

    # Keep validation as hold-out (no bleeding)
    X_val = X_hold
    y_val = y_hold

    # Train XGBoost
    bst = train_xgboost(X_train_bal, y_train_bal, X_val, y_val, feature_names)

    # Predict function for XGBoost
    def xgb_predict(X_):
        d = xgb.DMatrix(X_, feature_names=feature_names)
        probs = bst.predict(d)
        preds = np.argmax(probs, axis=1)
        return preds

    # Train CatBoost
    cb = train_catboost(X_train_bal, y_train_bal, X_val, y_val)

    def cb_predict(X_):
        pool = Pool(X_)
        preds = cb.predict(pool)
        preds = np.array(preds, dtype=int).reshape(-1)
        return preds

    # Evaluate on hold-out (validation)
    print("\n📈 Evaluating on validation set...")
    xgb_metrics = evaluate_model(xgb_predict, X_val, y_val, classes)
    cb_metrics = evaluate_model(cb_predict, X_val, y_val, classes)

    metrics = {"xgboost": xgb_metrics, "catboost": cb_metrics}
    with open(MODELS_DIR / "model_metrics_v3.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance for XGBoost (top features)
    fmap = bst.get_score(importance_type="gain")
    # convert fmap to sorted list
    feat_imp = sorted(fmap.items(), key=lambda x: x[1], reverse=True)
    with open(MODELS_DIR / "feature_importance_v3.json", "w") as f:
        json.dump(feat_imp[:200], f, indent=2)

    print("✅ Training complete. Metrics saved to models/")
    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
