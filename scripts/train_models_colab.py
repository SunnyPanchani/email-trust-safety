# scripts/train_models_v3.py
"""
Colab-ready training script (v3) — trains XGBoost + CatBoost using v3 feature artifacts.
Outputs:
 - models/xgboost_v3.json
 - models/catboost_v3.cbm
 - models/model_metrics_v3.json
 - models/feature_importance_v3.json
Run in Colab from project root (/content/email-trust-safety):
!python scripts/train_models_v3.py
"""
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import joblib

# ML libs
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Paths (Colab-friendly)
ROOT = Path(".")
FEATURE_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_BODY = FEATURE_DIR / "tfidf_body_train_v3.npz"
TFIDF_SUBJ = FEATURE_DIR / "tfidf_subject_train_v3.npz"
BODY_VEC_PKL = FEATURE_DIR / "tfidf_body_v3.pkl"
SUBJ_VEC_PKL = FEATURE_DIR / "tfidf_subject_v3.pkl"
META_PARQ = FEATURE_DIR / "metadata_train_v3.parquet"
REP_PARQ = FEATURE_DIR / "reputation_train_v3.parquet"
FEATURE_NAMES_JSON = FEATURE_DIR / "feature_names_v3.json"

RANDOM_STATE = 42
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

print("N_JOBS:", N_JOBS)
print("FEATURE_DIR:", FEATURE_DIR)

def load_sparse_and_tables():
    print("Loading sparse TF-IDF matrices...")
    X_body = load_npz(TFIDF_BODY)
    X_subj = load_npz(TFIDF_SUBJ)
    print(" body:", X_body.shape, " subj:", X_subj.shape)

    print("Loading metadata + reputation dataframes...")
    meta = pd.read_parquet(META_PARQ).reset_index(drop=True)
    rep = pd.read_parquet(REP_PARQ).reset_index(drop=True)

    # Ensure sizes align
    n = X_body.shape[0]
    assert X_subj.shape[0] == n, "TF-IDF row mismatch"
    assert len(meta) == n and len(rep) == n, "Metadata/Reputation row mismatch"

    # Convert meta+rep to sparse
    meta_df = pd.concat([meta, rep], axis=1)
    # Keep labels separately
    if "label" not in meta_df.columns:
        raise RuntimeError("metadata_train_v3.parquet does not contain 'label' column")
    y_raw = meta_df["label"].values
    meta_vals = csr_matrix(meta_df.drop(columns=["label"]).fillna(0).values)

    # Combine final feature matrix: [body | subject | meta+rep]
    X = hstack([X_body, X_subj, meta_vals], format="csr")
    print("Combined feature matrix shape:", X.shape)
    return X, y_raw, meta_df.drop(columns=["label"]), X_body.shape[1], X_subj.shape[1]

def build_feature_names():
    # Prefer to load full readable token names using the vectorizers
    body_vec = joblib.load(BODY_VEC_PKL)
    subj_vec = joblib.load(SUBJ_VEC_PKL)
    body_tokens = body_vec.get_feature_names_out()
    subj_tokens = subj_vec.get_feature_names_out()

    body_feats = [f"body_tfidf_{t}" for t in body_tokens]
    subj_feats = [f"subject_tfidf_{t}" for t in subj_tokens]

    # metadata + reputation column names from the parquet (we'll read them)
    meta_cols = pd.read_parquet(META_PARQ).drop(columns=["label"]).columns.tolist()
    rep_cols = pd.read_parquet(REP_PARQ).columns.tolist()

    feature_names = body_feats + subj_feats + meta_cols + rep_cols

    # Save (safety)
    with open(FEATURE_NAMES_JSON, "w", encoding="utf8") as f:
        json.dump(feature_names, f, indent=2)
    print("Built feature_names (len={}): saved to".format(len(feature_names)), FEATURE_NAMES_JSON)
    return feature_names

def encode_labels(y_raw):
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Classes:", le.classes_.tolist())
    return y, le

def targeted_oversample(X, y):
    # Slightly oversample the smallest class (likely phishing) with RandomOverSampler
    unique, counts = np.unique(y, return_counts=True)
    print("Before oversample:", dict(zip(unique, counts)))
    # find smallest label
    smallest = unique[np.argmin(counts)]
    # set target to 8% of largest class
    largest_count = counts.max()
    target = max(int(0.08 * largest_count), counts.min())
    ros = RandomOverSampler(sampling_strategy={int(smallest): target}, random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X, y)
    unique2, counts2 = np.unique(y_res, return_counts=True)
    print("After RandomOverSampler:", dict(zip(unique2, counts2)))
    return X_res, y_res

def compute_sample_weights(y_enc):
    # Define class weights manually (tune if needed)
    # Choose heavier weight for spam and phishing relative to ham
    # Map classes by sorted unique to ensure stable mapping: but we will require le.classes_ mapping
    # We'll compute weights from inverse class frequency as baseline
    unique, counts = np.unique(y_enc, return_counts=True)
    inv_freq = {k: (1.0 / v) for k, v in zip(unique, counts)}
    # normalize relative to ham (lowest weight) choose ham as min inv_freq
    min_if = min(inv_freq.values())
    weights_map = {k: (inv_freq[k] / min_if) for k in inv_freq}
    sample_weights = np.array([weights_map[int(c)] for c in y_enc], dtype=float)
    print("Sample weight mapping:", weights_map)
    return sample_weights

def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    params = {
        "objective": "multi:softprob",
        "num_class": len(np.unique(y_train)),
        "eta": 0.12,
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
    print("Training XGBoost (early stopping)...")
    bst = xgb.train(params, dtrain, num_boost_round=800, early_stopping_rounds=40, evals=[(dtrain, "train"), (dval, "val")], verbose_eval=20)
    bst.save_model(str(MODELS_DIR / "xgboost_v3.json"))
    print("Saved XGBoost ->", MODELS_DIR / "xgboost_v3.json")
    return bst

def train_catboost(X_train, y_train, X_val, y_val):
    # print("Training CatBoost (may use GPU if available)...")
    # # CatBoost requires dense or Pool object; Pool accepts CSR too
    # train_pool = Pool(X_train, label=y_train)
    # val_pool = Pool(X_val, label=y_val)
    # cb = CatBoostClassifier(
    #     iterations=1500,
    #     learning_rate=0.07,
    #     depth=8,
    #     loss_function="MultiClass",
    #     eval_metric="MultiClass",
    #     random_seed=RANDOM_STATE,
    #     early_stopping_rounds=60,
    #     task_type="GPU" if os.environ.get("COLAB_GPU", "0") == "1" else "CPU",
    #     thread_count=N_JOBS,
    #     verbose=100
    # )
    # cb.fit(train_pool, eval_set=val_pool)
    # cb.save_model(str(MODELS_DIR / "catboost_v3.cbm"))
    # print("Saved CatBoost ->", MODELS_DIR / "catboost_v3.cbm")
    # return cb
    pass

def evaluate_model_xgb(bst, X, y_true):
    d = xgb.DMatrix(X)
    probs = bst.predict(d)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")
    rep = classification_report(y_true, preds, digits=4)
    cm = confusion_matrix(y_true, preds)
    return {"accuracy": acc, "f1_weighted": f1, "report": rep, "confusion_matrix": cm.tolist()}

def evaluate_model_cb(cb, X, y_true):
    pool = Pool(X)
    preds = cb.predict(pool)
    preds = np.array(preds, dtype=int).reshape(-1)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")
    rep = classification_report(y_true, preds, digits=4)
    cm = confusion_matrix(y_true, preds)
    return {"accuracy": acc, "f1_weighted": f1, "report": rep, "confusion_matrix": cm.tolist()}

def main():
    t0 = time.time()
    X, y_raw, meta_df, n_body, n_subj = load_sparse_and_tables()
    feature_names = build_feature_names()
    y_enc, le = LabelEncoder().fit_transform(y_raw), LabelEncoder().fit(y_raw)  # capture encoder classes
    # Actually keep a LabelEncoder with classes
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_.tolist()
    print("Classes:", classes)

    # Split to train/val holdout
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.18, random_state=RANDOM_STATE, stratify=y_enc)
    print("Shapes:", X_train.shape, X_val.shape)

    # Targeted oversampling for smallest class only
    X_train_bal, y_train_bal = targeted_oversample(X_train, y_train)

    # compute sample weights on balanced set (use inverse freq)
    sample_weights = compute_sample_weights(y_train_bal)

    # Train XGBoost
    bst = train_xgboost(X_train_bal, y_train_bal, X_val, y_val, feature_names)

    # Train CatBoost
    # cb = train_catboost(X_train_bal, y_train_bal, X_val, y_val)

    # Evaluate on validation
    print("Evaluating XGBoost on validation...")
    xgb_metrics = evaluate_model_xgb(bst, X_val, y_val)
    print("Evaluating CatBoost on validation...")
    # cb_metrics = evaluate_model_cb(cb, X_val, y_val)

    # metrics = {"xgboost": xgb_metrics, "catboost": cb_metrics, "classes": classes}
    metrics = {"xgboost": xgb_metrics,  "classes": classes}
    with open(MODELS_DIR / "model_metrics_v3.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance for XGBoost (map feature indices to names)
    fmap = bst.get_score(importance_type="gain")
    feat_imp = sorted(fmap.items(), key=lambda x: x[1], reverse=True)
    with open(MODELS_DIR / "feature_importance_v3.json", "w") as f:
        json.dump(feat_imp[:300], f, indent=2)

    print("Training completed. Metrics saved to", MODELS_DIR / "model_metrics_v3.json")
    print("Elapsed (min):", (time.time()-t0)/60.0)

if __name__ == "__main__":
    main()
