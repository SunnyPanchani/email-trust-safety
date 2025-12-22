#!/bin/bash

# ============================================================
#  Email Trust Safety – Project Folder Structure Generator
# ============================================================

PROJECT_ROOT="email-trust-safety"

echo "📂 Creating project structure: $PROJECT_ROOT"
mkdir -p $PROJECT_ROOT

# -----------------------
# Data folders
# -----------------------
mkdir -p $PROJECT_ROOT/data/raw/enron_cmu
mkdir -p $PROJECT_ROOT/data/raw/spamassassin
mkdir -p $PROJECT_ROOT/data/raw/enron_spam_kaggle
mkdir -p $PROJECT_ROOT/data/raw/phishing_kaggle
mkdir -p $PROJECT_ROOT/data/processed

# -----------------------
# Scripts
# -----------------------
mkdir -p $PROJECT_ROOT/scripts
touch $PROJECT_ROOT/scripts/parsers.py
touch $PROJECT_ROOT/scripts/preprocess_all.py
touch $PROJECT_ROOT/scripts/validate.py
touch $PROJECT_ROOT/scripts/feature_engineering_v2.py
touch $PROJECT_ROOT/scripts/train_models.py
touch $PROJECT_ROOT/scripts/graph_analytics.py

# -----------------------
# Models
# -----------------------
mkdir -p $PROJECT_ROOT/models/spam_classifier
mkdir -p $PROJECT_ROOT/models/anomaly_detector
mkdir -p $PROJECT_ROOT/models/graph_models

# -----------------------
# API (FastAPI)
# -----------------------
mkdir -p $PROJECT_ROOT/api/routers
mkdir -p $PROJECT_ROOT/api/schemas
touch $PROJECT_ROOT/api/main.py

# -----------------------
# Dashboard (Streamlit)
# -----------------------
mkdir -p $PROJECT_ROOT/dashboard
touch $PROJECT_ROOT/dashboard/app.py

# -----------------------
# Notebooks
# -----------------------
mkdir -p $PROJECT_ROOT/notebooks
touch $PROJECT_ROOT/notebooks/01_data_exploration.ipynb
touch $PROJECT_ROOT/notebooks/02_feature_analysis.ipynb
touch $PROJECT_ROOT/notebooks/03_model_experiments.ipynb

# -----------------------
# Tests
# -----------------------
mkdir -p $PROJECT_ROOT/tests
touch $PROJECT_ROOT/tests/test_parsers.py
touch $PROJECT_ROOT/tests/test_preprocessing.py
touch $PROJECT_ROOT/tests/test_models.py

# -----------------------
# Root-level project files
# -----------------------
touch $PROJECT_ROOT/requirements.txt
touch $PROJECT_ROOT/README.md
touch $PROJECT_ROOT/README_PREPROCESSING.md
touch $PROJECT_ROOT/run_preprocessing.py
touch $PROJECT_ROOT/run_preprocessing.sh
touch $PROJECT_ROOT/.gitignore

echo "✅ Project structure created successfully!"
