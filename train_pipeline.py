#!/usr/bin/env python3
"""
Complete Training Pipeline for Email Trust & Safety
Runs end-to-end: Data Processing → Feature Engineering → Model Training → Evaluation

Usage:
    python train_pipeline.py --data-dir data/raw --output-dir models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for email classification"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
    def step1_preprocess_data(self):
        """Step 1: Preprocess raw email data"""
        logger.info("=" * 60)
        logger.info("STEP 1: Data Preprocessing")
        logger.info("=" * 60)
        
        try:
            # Run preprocessing script
            cmd = [
                sys.executable,
                "scripts/build_clean_dataset_master.py",
                "--input-dir", str(self.data_dir),
                "--output-dir", "data/processed"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            
            logger.info("✓ Data preprocessing completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Preprocessing failed: {e}")
            logger.error(e.stderr)
            return False
    
    def step2_feature_engineering(self):
        """Step 2: Extract features from processed data"""
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 60)
        
        try:
            # Run feature engineering
            cmd = [
                sys.executable,
                "scripts/feature_engineering_v2.py",
                "--train-path", "data/processed/train.parquet",
                "--val-path", "data/processed/val.parquet",
                "--test-path", "data/processed/test.parquet",
                "--output-dir", "data/features"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            
            logger.info("✓ Feature engineering completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Feature engineering failed: {e}")
            logger.error(e.stderr)
            return False
    
    def step3_train_model(self):
        """Step 3: Train XGBoost model"""
        logger.info("=" * 60)
        logger.info("STEP 3: Model Training")
        logger.info("=" * 60)
        
        try:
            # Run model training
            cmd = [
                sys.executable,
                "scripts/train_model.py",
                "--features-dir", "data/features",
                "--output-dir", str(self.output_dir),
                "--model-name", f"xgboost_v3_{self.timestamp}"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            
            logger.info("✓ Model training completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Model training failed: {e}")
            logger.error(e.stderr)
            return False
    
    def step4_validate_model(self):
        """Step 4: Validate model performance"""
        logger.info("=" * 60)
        logger.info("STEP 4: Model Validation")
        logger.info("=" * 60)
        
        try:
            # Run validation
            cmd = [
                sys.executable,
                "scripts/validate.py",
                "--model-path", f"{self.output_dir}/xgboost_v3_{self.timestamp}.json",
                "--features-dir", "data/features",
                "--output-report", f"{self.output_dir}/logs/validation_report_{self.timestamp}.json"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            
            logger.info("✓ Model validation completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Model validation failed: {e}")
            logger.error(e.stderr)
            return False
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        logger.info("=" * 60)
        logger.info("🚀 STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Data Directory: {self.data_dir}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Run all steps
        steps = [
            ("Preprocessing", self.step1_preprocess_data),
            ("Feature Engineering", self.step2_feature_engineering),
            ("Model Training", self.step3_train_model),
            ("Model Validation", self.step4_validate_model)
        ]
        
        for step_name, step_func in steps:
            success = step_func()
            if not success:
                logger.error(f"Pipeline failed at step: {step_name}")
                return False
        
        # Calculate duration
        duration = datetime.now() - start_time
        
        logger.info("=" * 60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration}")
        logger.info(f"Model saved to: {self.output_dir}/xgboost_v3_{self.timestamp}.json")
        logger.info("=" * 60)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete training pipeline for email classification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw email data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing step"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(args.data_dir, args.output_dir)
    
    # Run pipeline
    success = pipeline.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()