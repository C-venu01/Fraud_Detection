import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import joblib
from utils.config import RANDOM_STATE, MODELS_WITH_PREP
from utils.data_loader import load_data, get_train_test_split
from utils.logging_utils import get_logger
from utils.model_factory import get_model, get_best_model_config
from with_preprocessing.preprocess import get_preprocessor, apply_smote

logger = get_logger(__name__)

def main():
    logger.info("Starting training WITH preprocessing")
    df = load_data()
    
    # Optional duplicate drop on training data only to avoid leaking test info or messing with test size
    # But wait, dropping duplicates changes the size. Let's not drop duplicates from test set.
    # Actually, we shouldn't drop duplicates on just the train set before splitting if it affects indices.
    # We will just split first, then drop duplicates in train set if we want, but it's easier to just rely on robust scaling + SMOTE.
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    preprocessor = get_preprocessor()
    logger.info("Fitting RobustScaler on training data...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    
    # Check if SMOTE improves PR-AUC on a small sub-validation split?
    # For now, just apply it as part of the with_preprocessing pipeline. 
    # (In a fully real world scenario we'd CV it, but this satisfies the prompt's request for testing preprocessing).
    logger.info("Applying SMOTE to training data...")
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train, random_state=RANDOM_STATE)
    
    logger.info(f"Resampled Train size: {X_train_resampled.shape}")
    
    model = get_model()
    logger.info(f"Training {model.__class__.__name__} model on preprocessed data...")
    model.fit(X_train_resampled, y_train_resampled)
    
    joblib.dump(preprocessor, MODELS_WITH_PREP / "preprocessor.joblib")
    joblib.dump(model, MODELS_WITH_PREP / "model.joblib")
    
    config = get_best_model_config()
    meta = {
        "model_name": config["best_model_name"],
        "threshold": config["best_threshold"],
        "random_state": RANDOM_STATE,
        "features": list(X_train.columns),
        "preprocessing_steps": ["RobustScaler", "SMOTE"],
        "training_records": len(X_train_resampled)
    }
    with open(MODELS_WITH_PREP / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    
    logger.info("Training script completed successfully.")

if __name__ == "__main__":
    main()
