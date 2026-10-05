import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import joblib
from utils.config import RANDOM_STATE, MODELS_WITHOUT_PREP
from utils.data_loader import load_data, generate_dataset_statistics, get_train_test_split
from utils.logging_utils import get_logger
from utils.model_factory import get_model, get_best_model_config

logger = get_logger(__name__)

def main():
    logger.info("Starting training WITHOUT preprocessing")
    df = load_data()
    generate_dataset_statistics(df)
    
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    model = get_model()
    logger.info(f"Training {model.__class__.__name__} model...")
    model.fit(X_train, y_train)
    
    model_path = MODELS_WITHOUT_PREP / "model.joblib"
    joblib.dump(model, model_path)
    
    config = get_best_model_config()
    meta = {
        "model_name": config["best_model_name"],
        "threshold": config["best_threshold"],
        "random_state": RANDOM_STATE,
        "features": list(X_train.columns),
        "training_records": len(X_train)
    }
    with open(MODELS_WITHOUT_PREP / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
