import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from utils.config import MODELS_WITHOUT_PREP, RESULTS_WITHOUT_PREP
from utils.data_loader import load_data, get_train_test_split
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting evaluation WITHOUT preprocessing")
    
    model_path = MODELS_WITHOUT_PREP / "model.joblib"
    meta_path = MODELS_WITHOUT_PREP / "metadata.json"
    
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    threshold = meta.get("threshold", 0.5)
    
    df = load_data()
    _, X_test, _, y_test = get_train_test_split(df)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba)
    }
    
    with open(RESULTS_WITHOUT_PREP / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Save probs and truth for comparison script
    pd.DataFrame({"y_true": y_test, "y_proba": y_proba}).to_csv(RESULTS_WITHOUT_PREP / "test_predictions.csv", index=False)

if __name__ == "__main__":
    main()
