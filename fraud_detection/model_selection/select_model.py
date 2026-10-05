import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, recall_score, f1_score, precision_score
from utils.config import RANDOM_STATE, MODEL_SELECTION_DIR
from utils.data_loader import load_data, get_train_test_split
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_train, y_train, name):
    logger.info(f"Evaluating {name} using 3-fold cross validation...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    y_proba = cross_val_predict(model, X_train, y_train, cv=skf, method='predict_proba')[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    pr_auc = average_precision_score(y_train, y_proba)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    
    logger.info(f"{name} - PR-AUC: {pr_auc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return pr_auc, recall, f1, y_proba

def main():
    logger.info("Starting model selection process...")
    df = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, max_depth=10),
        "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1, max_depth=6, tree_method="hist")
    }
    
    results = []
    best_pr_auc = -1
    best_model_name = ""
    best_model_proba = None
    
    for name, model in models.items():
        pr_auc, recall, f1, proba = evaluate_model(model, X_train, y_train, name)
        results.append({"Model": name, "PR-AUC": pr_auc, "Recall": recall, "F1": f1})
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_model_proba = proba
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(MODEL_SELECTION_DIR / "model_comparison.csv", index=False)
    
    import tabulate
    logger.info(f"\nModel Comparison:\n{df_results.to_markdown(index=False)}")
    logger.info(f"Selected best model based on PR-AUC: {best_model_name}")
    
    # Threshold analysis
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_results = []
    for t in thresholds:
        pred = (best_model_proba >= t).astype(int)
        thresh_results.append({
            "Threshold": t,
            "Precision": precision_score(y_train, pred, zero_division=0),
            "Recall": recall_score(y_train, pred),
            "F1": f1_score(y_train, pred)
        })
    df_thresh = pd.DataFrame(thresh_results)
    logger.info(f"\nThreshold analysis for {best_model_name} (using out-of-fold validation probas):\n{df_thresh.to_markdown(index=False)}")
    
    # Select threshold maximizing F1
    best_thresh_row = df_thresh.loc[df_thresh['F1'].idxmax()]
    best_thresh = float(best_thresh_row['Threshold'])
    logger.info(f"Selected Threshold: {best_thresh}")
    
    config = {
        "best_model_name": best_model_name,
        "best_threshold": best_thresh,
        "scale_pos_weight": float(pos_weight) if best_model_name == "XGBoost" else None
    }
    with open(MODEL_SELECTION_DIR / "best_model_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    with open(MODEL_SELECTION_DIR / "model_selection_report.txt", "w") as f:
        f.write(f"Candidate Models Evaluation:\n{df_results.to_markdown(index=False)}\n\n")
        f.write(f"Selected Model: {best_model_name}\n")
        f.write(f"Selected Threshold: {best_thresh}\n")

if __name__ == "__main__":
    main()
