import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef
from utils.config import RANDOM_STATE, MODEL_SELECTION_DIR
from utils.data_loader import load_data, get_train_test_split
from utils.logging_utils import get_logger
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

def evaluate_pipeline(model_class, model_kwargs, X, y, use_smote, use_scale_pos_weight, use_class_weight, name):
    logger.info(f"Evaluating Pipeline: {name} (SMOTE: {use_smote}, scale_pos_weight: {use_scale_pos_weight}, class_weight: {use_class_weight})")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    oof_y_true = []
    oof_y_proba = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        if use_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
        kwargs = model_kwargs.copy()
        
        if use_scale_pos_weight and "scale_pos_weight" in kwargs:
            # calculate strictly from train fold to prevent leakage
            kwargs["scale_pos_weight"] = (len(y_train) - y_train.sum()) / y_train.sum()
            
        if use_class_weight and "class_weight" in kwargs:
            kwargs["class_weight"] = "balanced"
            
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_val)[:, 1]
        oof_y_true.extend(y_val)
        oof_y_proba.extend(proba)
        
    oof_y_true = np.array(oof_y_true)
    oof_y_proba = np.array(oof_y_proba)
    
    pr_auc = average_precision_score(oof_y_true, oof_y_proba)
    
    # Calculate scores at default threshold 0.50
    pred = (oof_y_proba >= 0.5).astype(int)
    recall = recall_score(oof_y_true, pred)
    precision = precision_score(oof_y_true, pred, zero_division=0)
    f1 = f1_score(oof_y_true, pred)
    
    logger.info(f"{name} PR-AUC (CV): {pr_auc:.4f}")
    return pr_auc, recall, precision, f1, oof_y_true, oof_y_proba

def main():
    logger.info("Starting model selection process...")
    df = load_data()
    # The requirement said: "Use: 80% development data, 20% final untouched test data. Then development data -> training + validation/CV"
    # get_train_test_split gives us 80% train, 20% test. We will use the 80% for CV here!
    X_dev, _, y_dev, _ = get_train_test_split(df)
    
    xgb_kwargs = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "scale_pos_weight": 1 # placeholder
    }
    
    rf_kwargs = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": None
    }
    
    lr_kwargs = {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "class_weight": None
    }
    
    pipelines = [
        # XGBoost Strategies
        {"name": "XGBoost (No Resampling + scale_pos_weight)", "model": XGBClassifier, "kwargs": xgb_kwargs, "smote": False, "spw": True, "cw": False},
        {"name": "XGBoost (SMOTE + no scale_pos_weight)", "model": XGBClassifier, "kwargs": xgb_kwargs, "smote": True, "spw": False, "cw": False},
        {"name": "XGBoost (SMOTE + scale_pos_weight)", "model": XGBClassifier, "kwargs": xgb_kwargs, "smote": True, "spw": True, "cw": False},
        {"name": "XGBoost (No Resampling)", "model": XGBClassifier, "kwargs": xgb_kwargs, "smote": False, "spw": False, "cw": False},
        
        # RandomForest Strategies
        {"name": "RandomForest (No Resampling + class_weight)", "model": RandomForestClassifier, "kwargs": rf_kwargs, "smote": False, "spw": False, "cw": True},
        {"name": "RandomForest (SMOTE)", "model": RandomForestClassifier, "kwargs": rf_kwargs, "smote": True, "spw": False, "cw": False},
        
        # LogisticRegression Strategies
        {"name": "LogisticRegression (No Resampling + class_weight)", "model": LogisticRegression, "kwargs": lr_kwargs, "smote": False, "spw": False, "cw": True}
    ]
    
    results = []
    best_pr_auc = -1
    best_pipeline = None
    best_y_true = None
    best_y_proba = None
    
    for p in pipelines:
        pr_auc, recall, precision, f1, oof_y_true, oof_y_proba = evaluate_pipeline(
            p["model"], p["kwargs"], X_dev, y_dev, p["smote"], p["spw"], p["cw"], p["name"]
        )
        results.append({
            "Pipeline": p["name"],
            "PR-AUC": pr_auc,
            "Precision (0.5)": precision,
            "Recall (0.5)": recall,
            "F1 (0.5)": f1
        })
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_pipeline = p
            best_y_true = oof_y_true
            best_y_proba = oof_y_proba
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(MODEL_SELECTION_DIR / "model_comparison.csv", index=False)
    
    logger.info(f"\nModel Comparison:\n{df_results.to_markdown(index=False)}")
    logger.info(f"Selected best pipeline based on PR-AUC: {best_pipeline['name']}")
    
    # Threshold Analysis using validation probabilities
    thresholds = np.linspace(0.01, 0.99, 99)
    thresh_res = []
    for t in thresholds:
        pred = (best_y_proba >= t).astype(int)
        
        tp = np.sum((pred == 1) & (best_y_true == 1))
        fp = np.sum((pred == 1) & (best_y_true == 0))
        fn = np.sum((pred == 0) & (best_y_true == 1))
        tn = np.sum((pred == 0) & (best_y_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mcc = matthews_corrcoef(best_y_true, pred)
        
        thresh_res.append({
            "Threshold": t,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })
        
    df_thresh = pd.DataFrame(thresh_res)
    
    best_f1_row = df_thresh.loc[df_thresh['F1'].idxmax()]
    best_mcc_row = df_thresh.loc[df_thresh['MCC'].idxmax()]
    
    prec_80 = df_thresh[df_thresh['Recall'] >= 0.80]
    best_prec_at_80_recall = prec_80.loc[prec_80['Precision'].idxmax()] if not prec_80.empty else None
    
    rec_80 = df_thresh[df_thresh['Precision'] >= 0.80]
    best_recall_at_80_prec = rec_80.loc[rec_80['Recall'].idxmax()] if not rec_80.empty else None
    
    # We choose F1 as the primary objective for the final threshold
    selected_threshold = float(best_f1_row['Threshold'])
    logger.info(f"Selected final Threshold (max F1): {selected_threshold:.3f}")
    
    report = "Threshold Optimization Report:\n"
    report += f"Best F1 Threshold: {best_f1_row['Threshold']:.3f} (F1: {best_f1_row['F1']:.4f}, Prec: {best_f1_row['Precision']:.4f}, Rec: {best_f1_row['Recall']:.4f})\n"
    report += f"Best MCC Threshold: {best_mcc_row['Threshold']:.3f} (MCC: {best_mcc_row['MCC']:.4f})\n"
    if best_prec_at_80_recall is not None:
        report += f"Best Precision at >= 80% Recall: {best_prec_at_80_recall['Precision']:.4f} (Threshold: {best_prec_at_80_recall['Threshold']:.3f})\n"
    else:
        report += "Best Precision at >= 80% Recall: Impossible constraint\n"
        
    if best_recall_at_80_prec is not None:
        report += f"Best Recall at >= 80% Precision: {best_recall_at_80_prec['Recall']:.4f} (Threshold: {best_recall_at_80_prec['Threshold']:.3f})\n"
    else:
        report += "Best Recall at >= 80% Precision: Impossible constraint\n"
        
    logger.info("\n" + report)
    
    # Map back to model names used in factory
    model_name = "XGBoost" if "XGBoost" in best_pipeline["name"] else "RandomForest" if "RandomForest" in best_pipeline["name"] else "LogisticRegression"
    
    config = {
        "best_model_name": model_name,
        "best_threshold": selected_threshold,
        "smote": best_pipeline["smote"],
        "scale_pos_weight": best_pipeline["spw"],
        "class_weight": best_pipeline["cw"]
    }
    with open(MODEL_SELECTION_DIR / "best_model_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    with open(MODEL_SELECTION_DIR / "model_selection_report.txt", "w") as f:
        f.write(f"Candidate Pipelines Evaluation:\n{df_results.to_markdown(index=False)}\n\n")
        f.write(f"Selected Pipeline: {best_pipeline['name']}\n\n")
        f.write(report)

if __name__ == "__main__":
    main()
