import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, matthews_corrcoef
)
from utils.config import RESULTS_WITHOUT_PREP, RESULTS_WITH_PREP, RESULTS_COMPARISON
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def plot_combined_roc(y_true_wout, proba_wout, y_true_with, proba_with):
    fpr_wout, tpr_wout, _ = roc_curve(y_true_wout, proba_wout)
    fpr_with, tpr_with, _ = roc_curve(y_true_with, proba_with)
    
    plt.figure()
    plt.plot(fpr_wout, tpr_wout, label=f'Without Prep (AUC = {auc(fpr_wout, tpr_wout):.4f})')
    plt.plot(fpr_with, tpr_with, label=f'With Prep (AUC = {auc(fpr_with, tpr_with):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_COMPARISON / "combined_roc_curve.png")
    plt.close()

def plot_combined_pr(y_true_wout, proba_wout, y_true_with, proba_with):
    p_wout, r_wout, _ = precision_recall_curve(y_true_wout, proba_wout)
    p_with, r_with, _ = precision_recall_curve(y_true_with, proba_with)
    
    # Calculate PR-AUC manually if not available directly via average_precision_score here, but we just use auc() for plot
    pr_auc_wout = auc(r_wout, p_wout)
    pr_auc_with = auc(r_with, p_with)
    
    plt.figure()
    plt.plot(r_wout, p_wout, label=f'Without Prep (AUC = {pr_auc_wout:.4f})')
    plt.plot(r_with, p_with, label=f'With Prep (AUC = {pr_auc_with:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Combined Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(RESULTS_COMPARISON / "combined_pr_curve.png")
    plt.close()

def evaluate_thresholds(y_true, proba, name, output_dir):
    thresholds = np.linspace(0.01, 0.99, 99)
    res = []
    
    precisions, recalls, f1s, mccs = [], [], [], []
    
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        mcc = matthews_corrcoef(y_true, y_pred)
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        mccs.append(mcc)
        
        res.append({
            "Threshold": t, "Precision": p, "Recall": r, "F1": f1, "MCC": mcc,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })
        
    df = pd.DataFrame(res)
    df.to_csv(output_dir / "threshold_metrics.csv", index=False)
    
    def plot_metric(metric_arr, title, filename):
        plt.figure()
        plt.plot(thresholds, metric_arr, marker='o', markersize=2)
        plt.xlabel('Threshold')
        plt.ylabel(title)
        plt.title(f'{title} vs Threshold - {name}')
        plt.grid(True)
        plt.savefig(output_dir / filename)
        plt.close()
        
    plot_metric(precisions, "Precision", "precision_vs_threshold.png")
    plot_metric(recalls, "Recall", "recall_vs_threshold.png")
    plot_metric(f1s, "F1 Score", "f1_vs_threshold.png")
    plot_metric(mccs, "MCC", "mcc_vs_threshold.png")
    
    return df

def generate_confusion_matrix_comparison(y_true_wout, y_pred_wout, y_true_with, y_pred_with):
    cm_wout = confusion_matrix(y_true_wout, y_pred_wout)
    cm_with = confusion_matrix(y_true_with, y_pred_with)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].matshow(cm_wout, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm_wout.shape[0]):
        for j in range(cm_wout.shape[1]):
            axes[0].text(x=j, y=i,s=cm_wout[i, j], va='center', ha='center')
    axes[0].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')
    axes[0].set_title('Without Preprocessing')
    
    axes[1].matshow(cm_with, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm_with.shape[0]):
        for j in range(cm_with.shape[1]):
            axes[1].text(x=j, y=i,s=cm_with[i, j], va='center', ha='center')
    axes[1].set_xlabel('Predicted label')
    axes[1].set_ylabel('True label')
    axes[1].set_title('With Preprocessing')
    
    plt.tight_layout()
    plt.savefig(RESULTS_COMPARISON / "confusion_matrix_comparison.png")
    plt.close()

def main():
    logger.info("Starting comparison of models...")
    
    with open(RESULTS_WITHOUT_PREP / "metrics.json", "r") as f: metrics_without = json.load(f)
    with open(RESULTS_WITH_PREP / "metrics.json", "r") as f: metrics_with = json.load(f)
    
    preds_wout = pd.read_csv(RESULTS_WITHOUT_PREP / "test_predictions.csv")
    preds_with = pd.read_csv(RESULTS_WITH_PREP / "test_predictions.csv")
    
    for name, df_preds, metrics in [("Without", preds_wout, metrics_without), ("With", preds_with, metrics_with)]:
        meta_path = (RESULTS_WITHOUT_PREP.parent.parent / "models" / f"{name.lower()}_preprocessing" / "metadata.json")
        with open(meta_path, "r") as f: meta = json.load(f)
        thresh = meta.get("threshold", 0.5)
        
        y_pred = (df_preds["y_proba"] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(df_preds["y_true"], y_pred).ravel()
        
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics["TP"] = int(tp)
        metrics["TN"] = int(tn)
        metrics["FP"] = int(fp)
        metrics["FN"] = int(fn)
        
        metrics["MCC"] = matthews_corrcoef(df_preds["y_true"], y_pred)
        
        # Calculate balanced accuracy
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["Balanced Accuracy"] = (tpr + tnr) / 2
        
    metrics_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC", "Balanced Accuracy", "FPR", "FNR", "TP", "FP", "FN", "TN"]
    
    data = []
    for metric in metrics_names:
        val_without = metrics_without.get(metric, 0)
        val_with = metrics_with.get(metric, 0)
        diff = val_with - val_without
        
        data.append({
            "Metric": metric,
            "Without Preprocessing": val_without if isinstance(val_without, int) else round(val_without, 4),
            "With Preprocessing": val_with if isinstance(val_with, int) else round(val_with, 4),
            "Difference": diff if isinstance(diff, int) else round(diff, 4)
        })
        
    df_comparison = pd.DataFrame(data)
    df_comparison.to_csv(RESULTS_COMPARISON / "comparison.csv", index=False)
    
    # Visualizations
    plot_combined_roc(preds_wout["y_true"], preds_wout["y_proba"], preds_with["y_true"], preds_with["y_proba"])
    plot_combined_pr(preds_wout["y_true"], preds_wout["y_proba"], preds_with["y_true"], preds_with["y_proba"])
    
    # Threshold curves
    evaluate_thresholds(preds_wout["y_true"], preds_wout["y_proba"], "Without Preprocessing", RESULTS_WITHOUT_PREP)
    evaluate_thresholds(preds_with["y_true"], preds_with["y_proba"], "With Preprocessing", RESULTS_WITH_PREP)
    
    # Confusion matrix comparison
    with open(RESULTS_WITHOUT_PREP.parent.parent / "models" / "without_preprocessing" / "metadata.json", "r") as f: thresh_wout = json.load(f)["threshold"]
    with open(RESULTS_WITH_PREP.parent.parent / "models" / "with_preprocessing" / "metadata.json", "r") as f: thresh_with = json.load(f)["threshold"]
    
    generate_confusion_matrix_comparison(
        preds_wout["y_true"], (preds_wout["y_proba"] >= thresh_wout).astype(int),
        preds_with["y_true"], (preds_with["y_proba"] >= thresh_with).astype(int)
    )
    
    # Report
    with open(RESULTS_WITH_PREP.parent.parent / "models" / "without_preprocessing" / "metadata.json", "r") as f: model_name = json.load(f)["model_name"]
    
    report = f"Selected Model: {model_name}\n\n"
    report += "Without Preprocessing:\n"
    for m in ["PR-AUC", "Recall", "Precision", "F1", "MCC"]:
        report += f"{m} = {metrics_without[m]:.4f}\n"
        
    report += "\nWith Preprocessing:\n"
    for m in ["PR-AUC", "Recall", "Precision", "F1", "MCC"]:
        report += f"{m} = {metrics_with[m]:.4f}\n"
        
    f1_diff = metrics_with["F1"] - metrics_without["F1"]
    pr_auc_diff = metrics_with["PR-AUC"] - metrics_without["PR-AUC"]
    
    report += "\nComparison Conclusion:\n"
    if metrics_without["F1"] > metrics_with["F1"] and metrics_without["Precision"] > metrics_with["Precision"]:
        if metrics_with["Recall"] > metrics_without["Recall"]:
            report += "Without preprocessing achieved higher precision and F1, while preprocessing achieved higher recall. Therefore the choice depends on whether minimizing false positives or minimizing missed fraud is more important."
        else:
            report += "Without preprocessing outperformed preprocessing across primary metrics (F1, Precision, Recall)."
    elif metrics_with["F1"] > metrics_without["F1"]:
        report += "Preprocessing yielded a superior F1 score and better overall balance of precision and recall."
    else:
        report += "The two approaches yielded mixed results. Please review the detailed threshold curves to select the best trade-off."
        
    with open(RESULTS_COMPARISON / "comparison_report.txt", "w") as f:
        f.write(report)
        
    logger.info(f"Comparison Table:\n\n{df_comparison.to_markdown(index=False)}")
    logger.info(f"Report:\n{report}")
    logger.info("Comparison completed successfully.")

if __name__ == "__main__":
    main()
