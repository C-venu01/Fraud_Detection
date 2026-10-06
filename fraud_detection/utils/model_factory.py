import json
from pathlib import Path
from utils.config import MODEL_SELECTION_DIR, RANDOM_STATE

def get_best_model_config():
    config_path = MODEL_SELECTION_DIR / "best_model_config.json"
    if not config_path.exists():
        raise FileNotFoundError("Model config not found. Run model_selection/select_model.py first.")
    with open(config_path, "r") as f:
        return json.load(f)

def get_model(y_train=None):
    """Instantiates the best model selected during model selection."""
    config = get_best_model_config()
    model_name = config["best_model_name"]
    
    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, class_weight='balanced' if config.get("class_weight") else None, random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, class_weight='balanced' if config.get("class_weight") else None, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        spw = 1
        if config.get("scale_pos_weight") and y_train is not None:
            spw = (len(y_train) - y_train.sum()) / y_train.sum()
            
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
