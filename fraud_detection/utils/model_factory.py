import json
from pathlib import Path
from utils.config import MODEL_SELECTION_DIR, RANDOM_STATE

def get_best_model_config():
    config_path = MODEL_SELECTION_DIR / "best_model_config.json"
    if not config_path.exists():
        raise FileNotFoundError("Model config not found. Run model_selection/select_model.py first.")
    with open(config_path, "r") as f:
        return json.load(f)

def get_model():
    """Instantiates the best model selected during model selection."""
    config = get_best_model_config()
    model_name = config["best_model_name"]
    
    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, max_depth=10)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, scale_pos_weight=config.get("scale_pos_weight", 1), 
                             random_state=RANDOM_STATE, n_jobs=-1, max_depth=6, tree_method="hist")
    else:
        raise ValueError(f"Unknown model: {model_name}")
