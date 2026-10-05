from pathlib import Path

# Project root is two levels up from this file (fraud_detection/utils/config.py -> fraud_detection)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_WITHOUT_PREP = MODELS_DIR / "without_preprocessing"
MODELS_WITH_PREP = MODELS_DIR / "with_preprocessing"

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_SELECTION_DIR = RESULTS_DIR / "model_selection"
RESULTS_WITHOUT_PREP = RESULTS_DIR / "without_preprocessing"
RESULTS_WITH_PREP = RESULTS_DIR / "with_preprocessing"
RESULTS_COMPARISON = RESULTS_DIR / "comparison"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_WITHOUT_PREP, MODELS_WITH_PREP, 
          MODEL_SELECTION_DIR, RESULTS_WITHOUT_PREP, RESULTS_WITH_PREP, RESULTS_COMPARISON]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COLUMN = "Class"
