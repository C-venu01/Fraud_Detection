import pandas as pd
import json
from sklearn.datasets import fetch_openml
from utils.config import RAW_DATA_DIR, TARGET_COLUMN, RESULTS_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_data() -> pd.DataFrame:
    """Loads the credit card fraud dataset. Downloads it if not present locally."""
    csv_path = RAW_DATA_DIR / "creditcard.csv"
    
    if csv_path.exists():
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        logger.info("Dataset not found locally. Downloading from OpenML (this may take a minute)...")
        # dataset ID 1597 is the credit card fraud dataset, but it's safer to fetch by name
        data = fetch_openml(name='creditcard', version=1, parser='auto', as_frame=True)
        df = data.frame
        logger.info(f"Saving downloaded dataset to {csv_path}")
        df.to_csv(csv_path, index=False)
        
    return df

def generate_dataset_statistics(df: pd.DataFrame) -> None:
    """Calculates dataset statistics and saves them to a JSON file."""
    stats = {}
    stats["total_records"] = len(df)
    
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN])
        fraud_records = int(df[TARGET_COLUMN].sum())
        legit_records = stats["total_records"] - fraud_records
        stats["legitimate_records"] = legit_records
        stats["fraud_records"] = fraud_records
        stats["fraud_percentage"] = (fraud_records / stats["total_records"]) * 100
    
    stats["number_of_features"] = len(df.columns) - (1 if TARGET_COLUMN in df.columns else 0)
    stats["feature_names"] = list(df.columns)
    stats["data_types"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    stats["missing_values"] = df.isnull().sum().to_dict()
    stats["duplicate_records"] = int(df.duplicated().sum())
    
    stats_file = RESULTS_DIR / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Saved dataset statistics to {stats_file}")

def get_train_test_split(df: pd.DataFrame):
    """Returns a consistent train/test split. Indices are deterministic based on random_state."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN])
        
    numeric_df = df.select_dtypes(include=['number'])
    X = numeric_df.drop(columns=[TARGET_COLUMN])
    y = numeric_df[TARGET_COLUMN]
    
    from utils.config import RANDOM_STATE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
