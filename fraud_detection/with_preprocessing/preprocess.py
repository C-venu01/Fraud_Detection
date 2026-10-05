from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd

def get_preprocessor():
    """
    Returns a preprocessing pipeline.
    For fraud detection, we often use RobustScaler to handle outliers.
    """
    return Pipeline([
        ('scaler', RobustScaler())
    ])

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Applies SMOTE to the training data to handle class imbalance.
    Note: SMOTE is only applied to the training set, never the test set.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
