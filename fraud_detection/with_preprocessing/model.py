from sklearn.linear_model import LogisticRegression

def get_model(random_state: int = 42) -> LogisticRegression:
    """Returns the same logistic regression model as the baseline."""
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced"
    )
