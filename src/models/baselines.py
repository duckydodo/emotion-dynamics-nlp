import pandas as pd


class MajorityClassifier:

    def __init__(self):
        self.majority_class = None

    def fit(self, X, y):
        self.majority_class = y.mode().iloc[0]

    def predict(self, X):
        return pd.Series(
            [self.majority_class] * len(X),
            index=X.index,
        )
