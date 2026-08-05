from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier:

    def __init__(
        self,
        random_state=42,
        max_iter=1000,
    ):

        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
