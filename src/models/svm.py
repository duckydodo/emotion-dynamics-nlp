from sklearn.svm import LinearSVC


class LinearSVMClassifier:

    def __init__(
        self,
        random_state=42,
    ):

        self.model = LinearSVC(
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
