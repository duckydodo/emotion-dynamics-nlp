from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatures:

    def __init__(
        self,
        max_features=None,
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        sublinear_tf=False,
    ):

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
