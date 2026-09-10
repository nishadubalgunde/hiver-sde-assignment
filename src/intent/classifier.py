import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class IntentClassifier:

    def __init__(
        self,
        training_path="data/processed/apple_silver_labeled.csv"
    ):
        self.training_path = training_path

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        self._train()

    def _train(self):
        df = pd.read_csv(self.training_path)

        texts = df["text"].fillna("").astype(str)
        labels = df["intent"].astype(str)

        X = self.vectorizer.fit_transform(texts)

        self.model.fit(X, labels)

    def predict(self, text):
        text = str(text)

        X = self.vectorizer.transform([text])

        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        classes = self.model.classes_

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        top_predictions = [
            {
                "intent": intent,
                "confidence": round(float(confidence), 4)
            }
            for intent, confidence in ranked[:3]
        ]

        return {
            "intent": prediction,
            "confidence": round(
                float(max(probabilities)), 4
            ),
            "top_predictions": top_predictions
        }


if __name__ == "__main__":

    classifier = IntentClassifier()

    test_messages = [
        "My iPhone battery is draining very quickly",
        "I cannot connect my iPhone to Wi-Fi",
        "My Apple ID is locked",
        "How do I turn on dark mode?"
    ]

    for message in test_messages:

        result = classifier.predict(message)

        print("\nMessage:")
        print(message)

        print("Prediction:")
        print(result)