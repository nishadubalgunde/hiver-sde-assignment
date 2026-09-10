import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


SILVER_PATH = "data/processed/apple_silver_labeled.csv"
GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"


def main():
    silver = pd.read_csv(SILVER_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    X_train = silver["text"].fillna("").astype(str)
    y_train = silver["intent"].astype(str)

    X_test = golden["target_message"].fillna("").astype(str)
    y_test = golden["intent"].astype(str)

    print("=== Dataset ===")
    print(f"Silver training examples: {len(X_train)}")
    print(f"Golden evaluation examples: {len(X_test)}")

    # Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # Train Logistic Regression classifier
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    # Evaluate ONLY on the untouched Golden Set
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n=== TF-IDF + Logistic Regression Baseline ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()