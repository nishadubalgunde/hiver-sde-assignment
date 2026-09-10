import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"


def main():
    df = pd.read_csv(GOLDEN_PATH)

    X = df["target_message"].fillna("").astype(str)
    y = df["intent"].astype(str)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    # NOTE:
    # For now this is a baseline evaluation on the Golden Set.
    # We will later create a proper train/development split from
    # non-Golden data and keep this 200-example set strictly held out.
    model.fit(X, y)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print("=== TF-IDF + Logistic Regression Baseline ===")
    print(f"Examples: {len(y)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()