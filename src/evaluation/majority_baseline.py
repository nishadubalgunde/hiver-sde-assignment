import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"


def main():
    df = pd.read_csv(GOLDEN_PATH)

    y_true = df["intent"].astype(str)

    # Find the most frequent intent
    majority_class = y_true.value_counts().idxmax()

    # Predict the same class for every example
    y_pred = [majority_class] * len(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("=== Majority-Class Baseline ===")
    print(f"Majority class: {majority_class}")
    print(f"Examples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()