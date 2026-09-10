import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from intent.classifier import IntentClassifier

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from intent.classifier import IntentClassifier

GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"
OUTPUT_PATH = "results/intent_predictions.csv"


def main():
    print("Loading classifier...")
    classifier = IntentClassifier()

    df = pd.read_csv(GOLDEN_PATH)

    predictions = []

    for _, row in df.iterrows():
        result = classifier.predict(row["target_message"])

        predictions.append({
            "conversation_id": row["conversation_id"],
            "tweet_id": row["tweet_id"],
            "target_message": row["target_message"],
            "true_intent": row["intent"],
            "predicted_intent": result["intent"],
            "confidence": result["confidence"]
        })

    results = pd.DataFrame(predictions)

    y_true = results["true_intent"]
    y_pred = results["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n=== Intent Classifier Evaluation ===")
    print(f"Golden examples: {len(results)}")
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

    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nPredictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()