import os
import pandas as pd


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"
RESULTS_PATH = "results/agent_predictions.csv"


def main():

    print("Loading Agent Predictions...")
    predictions = pd.read_csv(RESULTS_PATH)

    print(f"Prediction examples: {len(predictions)}")

    # --------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------

    required_columns = [
        "true_intent",
        "predicted_intent",
        "target_message",
        "intent_confidence",
        "decision"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:

        print("\nMissing columns:")
        print(missing_columns)

        print("\nAvailable columns:")
        print(predictions.columns.tolist())

        return

    # --------------------------------------------------
    # COMPARE PREDICTIONS
    # --------------------------------------------------

    predictions["correct"] = (
        predictions["true_intent"]
        == predictions["predicted_intent"]
    )

    failures = predictions[
        predictions["correct"] == False
    ].copy()

    print("\n" + "=" * 80)
    print("FAILURE SUMMARY")
    print("=" * 80)

    print(
        f"Incorrect predictions: {len(failures)}"
    )

    print(
        f"Correct predictions: "
        f"{len(predictions) - len(failures)}"
    )

    # --------------------------------------------------
    # TOP CONFUSION PAIRS
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP CONFUSION PAIRS")
    print("=" * 80)

    confusion_pairs = (
        failures
        .groupby(
            [
                "true_intent",
                "predicted_intent"
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        confusion_pairs
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # FAILURES BY TRUE INTENT
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FAILURES BY TRUE INTENT")
    print("=" * 80)

    failures_by_intent = (
        failures["true_intent"]
        .value_counts()
        .reset_index()
    )

    failures_by_intent.columns = [
        "intent",
        "failures"
    ]

    print(
        failures_by_intent
        .to_string(index=False)
    )

    # --------------------------------------------------
    # TOP FAILURE EXAMPLES
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP FAILURE EXAMPLES")
    print("=" * 80)

    for number, (_, row) in enumerate(
        failures.head(30).iterrows(),
        start=1
    ):

        print("\n" + "-" * 80)

        print(f"Failure #{number}")

        print("\nCustomer message:")
        print(row["target_message"])

        print("\nTrue intent:")
        print(row["true_intent"])

        print("\nPredicted intent:")
        print(row["predicted_intent"])

        print("\nConfidence:")
        print(row["intent_confidence"])

        print("\nSimilarity:")
        print(row["top_case_similarity"])

        print("\nDecision:")
        print(row["decision"])

        print("\nDecision reason:")
        print(row["decision_reason"])

        print("\nDraft reply:")
        print(row["draft_reply"])

    # --------------------------------------------------
    # SAVE FAILURE ANALYSIS
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    output_path = "results/failure_analysis.csv"

    failures.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()