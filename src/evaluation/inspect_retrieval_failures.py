import pandas as pd


RESULTS_PATH = "results/retrieval_intent_predictions.csv"


def inspect_failures():

    df = pd.read_csv(RESULTS_PATH)

    actionable = df[
        df["true_intent"] != "other_unclear"
    ].copy()

    failures = actionable[
        actionable["intent_match_at_1"] == 0
    ].copy()

    failures = failures.sort_values(
        "top1_similarity",
        ascending=False
    )

    print("=" * 80)
    print("RETRIEVAL FAILURE INSPECTION")
    print("=" * 80)

    print(
        f"Actionable examples: {len(actionable)}"
    )

    print(
        f"Top-1 failures: {len(failures)}"
    )

    print("\nShowing 20 highest-similarity failures")
    print("=" * 80)

    for i, (_, row) in enumerate(
        failures.head(20).iterrows(),
        start=1
    ):

        print(f"\n{'-' * 80}")
        print(f"FAILURE #{i}")

        print(
            f"Similarity: "
            f"{row['top1_similarity']}"
        )

        print(
            f"True intent: "
            f"{row['true_intent']}"
        )

        print(
            f"Retrieved intent: "
            f"{row['retrieved_intent_1']}"
        )

        print(
            f"Customer message:"
        )

        print(
            row["target_message"]
        )

        print(
            f"\nTop-3 retrieved intents:"
        )

        print(
            f"1. {row['retrieved_intent_1']}"
        )

        print(
            f"2. {row['retrieved_intent_2']}"
        )

        print(
            f"3. {row['retrieved_intent_3']}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    inspect_failures()