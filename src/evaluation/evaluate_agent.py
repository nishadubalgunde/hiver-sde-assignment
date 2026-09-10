import os
import sys
import json
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from agent import AppleSupportAgent


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"
OUTPUT_PATH = "results/agent_predictions.csv"


def main():

    print("Loading Golden Set...")

    golden = pd.read_csv(GOLDEN_PATH)

    print(f"Golden examples: {len(golden)}")

    agent = AppleSupportAgent()

    results = []

    for index, row in golden.iterrows():

        message = str(row["target_message"])
        true_intent = str(row["intent"])

        print(
            f"\nProcessing {index + 1}/{len(golden)}"
        )

        result = agent.handle_message(message)

        top_case_similarity = None

        if result["historical_cases"]:

            top_case_similarity = (
                result["historical_cases"][0]["similarity"]
            )

        results.append({

            "conversation_id":
                row["conversation_id"],

            "tweet_id":
                row["tweet_id"],

            "target_message":
                message,

            "true_intent":
                true_intent,

            "predicted_intent":
                result["intent"],

            "intent_confidence":
                result["confidence"],

            "top_case_similarity":
                top_case_similarity,

            "decision":
                result["decision"],

            "decision_reason":
                result["decision_reason"],

            "draft_reply":
                result["draft_reply"],

            # NEW:
            # Save historical evidence for LLM-as-judge
            "historical_cases":
              json.dumps(
    result["historical_cases"],
    ensure_ascii=False,
    default=str
)
        })

    predictions = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    predictions.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("AGENT EVALUATION")
    print("=" * 70)

    accuracy = (
        predictions["true_intent"]
        == predictions["predicted_intent"]
    ).mean()

    print(
        f"\nIntent Accuracy: {accuracy:.4f}"
    )

    print(
        f"Intent Accuracy (%): {accuracy * 100:.2f}%"
    )

    print("\nDecision Distribution:")

    print(
        predictions["decision"]
        .value_counts()
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()