import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
from retrieval.retriever import HistoricalRetriever


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"
SILVER_PATH = "data/processed/apple_silver_labeled.csv"
OUTPUT_PATH = "results/retrieval_intent_predictions.csv"


def evaluate_retrieval_intent(top_k=5):

    print("=" * 70)
    print("RETRIEVAL INTENT CONSISTENCY EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load Golden Set
    # ---------------------------------------------------------

    golden = pd.read_csv(GOLDEN_PATH)

    golden["tweet_id"] = golden["tweet_id"].astype(str)
    golden["intent"] = golden["intent"].astype(str)

    # ---------------------------------------------------------
    # Load Silver Labels
    # ---------------------------------------------------------

    silver = pd.read_csv(SILVER_PATH)

    silver["tweet_id"] = silver["tweet_id"].astype(str)
    silver["intent"] = silver["intent"].astype(str)

    silver_intent_map = dict(
        zip(
            silver["tweet_id"],
            silver["intent"]
        )
    )

    # ---------------------------------------------------------
    # Build Retriever
    # ---------------------------------------------------------

    retriever = HistoricalRetriever()

    results = []

    # ---------------------------------------------------------
    # Evaluate every Golden example
    # ---------------------------------------------------------

    for index, row in golden.iterrows():

        query = str(row["target_message"])
        true_intent = str(row["intent"])

        retrieved = retriever.retrieve(
            query,
            top_k=top_k
        )

        retrieved_intents = []

        for case in retrieved:

            tweet_id = str(
                case["customer_tweet_id"]
            )

            intent = silver_intent_map.get(
                tweet_id,
                "unknown"
            )

            retrieved_intents.append(intent)

        # Make sure we always have 5 slots
        while len(retrieved_intents) < top_k:
            retrieved_intents.append("unknown")

        # -----------------------------------------------------
        # Intent consistency
        # -----------------------------------------------------

        match_at_1 = (
            retrieved_intents[0] == true_intent
        )

        match_at_3 = (
            true_intent
            in retrieved_intents[:3]
        )

        match_at_5 = (
            true_intent
            in retrieved_intents[:5]
        )

        results.append({

            "golden_index": index,

            "conversation_id":
                str(row["conversation_id"]),

            "tweet_id":
                str(row["tweet_id"]),

            "target_message":
                query,

            "true_intent":
                true_intent,

            "retrieved_intent_1":
                retrieved_intents[0],

            "retrieved_intent_2":
                retrieved_intents[1],

            "retrieved_intent_3":
                retrieved_intents[2],

            "retrieved_intent_4":
                retrieved_intents[3],

            "retrieved_intent_5":
                retrieved_intents[4],

            "intent_match_at_1":
                int(match_at_1),

            "intent_match_at_3":
                int(match_at_3),

            "intent_match_at_5":
                int(match_at_5),

            "top1_similarity":
                retrieved[0]["similarity"]
                if retrieved else 0.0
        })

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(golden)} examples"
            )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # ALL 200 EXAMPLES
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL 200 GOLDEN EXAMPLES")
    print("=" * 70)

    total = len(results_df)

    consistency_1 = (
        results_df["intent_match_at_1"].mean()
    )

    consistency_3 = (
        results_df["intent_match_at_3"].mean()
    )

    consistency_5 = (
        results_df["intent_match_at_5"].mean()
    )

    print(
        f"Intent consistency@1: "
        f"{consistency_1:.2%}"
    )

    print(
        f"Intent consistency@3: "
        f"{consistency_3:.2%}"
    )

    print(
        f"Intent consistency@5: "
        f"{consistency_5:.2%}"
    )

    # ---------------------------------------------------------
    # ACTIONABLE EXAMPLES
    # Exclude other_unclear
    # ---------------------------------------------------------

    actionable = results_df[
        results_df["true_intent"] != "other_unclear"
    ].copy()

    print("\n" + "=" * 70)
    print("ACTIONABLE EXAMPLES")
    print("=" * 70)

    print(
        f"Actionable examples: "
        f"{len(actionable)}"
    )

    if len(actionable) > 0:

        actionable_1 = (
            actionable["intent_match_at_1"].mean()
        )

        actionable_3 = (
            actionable["intent_match_at_3"].mean()
        )

        actionable_5 = (
            actionable["intent_match_at_5"].mean()
        )

        print(
            f"Intent consistency@1: "
            f"{actionable_1:.2%}"
        )

        print(
            f"Intent consistency@3: "
            f"{actionable_3:.2%}"
        )

        print(
            f"Intent consistency@5: "
            f"{actionable_5:.2%}"
        )

    # ---------------------------------------------------------
    # Similarity
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIMILARITY")
    print("=" * 70)

    print(
        f"Mean top-1 similarity: "
        f"{results_df['top1_similarity'].mean():.4f}"
    )

    print(
        f"Median top-1 similarity: "
        f"{results_df['top1_similarity'].median():.4f}"
    )

    print("\n" + "=" * 70)

    print(
        f"Saved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate_retrieval_intent(top_k=5)