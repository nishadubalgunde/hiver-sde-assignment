import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from retrieval.retriever import HistoricalRetriever


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"
OUTPUT_PATH = "results/retrieval_predictions.csv"


def evaluate_retrieval(top_k=5):
    golden = pd.read_csv(GOLDEN_PATH)

    retriever = HistoricalRetriever()

    results = []

    for index, row in golden.iterrows():

        query = str(row["target_message"])
        true_intent = str(row["intent"])
        conversation_id = str(row["conversation_id"])

        retrieved = retriever.retrieve(query, top_k=top_k)

        if retrieved:
            best = retrieved[0]

            results.append({
                "golden_index": index,
                "conversation_id": conversation_id,
                "target_message": query,
                "true_intent": true_intent,
                "retrieved_customer_message": best["customer_message"],
                "retrieved_support_response": best["support_response"],
                "similarity": best["similarity"]
            })

        else:
            results.append({
                "golden_index": index,
                "conversation_id": conversation_id,
                "target_message": query,
                "true_intent": true_intent,
                "retrieved_customer_message": "",
                "retrieved_support_response": "",
                "similarity": 0.0
            })

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/200 examples")

    results_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"\nGolden examples: {len(results_df)}")

    print(
        f"Mean similarity: "
        f"{results_df['similarity'].mean():.4f}"
    )

    print(
        f"Median similarity: "
        f"{results_df['similarity'].median():.4f}"
    )

    print(
        f"Similarity >= 0.20: "
        f"{(results_df['similarity'] >= 0.20).mean():.2%}"
    )

    print(
        f"Similarity >= 0.30: "
        f"{(results_df['similarity'] >= 0.30).mean():.2%}"
    )

    print(
        f"Similarity >= 0.40: "
        f"{(results_df['similarity'] >= 0.40).mean():.2%}"
    )

    print(
        f"Similarity >= 0.50: "
        f"{(results_df['similarity'] >= 0.50).mean():.2%}"
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_retrieval(top_k=5)