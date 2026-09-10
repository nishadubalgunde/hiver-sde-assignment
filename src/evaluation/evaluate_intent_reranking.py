import os
import sys

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"
SILVER_PATH = "data/processed/apple_silver_labeled.csv"
CONVERSATIONS_PATH = "data/processed/apple_conversations.csv"
OUTPUT_PATH = "results/intent_reranking_predictions.csv"


class IntentClassifier:
    def __init__(self):
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
        df = pd.read_csv(SILVER_PATH)

        texts = df["text"].fillna("").astype(str)
        labels = df["intent"].fillna("").astype(str)

        X = self.vectorizer.fit_transform(texts)

        self.model.fit(X, labels)

    def predict(self, text):
        X = self.vectorizer.transform([str(text)])

        probabilities = self.model.predict_proba(X)[0]
        classes = self.model.classes_

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked


def build_historical_cases():
    print("Loading historical conversations...")

    conversations = pd.read_csv(CONVERSATIONS_PATH)

    golden = pd.read_csv(GOLDEN_PATH)

    golden_conversations = set(
        golden["conversation_id"].astype(str)
    )

    conversations["conversation_id"] = (
        conversations["conversation_id"].astype(str)
    )

    # Prevent Golden Set leakage
    conversations = conversations[
        ~conversations["conversation_id"].isin(golden_conversations)
    ].copy()

    conversations["created_at"] = pd.to_datetime(
        conversations["created_at"],
        errors="coerce"
    )

    conversations = conversations.sort_values(
        ["conversation_id", "created_at"]
    )

    cases = []

    for conversation_id, group in conversations.groupby(
        "conversation_id"
    ):
        messages = group.to_dict("records")

        for i, message in enumerate(messages):

            # Customer message
            if message["inbound"] is not True:
                continue

            response = None

            # Find the next AppleSupport response
            for j in range(i + 1, len(messages)):

                candidate = messages[j]

                if candidate["inbound"] is False:
                    response = candidate
                    break

            if response is None:
                continue

            cases.append(
                {
                    "conversation_id": conversation_id,
                    "customer_message": str(message["text"]),
                    "support_response": str(response["text"]),
                    "customer_tweet_id": message["tweet_id"],
                    "support_tweet_id": response["tweet_id"]
                }
            )

    cases_df = pd.DataFrame(cases)

    print(
        f"Historical customer-response pairs: {len(cases_df)}"
    )

    return cases_df


def assign_silver_intents(cases_df):
    """
    Assign silver intent labels to historical customer messages.
    This is used only as a proxy for reranking evaluation.
    """

    silver = pd.read_csv(SILVER_PATH)

    silver_lookup = dict(
        zip(
            silver["tweet_id"].astype(str),
            silver["intent"].astype(str)
        )
    )

    cases_df["silver_intent"] = (
        cases_df["customer_tweet_id"]
        .astype(str)
        .map(silver_lookup)
        .fillna("other_unclear")
    )

    return cases_df


def main():

    print("=" * 80)
    print("INTENT-AWARE RETRIEVAL RERANKING EVALUATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load Golden Set
    # ---------------------------------------------------------

    golden = pd.read_csv(GOLDEN_PATH)

    golden["intent"] = (
        golden["intent"]
        .fillna("")
        .astype(str)
    )

    golden["target_message"] = (
        golden["target_message"]
        .fillna("")
        .astype(str)
    )

    print(f"Golden examples: {len(golden)}")

    # ---------------------------------------------------------
    # 2. Build historical retrieval cases
    # ---------------------------------------------------------

    cases_df = build_historical_cases()

    cases_df = assign_silver_intents(cases_df)

    # ---------------------------------------------------------
    # 3. Build TF-IDF retrieval index
    # ---------------------------------------------------------

    retrieval_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    retrieval_matrix = retrieval_vectorizer.fit_transform(
        cases_df["customer_message"]
    )

    # ---------------------------------------------------------
    # 4. Train intent classifier
    # ---------------------------------------------------------

    classifier = IntentClassifier()

    # ---------------------------------------------------------
    # 5. Evaluate every Golden example
    # ---------------------------------------------------------

    results = []

    for index, row in golden.iterrows():

        message = row["target_message"]

        true_intent = row["intent"]

        # ---------------------------------------------
        # Predict intent
        # ---------------------------------------------

        ranked_intents = classifier.predict(message)

        predicted_intent = ranked_intents[0][0]

        predicted_confidence = float(
            ranked_intents[0][1]
        )

        # ---------------------------------------------
        # Retrieve top 10 lexical matches
        # ---------------------------------------------

        query_vector = retrieval_vectorizer.transform(
            [message]
        )

        scores = cosine_similarity(
            query_vector,
            retrieval_matrix
        )[0]

        top_indices = np.argsort(scores)[::-1][:10]

        candidates = []

        for candidate_index in top_indices:

            candidate = cases_df.iloc[candidate_index]

            candidates.append(
                {
                    "index": candidate_index,
                    "similarity": float(
                        scores[candidate_index]
                    ),
                    "intent": candidate["silver_intent"]
                }
            )

        # ---------------------------------------------
        # Original ranking = similarity only
        # ---------------------------------------------

        original_top5 = candidates[:5]

        # ---------------------------------------------
        # Intent-aware reranking
        #
        # Same intent gets a bonus.
        # We keep similarity important.
        # ---------------------------------------------

        for candidate in candidates:

            same_intent = (
                candidate["intent"] == predicted_intent
            )

            candidate["rerank_score"] = (
                candidate["similarity"]
                + (0.20 if same_intent else 0.0)
            )

        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        reranked_top5 = reranked[:5]

        original_intents = [
            x["intent"]
            for x in original_top5
        ]

        reranked_intents = [
            x["intent"]
            for x in reranked_top5
        ]

        results.append(
            {
                "conversation_id": row["conversation_id"],
                "tweet_id": row["tweet_id"],
                "target_message": message,
                "true_intent": true_intent,

                "predicted_intent": predicted_intent,
                "predicted_confidence": round(
                    predicted_confidence,
                    4
                ),

                "original_top1_intent":
                    original_intents[0]
                    if original_intents
                    else "",

                "reranked_top1_intent":
                    reranked_intents[0]
                    if reranked_intents
                    else "",

                "original_top5_intents":
                    " | ".join(original_intents),

                "reranked_top5_intents":
                    " | ".join(reranked_intents),

                "original_top1_similarity":
                    round(
                        original_top5[0]["similarity"],
                        4
                    )
                    if original_top5
                    else 0.0,

                "reranked_top1_score":
                    round(
                        reranked_top5[0]["rerank_score"],
                        4
                    )
                    if reranked_top5
                    else 0.0
            }
        )

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(golden)} examples"
            )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # 6. Evaluation functions
    # ---------------------------------------------------------

    def consistency_at_k(
        result_df,
        column,
        k
    ):

        correct = 0

        for _, row in result_df.iterrows():

            intents = str(
                row[column]
            ).split(" | ")

            intents = intents[:k]

            if row["true_intent"] in intents:
                correct += 1

        return correct / len(result_df)

    # ---------------------------------------------------------
    # 7. Overall results
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("ORIGINAL RETRIEVAL")
    print("=" * 80)

    print(
        f"@1: {consistency_at_k(results_df, 'original_top5_intents', 1) * 100:.2f}%"
    )

    print(
        f"@3: {consistency_at_k(results_df, 'original_top5_intents', 3) * 100:.2f}%"
    )

    print(
        f"@5: {consistency_at_k(results_df, 'original_top5_intents', 5) * 100:.2f}%"
    )

    print()
    print("=" * 80)
    print("INTENT-AWARE RERANKING")
    print("=" * 80)

    print(
        f"@1: {consistency_at_k(results_df, 'reranked_top5_intents', 1) * 100:.2f}%"
    )

    print(
        f"@3: {consistency_at_k(results_df, 'reranked_top5_intents', 3) * 100:.2f}%"
    )

    print(
        f"@5: {consistency_at_k(results_df, 'reranked_top5_intents', 5) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # 8. Actionable subset
    # ---------------------------------------------------------

    actionable = results_df[
        results_df["true_intent"] != "other_unclear"
    ].copy()

    print()
    print("=" * 80)
    print("ACTIONABLE EXAMPLES")
    print("=" * 80)

    print(
        f"Actionable examples: {len(actionable)}"
    )

    print(
        f"Original @1: "
        f"{consistency_at_k(actionable, 'original_top5_intents', 1) * 100:.2f}%"
    )

    print(
        f"Reranked @1: "
        f"{consistency_at_k(actionable, 'reranked_top5_intents', 1) * 100:.2f}%"
    )

    print(
        f"Original @5: "
        f"{consistency_at_k(actionable, 'original_top5_intents', 5) * 100:.2f}%"
    )

    print(
        f"Reranked @5: "
        f"{consistency_at_k(actionable, 'reranked_top5_intents', 5) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # 9. Save
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Saved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()