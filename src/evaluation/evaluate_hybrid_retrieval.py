import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GOLDEN_PATH = "data/golden/apple_golden_set_200.csv"
CONVERSATIONS_PATH = "data/processed/apple_conversations.csv"
SILVER_PATH = "data/processed/apple_silver_labeled.csv"

OUTPUT_PATH = "results/hybrid_retrieval_predictions.csv"


def is_weak_message(text):
    """
    Detect messages that are too short or context-dependent
    to retrieve reliably using lexical similarity alone.
    """

    text = str(text).strip().lower()

    # Remove AppleSupport mention
    text = text.replace("@applesupport", "").strip()

    words = text.split()

    if len(words) <= 3:
        return True

    weak_phrases = [
        "done",
        "thanks",
        "thank you",
        "okay",
        "ok",
        "yes",
        "no",
        "i did",
        "it worked",
        "i think it worked",
        "dm sent",
        "dmed you",
        "still waiting",
        "waiting for response",
        "no it does not",
        "what do you mean",
        "what do you mean by that"
    ]

    if text in weak_phrases:
        return True

    return False


def build_historical_cases():

    print("Loading historical conversations...")

    conversations = pd.read_csv(
        CONVERSATIONS_PATH
    )

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    silver = pd.read_csv(
        SILVER_PATH
    )

    conversations["conversation_id"] = (
        conversations["conversation_id"].astype(str)
    )

    golden_conversations = set(
        golden["conversation_id"].astype(str)
    )

    # Never retrieve from Golden conversations
    conversations = conversations[
        ~conversations["conversation_id"].isin(
            golden_conversations
        )
    ].copy()

    conversations["created_at"] = pd.to_datetime(
        conversations["created_at"],
        errors="coerce"
    )

    conversations = conversations.sort_values(
        ["conversation_id", "created_at"]
    )

    silver["tweet_id"] = (
        silver["tweet_id"].astype(str)
    )

    silver["intent"] = (
        silver["intent"].astype(str)
    )

    silver_intent_map = dict(
        zip(
            silver["tweet_id"],
            silver["intent"]
        )
    )

    cases = []

    for conversation_id, group in conversations.groupby(
        "conversation_id"
    ):

        messages = group.to_dict(
            "records"
        )

        for i, message in enumerate(messages):

            if message["inbound"] is not True:
                continue

            response = None

            for j in range(i + 1, len(messages)):

                candidate = messages[j]

                if candidate["inbound"] is False:
                    response = candidate
                    break

            if response is None:
                continue

            customer_message = str(
                message["text"]
            )

            # Previous conversation messages
            context_messages = []

            start = max(
                0,
                i - 3
            )

            for k in range(start, i):

                previous = messages[k]

                speaker = (
                    "Customer"
                    if previous["inbound"] is True
                    else "AppleSupport"
                )

                context_messages.append(
                    f"{speaker}: {previous['text']}"
                )

            context = " ".join(
                context_messages
            )

            cases.append({

                "conversation_id":
                    str(conversation_id),

                "customer_message":
                    customer_message,

                "context":
                    context,

                "customer_tweet_id":
                    str(message["tweet_id"]),

                "support_response":
                    str(response["text"]),

                "support_tweet_id":
                    str(response["tweet_id"]),

                "silver_intent":
                    silver_intent_map.get(
                        str(message["tweet_id"]),
                        "unknown"
                    )
            })

    cases_df = pd.DataFrame(
        cases
    )

    # Create two representations
    cases_df["message_text"] = (
        cases_df["customer_message"]
        .str.replace(
            "@AppleSupport",
            "",
            regex=False
        )
        .fillna("")
    )

    cases_df["context_text"] = (
        cases_df["context"]
        .fillna("")
    )

    print(
        f"Historical customer-response pairs: "
        f"{len(cases_df)}"
    )

    return cases_df


def evaluate():

    print("=" * 80)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 80)

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    golden["intent"] = (
        golden["intent"].astype(str)
    )

    historical = build_historical_cases()

    # ---------------------------------------------------------
    # Build separate TF-IDF indexes
    # ---------------------------------------------------------

    message_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    context_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    message_matrix = (
        message_vectorizer.fit_transform(
            historical["message_text"]
        )
    )

    context_matrix = (
        context_vectorizer.fit_transform(
            historical["context_text"]
        )
    )

    results = []

    weak_count = 0
    normal_count = 0

    for index, row in golden.iterrows():

        true_intent = str(
            row["intent"]
        )

        target_message = str(
            row["target_message"]
        )

        context = str(
            row.get(
                "conversation_context",
                ""
            )
        )

        weak = is_weak_message(
            target_message
        )

        if weak:
            weak_count += 1

            query = (
                context
                + " Customer: "
                + target_message
            )

            query_vector = (
                context_vectorizer.transform(
                    [query]
                )
            )

            scores = cosine_similarity(
                query_vector,
                context_matrix
            )[0]

        else:
            normal_count += 1

            clean_message = (
                target_message
                .replace(
                    "@AppleSupport",
                    ""
                )
            )

            query_vector = (
                message_vectorizer.transform(
                    [clean_message]
                )
            )

            scores = cosine_similarity(
                query_vector,
                message_matrix
            )[0]

        top_indices = np.argsort(
            scores
        )[::-1][:5]

        retrieved_intents = []

        for retrieved_index in top_indices:

            case = historical.iloc[
                retrieved_index
            ]

            retrieved_intents.append(
                case["silver_intent"]
            )

        while len(retrieved_intents) < 5:

            retrieved_intents.append(
                "unknown"
            )

        match_at_1 = (
            retrieved_intents[0]
            == true_intent
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

            "golden_index":
                index,

            "conversation_id":
                str(row["conversation_id"]),

            "tweet_id":
                str(row["tweet_id"]),

            "target_message":
                target_message,

            "true_intent":
                true_intent,

            "weak_message":
                int(weak),

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
                round(
                    float(
                        scores[
                            top_indices[0]
                        ]
                    ),
                    4
                )
        })

        if (index + 1) % 25 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(golden)} examples"
            )

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # ALL EXAMPLES
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL 200 GOLDEN EXAMPLES")
    print("=" * 80)

    print(
        f"Intent consistency@1: "
        f"{results_df['intent_match_at_1'].mean():.2%}"
    )

    print(
        f"Intent consistency@3: "
        f"{results_df['intent_match_at_3'].mean():.2%}"
    )

    print(
        f"Intent consistency@5: "
        f"{results_df['intent_match_at_5'].mean():.2%}"
    )

    # ---------------------------------------------------------
    # ACTIONABLE
    # ---------------------------------------------------------

    actionable = results_df[
        results_df["true_intent"]
        != "other_unclear"
    ]

    print("\n" + "=" * 80)
    print("ACTIONABLE EXAMPLES")
    print("=" * 80)

    print(
        f"Actionable examples: "
        f"{len(actionable)}"
    )

    print(
        f"Intent consistency@1: "
        f"{actionable['intent_match_at_1'].mean():.2%}"
    )

    print(
        f"Intent consistency@3: "
        f"{actionable['intent_match_at_3'].mean():.2%}"
    )

    print(
        f"Intent consistency@5: "
        f"{actionable['intent_match_at_5'].mean():.2%}"
    )

    # ---------------------------------------------------------
    # WEAK VS NORMAL
    # ---------------------------------------------------------

    weak = results_df[
        results_df["weak_message"] == 1
    ]

    normal = results_df[
        results_df["weak_message"] == 0
    ]

    print("\n" + "=" * 80)
    print("MESSAGE TYPE BREAKDOWN")
    print("=" * 80)

    print(
        f"Weak/context-dependent messages: "
        f"{len(weak)}"
    )

    print(
        f"Normal/informative messages: "
        f"{len(normal)}"
    )

    if len(weak) > 0:

        print("\nWeak messages:")

        print(
            f"@1: "
            f"{weak['intent_match_at_1'].mean():.2%}"
        )

        print(
            f"@3: "
            f"{weak['intent_match_at_3'].mean():.2%}"
        )

        print(
            f"@5: "
            f"{weak['intent_match_at_5'].mean():.2%}"
        )

    if len(normal) > 0:

        print("\nNormal messages:")

        print(
            f"@1: "
            f"{normal['intent_match_at_1'].mean():.2%}"
        )

        print(
            f"@3: "
            f"{normal['intent_match_at_3'].mean():.2%}"
        )

        print(
            f"@5: "
            f"{normal['intent_match_at_5'].mean():.2%}"
        )

    # ---------------------------------------------------------
    # SIMILARITY
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SIMILARITY")
    print("=" * 80)

    print(
        f"Mean top-1 similarity: "
        f"{results_df['top1_similarity'].mean():.4f}"
    )

    print(
        f"Median top-1 similarity: "
        f"{results_df['top1_similarity'].median():.4f}"
    )

    print("\n" + "=" * 80)

    print(
        f"Saved results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate()