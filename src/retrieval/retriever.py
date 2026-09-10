import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalRetriever:

    def __init__(
        self,
        conversations_path="data/processed/apple_conversations.csv",
        golden_path="data/golden/apple_golden_set_200.csv",
        silver_path="data/processed/apple_silver_labeled.csv"
    ):
        self.conversations_path = conversations_path
        self.golden_path = golden_path
        self.silver_path = silver_path

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )

        self._load_data()
        self._build_index()

    def _load_data(self):

        # Load conversation data
        df = pd.read_csv(self.conversations_path)

        # Load Golden Set
        golden = pd.read_csv(self.golden_path)

        # Load Silver Labels
        silver = pd.read_csv(self.silver_path)

        # Prepare Golden conversation IDs
        golden_conversations = set(
            golden["conversation_id"].astype(str)
        )

        # Prepare IDs
        df["conversation_id"] = (
            df["conversation_id"].astype(str)
        )

        silver["tweet_id"] = (
            silver["tweet_id"].astype(str)
        )

        # Create tweet_id -> intent lookup
        silver_intents = dict(
            zip(
                silver["tweet_id"],
                silver["intent"].astype(str)
            )
        )

        # Exclude Golden Set conversations
        df = df[
            ~df["conversation_id"].isin(
                golden_conversations
            )
        ].copy()

        # Sort messages chronologically
        df["created_at"] = pd.to_datetime(
            df["created_at"],
            errors="coerce",
            format="mixed"
        )

        df = df.sort_values(
            ["conversation_id", "created_at"]
        )

        self.df = df.reset_index(drop=True)

        # Build customer -> AppleSupport response pairs
        self.cases = []

        for conversation_id, group in self.df.groupby(
            "conversation_id"
        ):

            messages = group.to_dict("records")

            for i, message in enumerate(messages):

                # Only customer messages
                if message["inbound"] is not True:
                    continue

                # Find the next AppleSupport response
                response = None

                for j in range(i + 1, len(messages)):

                    candidate = messages[j]

                    if candidate["inbound"] is False:
                        response = candidate
                        break

                # Skip if no response exists
                if response is None:
                    continue

                # Get silver intent
                retrieved_intent = silver_intents.get(
                    str(message["tweet_id"]),
                    "unknown"
                )

                self.cases.append(
                    {
                        "conversation_id": conversation_id,

                        "customer_message": message["text"],

                        "customer_tweet_id": message["tweet_id"],

                        "customer_created_at": message["created_at"],

                        "support_response": response["text"],

                        "support_tweet_id": response["tweet_id"],

                        "support_created_at": response["created_at"],

                        "retrieved_intent": retrieved_intent
                    }
                )

        self.cases_df = pd.DataFrame(self.cases)

        print(
            f"Historical conversations available: "
            f"{len(self.df)}"
        )

        print(
            f"Golden conversations excluded: "
            f"{len(golden_conversations)}"
        )

        print(
            f"Historical customer-response pairs: "
            f"{len(self.cases_df)}"
        )

    def _build_index(self):

        texts = (
            self.cases_df["customer_message"]
            .fillna("")
            .astype(str)
        )

        self.matrix = self.vectorizer.fit_transform(
            texts
        )

    def retrieve(self, query, top_k=3):

        query_vector = self.vectorizer.transform(
            [str(query)]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            row = self.cases_df.iloc[index]

            results.append(
                {
                    "conversation_id":
                        row["conversation_id"],

                    "customer_message":
                        row["customer_message"],

                    "support_response":
                        row["support_response"],

                    "customer_tweet_id":
                        row["customer_tweet_id"],

                    "support_tweet_id":
                        row["support_tweet_id"],

                    "similarity":
                        round(
                            float(scores[index]),
                            4
                        ),

                    "retrieved_intent":
                        row["retrieved_intent"]
                }
            )

        return results


if __name__ == "__main__":

    retriever = HistoricalRetriever()

    test_queries = [

        "My iPhone battery is draining very quickly",

        "I cannot connect my iPhone to Wi-Fi",

        "My Apple ID is locked"
    ]

    for query in test_queries:

        print("\n" + "=" * 70)

        print("QUERY:")
        print(query)

        results = retriever.retrieve(
            query,
            top_k=3
        )

        print("\nTOP HISTORICAL CASES:")

        for result in results:

            print(
                f"\nSimilarity: "
                f"{result['similarity']}"
            )

            print(
                f"Retrieved Intent: "
                f"{result['retrieved_intent']}"
            )

            print(
                f"Customer: "
                f"{result['customer_message']}"
            )

            print(
                f"AppleSupport: "
                f"{result['support_response']}"
            )