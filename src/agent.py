from intent.classifier import IntentClassifier
from retrieval.retriever import HistoricalRetriever
from generation.reply_generator import ReplyGenerator
from decision.decision_engine import DecisionEngine


class AppleSupportAgent:

    def __init__(self):

        print("Loading Intent Classifier...")
        self.classifier = IntentClassifier()

        print("Loading Historical Retriever...")
        self.retriever = HistoricalRetriever()

        print("Loading Reply Generator...")
        self.generator = ReplyGenerator()

        print("Loading Decision Engine...")
        self.decision_engine = DecisionEngine()

        print("Agent ready.")

    def handle_message(self, customer_message):

        # Step 1: Intent classification
        classification = self.classifier.predict(
            customer_message
        )

        intent = classification["intent"]
        confidence = classification["confidence"]

        # Step 2: Retrieve historical AppleSupport cases
        historical_cases = self.retriever.retrieve(
            customer_message,
            top_k=3
        )

        # Step 3: Generate grounded reply
        reply = self.generator.generate_reply(
            customer_message=customer_message,
            intent=intent,
            historical_cases=historical_cases
        )

        # Step 4: Decide AUTO_HANDLE or ESCALATE
        # The decision engine now also checks the generated reply.
        decision = self.decision_engine.decide(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=confidence,
            historical_cases=historical_cases,
            draft_reply=reply
        )

        return {
            "customer_message": customer_message,
            "intent": intent,
            "confidence": confidence,
            "top_predictions": classification["top_predictions"],
            "historical_cases": historical_cases,
            "draft_reply": reply,
            "decision": decision["decision"],
            "decision_reason": decision["reason"]
        }


if __name__ == "__main__":

    agent = AppleSupportAgent()

    test_messages = [
        "My Apple ID has been disabled",
        "My iPhone battery is draining very quickly",
        "I cannot connect my iPhone to Wi-Fi",
        "Done",
        "Thanks"
    ]

    for message in test_messages:

        print("\n" + "=" * 80)

        print("CUSTOMER MESSAGE:")
        print(message)

        result = agent.handle_message(message)

        print("\nINTENT:")
        print(result["intent"])

        print("\nCONFIDENCE:")
        print(result["confidence"])

        print("\nTOP PREDICTIONS:")

        for prediction in result["top_predictions"]:
            print(prediction)

        print("\nTOP HISTORICAL CASE:")

        if result["historical_cases"]:

            case = result["historical_cases"][0]

            print(
                "Similarity:",
                case["similarity"]
            )

            print(
                "Retrieved Intent:",
                case.get("retrieved_intent")
            )

            print(
                "Customer:",
                case["customer_message"]
            )

            print(
                "AppleSupport:",
                case["support_response"]
            )

        print("\nDRAFT REPLY:")
        print(result["draft_reply"])

        print("\nDECISION:")
        print(result["decision"])

        print("\nDECISION REASON:")
        print(result["decision_reason"])