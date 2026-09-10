class DecisionEngine:

    def __init__(
        self,
        min_intent_confidence=0.75,
        min_retrieval_similarity=0.55
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def decide(
        self,
        customer_message,
        intent,
        intent_confidence,
        historical_cases,
        draft_reply=None
    ):

        reasons = []

        # ---------------------------------------------------------
        # 1. INTENT CONFIDENCE CHECK
        # ---------------------------------------------------------

        if intent_confidence < self.min_intent_confidence:
            reasons.append(
                "Intent confidence is below the safety threshold."
            )

        # ---------------------------------------------------------
        # 2. SHORT / CONTEXT-DEPENDENT MESSAGE CHECK
        # ---------------------------------------------------------

        normalized_message = customer_message.strip().lower()
        words = normalized_message.split()

        weak_phrases = {
            "done",
            "thanks",
            "thank you",
            "okay",
            "ok",
            "yes",
            "no",
            "still happening",
            "still not working",
            "dm sent",
            "i did"
        }

        if len(words) <= 3 or normalized_message in weak_phrases:
            reasons.append(
                "Customer message is short or context-dependent."
            )

        # ---------------------------------------------------------
        # 3. FIND BEST INTENT-COMPATIBLE HISTORICAL CASE
        # ---------------------------------------------------------

        compatible_cases = [
            case
            for case in historical_cases
            if case.get("retrieved_intent") == intent
        ]

        if not historical_cases:

            reasons.append(
                "No historical AppleSupport case was retrieved."
            )

        elif not compatible_cases:

            reasons.append(
                "No retrieved historical case matches the predicted intent."
            )

        else:

            compatible_cases.sort(
                key=lambda case: case.get("similarity", 0),
                reverse=True
            )

            best_compatible_case = compatible_cases[0]

            best_similarity = best_compatible_case.get(
                "similarity",
                0
            )

            if best_similarity < self.min_retrieval_similarity:
                reasons.append(
                    "Best intent-compatible historical case "
                    "has weak similarity."
                )

        # ---------------------------------------------------------
        # 4. REPLY SAFETY CHECK
        # ---------------------------------------------------------

        if not draft_reply:

            reasons.append(
                "No usable customer-facing reply was generated."
            )

        else:

            reply = str(draft_reply).strip()

            generic_reply = (
                "Please contact Apple Support for further assistance "
                "with this issue."
            )

            if reply == generic_reply:

                reasons.append(
                    "No sufficiently strong historical evidence was "
                    "available for a grounded reply."
                )

            elif len(reply) < 25:

                reasons.append(
                    "Generated reply is too short to provide reliable "
                    "customer assistance."
                )

            # Detect malformed cleaned historical responses
            if reply.endswith(":"):

                reasons.append(
                    "Generated reply appears incomplete after cleaning."
                )

            # Detect dangling phrases caused by removed URLs
            if reply.lower().endswith(
                ("here:", "here", "link:", "below:")
            ):

                reasons.append(
                    "Generated reply appears incomplete after cleaning."
                )

        # ---------------------------------------------------------
        # 5. FINAL DECISION
        # ---------------------------------------------------------

        if reasons:

            decision = "ESCALATE"

            reason = " ".join(reasons)

        else:

            decision = "AUTO_HANDLE"

            reason = (
                "High intent confidence, strong intent-compatible "
                "historical evidence, and a usable grounded reply "
                "were found."
            )

        return {
            "decision": decision,
            "reason": reason
        }


if __name__ == "__main__":

    engine = DecisionEngine()

    result = engine.decide(
        customer_message="My Apple ID has been disabled",
        intent="apple_id_account",
        intent_confidence=0.9936,
        historical_cases=[
            {
                "similarity": 0.40,
                "retrieved_intent": "other_unclear"
            },
            {
                "similarity": 0.58,
                "retrieved_intent": "apple_id_account"
            }
        ],
        draft_reply=(
            "Apple Support can help you with this issue."
        )
    )

    print("\nDecision:")
    print(result)