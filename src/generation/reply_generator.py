import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


class ReplyGenerator:

    def __init__(self, model="gpt-5.6-luna"):
        self.model = model
        self.client = OpenAI(api_key=API_KEY) if API_KEY else None

    def _clean_response(self, response):
        """
        Clean a historical AppleSupport response before using it
        as a fallback customer-facing reply.
        """

        response = str(response).strip()

        if not response:
            return ""

        # Remove Twitter-style user mentions such as @739664
        response = re.sub(r"@\w+", "", response)

        # Remove t.co links
        response = re.sub(r"https?://t\.co/\S+", "", response)

        # Remove excessive whitespace
        response = re.sub(r"\s+", " ", response).strip()

        return response

    def _select_safe_historical_case(
        self,
        historical_cases,
        intent
    ):
        """
        Select the strongest historical case that matches
        the predicted intent.
        """

        compatible_cases = [
            case
            for case in historical_cases
            if case.get("retrieved_intent") == intent
        ]

        if not compatible_cases:
            return None

        compatible_cases.sort(
            key=lambda case: case.get("similarity", 0),
            reverse=True
        )

        best_case = compatible_cases[0]

        # Require reasonably strong evidence
        if best_case.get("similarity", 0) < 0.40:
            return None

        return best_case

    def generate_reply(
        self,
        customer_message,
        intent,
        historical_cases
    ):

        # --------------------------------------------------
        # LLM GENERATION
        # --------------------------------------------------

        if self.client:

            evidence = "\n\n".join(
                [
                    f"Historical customer message: "
                    f"{case['customer_message']}\n"
                    f"Historical AppleSupport response: "
                    f"{case['support_response']}"
                    for case in historical_cases
                ]
            )

            prompt = f"""
You are an AI customer-support reply assistant for AppleSupport.

Customer message:
{customer_message}

Detected intent:
{intent}

Historical AppleSupport cases:
{evidence}

STRICT RULES:

1. Ground the reply only in the historical AppleSupport responses.
2. Do not invent policies, refunds, guarantees, troubleshooting steps,
   links, or actions that are not supported by the evidence.
3. Do not claim Apple has performed an action.
4. Do not mention the detected intent.
5. Do not mention that you are an AI.
6. Keep the response concise and professional.
7. Do not copy Twitter usernames.
8. If the historical evidence is weak or unrelated, politely ask the
   customer to contact Apple Support for further assistance.
9. Return only the customer-facing reply.

Customer-facing reply:
"""

            try:

                response = self.client.responses.create(
                    model=self.model,
                    input=prompt
                )

                reply = response.output_text.strip()

                if reply:
                    return self._clean_response(reply)

            except Exception as e:

                print(
                    f"LLM unavailable: {type(e).__name__}"
                )

                print(
                    "Using safe historical fallback."
                )

        # --------------------------------------------------
        # SAFE HISTORICAL FALLBACK
        # --------------------------------------------------

        best_case = self._select_safe_historical_case(
            historical_cases,
            intent
        )

        if best_case:

            historical_response = self._clean_response(
                best_case.get("support_response", "")
            )

            if historical_response:
                return historical_response

        # --------------------------------------------------
        # FINAL SAFE FALLBACK
        # --------------------------------------------------

        return (
            "Please contact Apple Support for further assistance "
            "with this issue."
        )


if __name__ == "__main__":

    generator = ReplyGenerator()

    test_cases = [
        {
            "similarity": 0.60,
            "retrieved_intent": "battery_power",
            "customer_message":
                "My battery is draining quickly",
            "support_response":
                "@123456 Let's take a closer look. "
                "https://t.co/example"
        },
        {
            "similarity": 0.70,
            "retrieved_intent": "other_unclear",
            "customer_message":
                "Thanks",
            "support_response":
                "@789012 You're welcome!"
        }
    ]

    print(
        generator.generate_reply(
            customer_message="My battery is draining quickly",
            intent="battery_power",
            historical_cases=test_cases
        )
    )