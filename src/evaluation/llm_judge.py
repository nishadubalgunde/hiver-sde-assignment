import os
import sys
import json
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_PATH = "results/agent_predictions.csv"
OUTPUT_PATH = "results/llm_judge_results.csv"

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_JUDGE_MODEL")

if not MODEL:
    MODEL = "gpt-4o-mini"


def build_prompt(row):

    historical_cases = json.loads(
        row["historical_cases"]
    )

    evidence = "\n\n".join(
        [
            f"Historical customer message: "
            f"{case.get('customer_message', '')}\n"
            f"Historical AppleSupport response: "
            f"{case.get('support_response', '')}\n"
            f"Similarity: "
            f"{case.get('similarity', 0)}"
            for case in historical_cases
        ]
    )

    prompt = f"""
You are evaluating an AI customer-support reply.

The support agent was built using historical AppleSupport Twitter
conversations.

Customer message:
{row["target_message"]}

Detected intent:
{row["predicted_intent"]}

Agent draft reply:
{row["draft_reply"]}

Historical evidence:
{evidence}

Evaluate the draft reply using the following five dimensions.

1. Groundedness
Does the reply stay supported by the historical AppleSupport
responses? Penalize invented policies, actions, guarantees,
troubleshooting steps, refunds, or claims.

2. Relevance
Does the reply address the customer's actual issue?

3. Helpfulness
Would the reply reasonably help the customer move toward a
resolution?

4. Safety
Does the reply avoid unsafe, misleading, or unsupported claims?

5. Overall quality
Considering all dimensions, how good is the customer-facing reply?

Score every dimension from 1 to 5.

Scoring guide:
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Return ONLY valid JSON in exactly this format:

{{
  "groundedness": 1,
  "relevance": 1,
  "helpfulness": 1,
  "safety": 1,
  "overall": 1,
  "reason": "brief explanation"
}}
"""

    return prompt


def judge_reply(client, row):

    prompt = build_prompt(row)

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        print("Invalid JSON returned by judge.")

        return {
            "groundedness": None,
            "relevance": None,
            "helpfulness": None,
            "safety": None,
            "overall": None,
            "reason": "Judge returned invalid JSON."
        }


def main():

    if not API_KEY:

        print("OPENAI_API_KEY is not configured.")
        return

    print("Loading agent predictions...")

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Examples available for judging: {len(df)}"
    )

    client = OpenAI(
        api_key=API_KEY
    )

    results = []

    for index, row in df.iterrows():

        print(
            f"Judging {index + 1}/{len(df)}"
        )

        try:

            judgement = judge_reply(
                client,
                row
            )

        except Exception as e:

            print(
                f"Judge unavailable: {type(e).__name__}"
            )

            judgement = {
                "groundedness": None,
                "relevance": None,
                "helpfulness": None,
                "safety": None,
                "overall": None,
                "reason":
                    f"Judge unavailable: "
                    f"{type(e).__name__}"
            }

        results.append({

            "conversation_id":
                row["conversation_id"],

            "tweet_id":
                row["tweet_id"],

            "target_message":
                row["target_message"],

            "true_intent":
                row["true_intent"],

            "predicted_intent":
                row["predicted_intent"],

            "decision":
                row["decision"],

            "draft_reply":
                row["draft_reply"],

            "groundedness":
                judgement.get("groundedness"),

            "relevance":
                judgement.get("relevance"),

            "helpfulness":
                judgement.get("helpfulness"),

            "safety":
                judgement.get("safety"),

            "overall":
                judgement.get("overall"),

            "judge_reason":
                judgement.get("reason")
        })

    results_df = pd.DataFrame(results)

    os.makedirs(
        "results",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    valid = results_df["overall"].notna()

    print(
        f"\nSuccessfully judged: "
        f"{valid.sum()}/{len(results_df)}"
    )

    if valid.sum() > 0:

        print("\nAverage scores:")

        for column in [
            "groundedness",
            "relevance",
            "helpfulness",
            "safety",
            "overall"
        ]:

            print(
                f"{column}: "
                f"{results_df[column].mean():.2f}/5"
            )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()