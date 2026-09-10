import os
import pandas as pd


INPUT_PATH = "results/agent_predictions.csv"
OUTPUT_PATH = "results/human_eval_30.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    print(f"Total agent predictions: {len(df)}")

    auto = df[df["decision"] == "AUTO_HANDLE"].copy()
    escalate = df[df["decision"] == "ESCALATE"].copy()

    print(f"AUTO_HANDLE available: {len(auto)}")
    print(f"ESCALATE available: {len(escalate)}")

    # Keep every AUTO_HANDLE example.
    auto_sample = auto

    # Randomly select 13 escalated examples.
    escalate_sample = escalate.sample(
        n=min(13, len(escalate)),
        random_state=42
    )

    human_eval = pd.concat(
        [
            auto_sample,
            escalate_sample
        ],
        ignore_index=True
    )

    # Shuffle the final set so the evaluator does not see
    # all AUTO_HANDLE examples together.
    human_eval = human_eval.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Human evaluation fields.
    human_eval["human_groundedness"] = ""
    human_eval["human_relevance"] = ""
    human_eval["human_helpfulness"] = ""
    human_eval["human_safety"] = ""
    human_eval["human_overall"] = ""
    human_eval["human_notes"] = ""

    os.makedirs(
        "results",
        exist_ok=True
    )

    human_eval.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("HUMAN EVALUATION SET")
    print("=" * 70)

    print(f"\nExamples selected: {len(human_eval)}")

    print("\nDecision distribution:")
    print(
        human_eval["decision"].value_counts()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()