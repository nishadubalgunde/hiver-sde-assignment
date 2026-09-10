import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os


HUMAN_PATH = "results/human_eval_30.csv"
JUDGE_PATH = "results/llm_judge_results.csv"
OUTPUT_PATH = "results/judge_human_agreement.csv"


DIMENSIONS = [
    "groundedness",
    "relevance",
    "helpfulness",
    "safety",
    "overall"
]


def main():

    if not os.path.exists(HUMAN_PATH):
        print(f"Human evaluation file not found: {HUMAN_PATH}")
        return

    if not os.path.exists(JUDGE_PATH):
        print(f"LLM judge results not found: {JUDGE_PATH}")
        print()
        print("This is expected because the OpenAI API has no credits.")
        print("Run this script again after the LLM judge has produced results.")
        return

    human = pd.read_csv(HUMAN_PATH)
    judge = pd.read_csv(JUDGE_PATH)

    merged = human.merge(
        judge,
        on="tweet_id",
        suffixes=("_human", "_judge")
    )

    print("=" * 70)
    print("LLM JUDGE vs HUMAN AGREEMENT")
    print("=" * 70)

    print(f"\nExamples compared: {len(merged)}")

    results = []

    for dimension in DIMENSIONS:

        human_col = f"human_{dimension}"
        judge_col = f"judge_{dimension}"

        if human_col not in merged.columns:
            print(f"\nMissing human column: {human_col}")
            continue

        if judge_col not in merged.columns:
            print(f"\nMissing judge column: {judge_col}")
            continue

        subset = merged[[human_col, judge_col]].dropna()

        if len(subset) == 0:
            continue

        human_scores = subset[human_col].astype(int)
        judge_scores = subset[judge_col].astype(int)

        exact_agreement = (
            human_scores == judge_scores
        ).mean()

        within_one = (
            (human_scores - judge_scores).abs() <= 1
        ).mean()

        mean_absolute_error = (
            human_scores - judge_scores
        ).abs().mean()

        weighted_kappa = cohen_kappa_score(
            human_scores,
            judge_scores,
            weights="quadratic"
        )

        results.append({
            "dimension": dimension,
            "examples": len(subset),
            "exact_agreement": round(exact_agreement, 4),
            "within_1_agreement": round(within_one, 4),
            "mean_absolute_error": round(mean_absolute_error, 4),
            "weighted_kappa": round(weighted_kappa, 4)
        })

        print(f"\n{dimension.upper()}")
        print(f"Examples:             {len(subset)}")
        print(f"Exact agreement:      {exact_agreement:.2%}")
        print(f"Within 1 point:       {within_one:.2%}")
        print(f"Mean absolute error:  {mean_absolute_error:.2f}")
        print(f"Weighted Cohen kappa: {weighted_kappa:.3f}")

    if results:

        results_df = pd.DataFrame(results)

        os.makedirs("results", exist_ok=True)

        results_df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print("\n" + "=" * 70)
        print(f"Saved to: {OUTPUT_PATH}")
        print("=" * 70)


if __name__ == "__main__":
    main()