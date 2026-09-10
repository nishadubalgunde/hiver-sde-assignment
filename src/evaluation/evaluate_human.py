import pandas as pd

path = "results/human_eval_30.csv"

df = pd.read_csv(path)

score_columns = [
    "human_groundedness",
    "human_relevance",
    "human_helpfulness",
    "human_safety",
    "human_overall"
]

# Check that all scores are present
for column in score_columns:
    if df[column].isna().any():
        raise ValueError(f"Missing scores found in {column}")

print("=" * 70)
print("HUMAN EVALUATION RESULTS")
print("=" * 70)

print(f"\nExamples evaluated: {len(df)}")

print("\nAverage scores:")
for column in score_columns:
    average = df[column].mean()
    print(f"{column.replace('human_', '').title():20s}: {average:.2f} / 5")

print("\nOverall score distribution:")
print(df["human_overall"].value_counts().sort_index())

print("\nDecision-wise evaluation:")

decision_summary = (
    df.groupby("decision")[score_columns]
    .mean()
    .round(2)
)

print(decision_summary)

# Count low-quality replies
low_quality = df[df["human_overall"] <= 2]

print(f"\nLow-quality replies (Overall <= 2): {len(low_quality)}")

if len(low_quality) > 0:
    print("\nLow-quality examples:")
    print(
        low_quality[
            ["tweet_id", "decision", "human_overall"]
        ].to_string(index=False)
    )

# Save summary
summary = {
    "examples_evaluated": len(df),
    "average_groundedness": round(df["human_groundedness"].mean(), 2),
    "average_relevance": round(df["human_relevance"].mean(), 2),
    "average_helpfulness": round(df["human_helpfulness"].mean(), 2),
    "average_safety": round(df["human_safety"].mean(), 2),
    "average_overall": round(df["human_overall"].mean(), 2),
    "low_quality_count": len(low_quality)
}

summary_df = pd.DataFrame([summary])

output_path = "results/human_evaluation_summary.csv"
summary_df.to_csv(output_path, index=False)

print(f"\nSummary saved to: {output_path}")
print("\n" + "=" * 70)