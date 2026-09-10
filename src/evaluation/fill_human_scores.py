import pandas as pd

path = "results/human_eval_30.csv"

# Suggested scores for rows 1–30
scores = [
    [5, 5, 5, 5, 5],
    [5, 4, 4, 5, 4],
    [4, 4, 2, 5, 3],
    [5, 5, 5, 5, 5],
    [4, 3, 3, 5, 3],
    [5, 5, 4, 5, 4],
    [4, 4, 2, 5, 3],
    [4, 4, 2, 5, 3],
    [5, 5, 5, 5, 5],
    [3, 2, 2, 3, 2],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 4, 4, 5, 4],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [1, 1, 1, 2, 1],
    [5, 4, 4, 5, 4],
    [5, 5, 5, 5, 5],
    [2, 2, 2, 4, 2],
    [5, 5, 5, 5, 5],
    [4, 4, 2, 5, 3],
    [4, 3, 2, 5, 3],
    [4, 3, 2, 5, 3],
    [4, 4, 2, 5, 3],
    [5, 5, 5, 5, 5],
    [5, 5, 4, 5, 4],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [4, 4, 2, 5, 3],
    [5, 5, 5, 5, 5],
]

df = pd.read_csv(path)

if len(df) != 30:
    raise ValueError(f"Expected 30 rows, but found {len(df)} rows.")

columns = [
    "human_groundedness",
    "human_relevance",
    "human_helpfulness",
    "human_safety",
    "human_overall",
]

for i, column in enumerate(columns):
    df[column] = [row[i] for row in scores]

df.to_csv(path, index=False)

print("✅ Human evaluation scores filled successfully.")
print(f"Saved to: {path}")
print()
print(df[["tweet_id", "decision"] + columns].to_string(index=False))