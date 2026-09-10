import pandas as pd

INPUT = "data/processed/apple_customer_messages.csv"
OUTPUT = "data/golden/apple_golden_sample_200.csv"

print("Loading Apple customer messages...")

df = pd.read_csv(INPUT)

df["text"] = df["text"].fillna("").str.strip()

# Remove empty messages
df = df[df["text"] != ""].copy()

print(f"Customer messages available: {len(df):,}")

# ---------------------------------------------------------
# Select conversations rather than isolated messages.
# This prevents excessive sampling of messages from the
# same conversation.
# ---------------------------------------------------------

conversation_ids = df["conversation_id"].drop_duplicates()

sample_conversations = conversation_ids.sample(
    n=200,
    random_state=42
)

sample = df[
    df["conversation_id"].isin(sample_conversations)
].copy()

# Keep only one customer message per selected conversation.
# Prefer the first customer message because it usually
# contains the main issue.
sample = (
    sample
    .sort_values(["conversation_id", "position"])
    .groupby("conversation_id", as_index=False)
    .first()
)

# ---------------------------------------------------------
# Add fields for manual annotation
# ---------------------------------------------------------

sample["intent"] = ""
sample["label_notes"] = ""

# Reorder columns
sample = sample[
    [
        "conversation_id",
        "position",
        "tweet_id",
        "created_at",
        "text",
        "intent",
        "label_notes",
    ]
]

sample.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("GOLDEN SET SAMPLE CREATED")
print("=" * 60)

print(f"Examples: {len(sample):,}")
print(f"Output: {OUTPUT}")

print()
print("Intent column is intentionally empty.")
print("These examples must be manually labelled.")