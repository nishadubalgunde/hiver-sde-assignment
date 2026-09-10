import pandas as pd

MESSAGES = "data/processed/apple_conversations.csv"
GOLDEN = "data/golden/apple_golden_sample_200.csv"
OUTPUT = "data/golden/apple_golden_set_200.csv"

print("Loading conversation data...")
df = pd.read_csv(MESSAGES)

df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
df["conversation_id"] = pd.to_numeric(df["conversation_id"], errors="coerce")

golden = pd.read_csv(GOLDEN)

# Build lookup for all messages
conversation_lookup = (
    df.sort_values(["conversation_id", "position"])
    .groupby("conversation_id")
)

records = []

for _, target in golden.iterrows():

    conversation_id = target["conversation_id"]
    target_tweet_id = target["tweet_id"]

    if conversation_id not in conversation_lookup.groups:
        continue

    conversation = conversation_lookup.get_group(
        conversation_id
    ).copy()

    # Keep messages up to and including the target message
    context = conversation[
        conversation["tweet_id"] <= target_tweet_id
    ]

    context_lines = []

    for _, message in context.iterrows():

        if message["inbound"] == True:
            speaker = "CUSTOMER"
        else:
            speaker = "APPLESUPPORT"

        context_lines.append(
            f"{speaker}: {message['text']}"
        )

    records.append({
        "conversation_id": conversation_id,
        "tweet_id": target_tweet_id,
        "created_at": target["created_at"],
        "target_message": target["text"],
        "conversation_context": "\n".join(context_lines),
        "intent": "",
        "label_notes": ""
    })

result = pd.DataFrame(records)

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 60)
print("GOLDEN SET WITH CONTEXT CREATED")
print("=" * 60)

print(f"Examples: {len(result):,}")
print(f"Output: {OUTPUT}")

print()
print("Columns:")
print(list(result.columns))

print()
print("First example:")
if len(result) > 0:
    print(result.iloc[0]["conversation_context"])